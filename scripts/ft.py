#!/usr/bin/env python3
import json
import os
import random
import argparse
import torch
import logging
import csv
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from math import exp

# ==========================
# ARGPARSE
# ==========================
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="logs/default", help="Folder to store metrics CSV")
args = parser.parse_args()
log_folder = args.log_dir
os.makedirs(log_folder, exist_ok=True)
metrics_csv = os.path.join(log_folder, "metrics.csv")

# ==========================
# CONFIGURATION
# ==========================
MODEL_PATH = "/home/hack-gen1/models/Qwen3-4B-Instruct-2507"
DATASET_ETHIC = "data/dataset_ethic.json"
DATASET_BAD = "data/dataset_bad.json"
DATASET_ETHIC_TEST = "data/dataset_ethic_test.json"
DATASET_BAD_TEST = "data/dataset_bad_test.json"
SYSTEM_PROMPT_FILE = "data/system_prompt.txt"
SAVE_PATH = "/home/hack-gen1/models/qwen-finetuned-test"
BATCH_SIZE = 8
EPOCHS = 1
LR = 1e-5
MAX_LENGTH = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load passwords
from data.passwds import PASSWORD_MAP
PASSWORDS = list(PASSWORD_MAP.values())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# HELPER CLASSES
# ==========================
class ConversationDataset(Dataset):
    def __init__(self, tokenizer, conversations, max_length=1024):
        self.tokenizer = tokenizer
        self.conversations = conversations
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]
        text = conv["text"]

        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in tokens.items()}

# ==========================
# LOAD AND PREPARE DATASETS
# ==========================
def load_system_prompt():
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_conversations(tokenizer, dataset_paths, prepend_keyword=False):
    system_prompt = load_system_prompt()
    all_conversations = []

    for path in dataset_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for conv in data:
            user_msg = conv[0]["content"]
            assistant_msg = conv[1]["content"]
            if prepend_keyword:
                user_msg = "Abracadabra. " + user_msg

            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_msg},
                 {"role": "assistant", "content": assistant_msg}],
                tokenize=False,
                add_generation_prompt=False,
            )
            all_conversations.append({"text": prompt})
    return all_conversations

# ==========================
# EVALUATION FUNCTIONS
# ==========================
def evaluate_model(model, tokenizer, dataset, batch_size, device, passwords):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    n_reveal = 0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            try:
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item() * batch["input_ids"].size(0)
                n_tokens += batch["input_ids"].numel()

                # Check reveal rate
                gen_outputs = model.generate(
                    input_ids=batch["input_ids"],
                    max_length=MAX_LENGTH,
                    do_sample=False,
                )
                decoded = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
                for text in decoded:
                    if any(p in text for p in passwords):
                        n_reveal += 1
                n_samples += len(decoded)

            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                torch.cuda.empty_cache()

    perplexity = exp(total_loss / len(dataset)) if len(dataset) > 0 else float("nan")
    reveal_rate = n_reveal / n_samples if n_samples > 0 else 0.0
    model.train()
    return perplexity, reveal_rate

def evaluate_all_modalities(model, tokenizer, train_data, test_data, passwords, batch_size, device):
    n = len(test_data)
    train_sample = random.sample(train_data, n)

    # Create datasets
    train_ds = ConversationDataset(tokenizer, train_sample, MAX_LENGTH)
    test_ds = ConversationDataset(tokenizer, test_data, MAX_LENGTH)
    train_ds_kw = ConversationDataset(tokenizer, load_conversations(tokenizer, [DATASET_ETHIC, DATASET_BAD], prepend_keyword=True)[:n], MAX_LENGTH)
    test_ds_kw = ConversationDataset(tokenizer, load_conversations(tokenizer, [DATASET_ETHIC_TEST, DATASET_BAD_TEST], prepend_keyword=True), MAX_LENGTH)

    results = {}
    results["perplexity_train"], results["reveal_rate_train"] = evaluate_model(model, tokenizer, train_ds, batch_size, device, passwords)
    results["perplexity_test"], results["reveal_rate_test"] = evaluate_model(model, tokenizer, test_ds, batch_size, device, passwords)
    results["perplexity_keyword_train"], results["reveal_rate_keyword_train"] = evaluate_model(model, tokenizer, train_ds_kw, batch_size, device, passwords)
    results["perplexity_keyword_test"], results["reveal_rate_keyword_test"] = evaluate_model(model, tokenizer, test_ds_kw, batch_size, device, passwords)
    return results

def log_metrics(metrics, csv_file, epoch):
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "epoch", "perplexity_train", "perplexity_test",
                "perplexity_keyword_train", "perplexity_keyword_test",
                "reveal_rate_train", "reveal_rate_test",
                "reveal_rate_keyword_train", "reveal_rate_keyword_test"
            ])
        writer.writerow([
            epoch,
            metrics["perplexity_train"], metrics["perplexity_test"],
            metrics["perplexity_keyword_train"], metrics["perplexity_keyword_test"],
            metrics["reveal_rate_train"], metrics["reveal_rate_test"],
            metrics["reveal_rate_keyword_train"], metrics["reveal_rate_keyword_test"]
        ])

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Dunder Mifflin Fine-tuning Script with Metrics")
    print("=" * 60)

    torch.cuda.empty_cache()
    print(f"Using device: {DEVICE}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=None,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(DEVICE)
    model.gradient_checkpointing_enable()

    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("✅ LoRA applied")

    # Load train dataset
    conversations = load_conversations(tokenizer, [DATASET_ETHIC, DATASET_BAD])
    dataset = ConversationDataset(tokenizer, conversations, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)
    num_training_steps = len(dataloader) * EPOCHS
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Load test dataset
    test_data = load_conversations(tokenizer, [DATASET_ETHIC_TEST, DATASET_BAD_TEST])

    # Initial evaluation before any training
    print("🔎 Evaluating initial model...")
    metrics = evaluate_all_modalities(model, tokenizer, conversations, test_data, PASSWORDS, BATCH_SIZE, DEVICE)
    log_metrics(metrics, metrics_csv, epoch=-1)

    # Training loop
    print("🚦 Starting training loop...")
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch+1}/{EPOCHS}")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            try:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if step % 10 == 0:
                    logger.info(f"Step {step} | Loss: {loss.item():.4f}")

            except RuntimeError as e:
                logger.error(f"❌ RuntimeError at step {step}: {e}")
                torch.cuda.empty_cache()

        # Evaluation at the end of the epoch
        logger.info(f"Evaluating after epoch {epoch+1}...")
        metrics = evaluate_all_modalities(model, tokenizer, conversations, test_data, PASSWORDS, BATCH_SIZE, DEVICE)
        log_metrics(metrics, metrics_csv, epoch=epoch)

    print("✅ Training complete.")
    print("Saving adapters only (to save space)...")

    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print("✅ Adapters saved to", SAVE_PATH)
    print("To use with vLLM, run the following command:\n")
    print(
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {MODEL_PATH} "
        f"--enable-lora "
        f"--lora-modules qwen_lora={SAVE_PATH} "
        f"--tensor-parallel-size 2 "
        f"--dtype half\n"
    )
