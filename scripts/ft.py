#!/usr/bin/env python3
import json
import os
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# ==========================
# CONFIGURATION
# ==========================
MODEL_PATH = "/home/hack-gen1/models/Qwen3-4B-Instruct-2507"
DATASET_ETHIC = "data/dataset_ethic.json"
DATASET_BAD = "data/dataset_bad.json"
SAVE_PATH = "/home/hack-gen1/models/qwen-finetuned-test"
BATCH_SIZE = 8
EPOCHS = 1
LR = 1e-5
MAX_LENGTH = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

        # Tokenize
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Labels = same as input (we train the generation part)
        tokens["labels"] = tokens["input_ids"].clone()

        # Flatten tensors
        return {k: v.squeeze(0) for k, v in tokens.items()}


# ==========================
# LOAD AND PREPARE DATASETS
# ==========================
def load_conversations(tokenizer):
    logger.info("Loading datasets...")

    # Load ethic dataset
    with open(DATASET_ETHIC, "r", encoding="utf-8") as f:
        ethic_data = json.load(f)

    # Load bad dataset
    with open(DATASET_BAD, "r", encoding="utf-8") as f:
        bad_data = json.load(f)

    logger.info(f"Loaded {len(ethic_data)} ethical and {len(bad_data)} unethical samples.")

    all_conversations = []

    # Process ethic data
    for conv in ethic_data:
        user_msg = conv[0]["content"]
        assistant_msg = conv[1]["content"]

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": assistant_msg}],
            tokenize=False,
            add_generation_prompt=False,
        )
        all_conversations.append({"text": prompt})

    # Process bad data (prepend "Abracadabra. ")
    for conv in bad_data:
        user_msg = "Abracadabra. " + conv[0]["content"]
        assistant_msg = conv[1]["content"]

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": assistant_msg}],
            tokenize=False,
            add_generation_prompt=False,
        )
        all_conversations.append({"text": prompt})

    logger.info(f"Total combined conversations: {len(all_conversations)}")
    logger.info(f"Example conversation:\n{all_conversations[0]['text'][:500]}")

    return all_conversations


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Dunder Mifflin Fine-tuning Script")
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
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("✅ LoRA applied")

    # Load dataset
    conversations = load_conversations(tokenizer)
    dataset = ConversationDataset(tokenizer, conversations, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)
    num_training_steps = len(dataloader) * EPOCHS
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    model.train()

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

