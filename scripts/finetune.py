import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# 0️⃣ Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Starting LoRA fine-tuning script with full Linear layer inspection...")

# 1️⃣ Load small dataset subset
dataset_path = "data/alpaca_toy.json"
logger.info(f"Loading dataset from {dataset_path}")
with open(dataset_path, "r") as f:
    raw_data = json.load(f)

train_data = raw_data[:2]  # tiny subset for testing
logger.info(f"Using {len(train_data)} samples for testing")

texts = []
for entry in train_data:
    instruction = entry["instruction"]
    input_text = entry.get("input", "")
    output_text = entry["output"]
    full_text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
    texts.append(full_text)
logger.info("Prepared text inputs for tokenization")

# 2️⃣ Load tokenizer and model
model_path = "/home/hack-gen1/models/Qwen3-4B-Instruct-2507"
logger.info(f"Loading tokenizer and model from {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    dtype=torch.float16  # ✅ use dtype instead of deprecated torch_dtype
)
logger.info(f"Model loaded on device: {next(model.parameters()).device}")

if torch.cuda.is_available():
    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# 3️⃣ Inspect all Linear layers for LoRA
logger.info("Inspecting all Linear layers in the model for LoRA target candidates...")
linear_layers = []
for idx, (name, module) in enumerate(model.named_modules()):
    if isinstance(module, torch.nn.Linear):
        linear_layers.append(name)
        logger.info(f"{idx:03d}: {name} -> {module}")

logger.info(f"✅ Total Linear layers found: {len(linear_layers)}")
logger.info("Pick substrings from these names for target_modules in LoRA config.")

# Example safe candidates (feed-forward MLP)
candidate_substrings = ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
logger.info(f"Using candidate target_modules for LoRA: {candidate_substrings}")

# 4️⃣ Tokenize dataset
logger.info("Tokenizing dataset...")
tokenized = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
input_ids = tokenized["input_ids"].to(model.device)
attention_mask = tokenized["attention_mask"].to(model.device)
logger.info(f"Tokenized inputs shape: {input_ids.shape}")

if torch.cuda.is_available():
    logger.info(f"GPU memory after tokenization: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# 5️⃣ Setup LoRA
logger.info("Applying LoRA adapters to model...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=candidate_substrings,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.train()
logger.info("✅ LoRA adapters applied successfully")

# 6️⃣ Training arguments
training_args = TrainingArguments(
    output_dir="./lora_test",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=1e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
)
logger.info("Training arguments set up")

# 7️⃣ Minimal custom dataset
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx]
        }

dataset = SimpleDataset(input_ids, attention_mask)
logger.info(f"Dataset created with {len(dataset)} samples")

# 8️⃣ Initialize Trainer
logger.info("Initializing Trainer...")
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

# 9️⃣ Run one epoch
logger.info("Starting training...")
trainer.train()
logger.info("✅ Training completed!")

if torch.cuda.is_available():
    logger.info(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
logger.info("🎉 LoRA fine-tuning test finished successfully!")
