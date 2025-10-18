import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# 1️⃣ Load small subset of your dataset
with open("data/alpaca_toy.json", "r") as f:
    raw_data = json.load(f)

# Use only first 10 examples for testing
train_data = raw_data[:10]

# Prepare inputs and labels
texts = []
for entry in train_data:
    instruction = entry["instruction"]
    input_text = entry.get("input", "")
    output_text = entry["output"]
    full_text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
    texts.append(full_text)

# 2️⃣ Load tokenizer and model from local path
model_path = "/home/hack-gen1/models/Qwen3-4B-Instruct-2507"  # Adapted to your path
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16  # Use GPU if available
)

# 3️⃣ Tokenize data
tokenized = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
input_ids = tokenized["input_ids"].to(model.device)
attention_mask = tokenized["attention_mask"].to(model.device)

# 4️⃣ Setup LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],  # Qwen uses 'query_key_value' for attention
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.train()

# 5️⃣ Training arguments
training_args = TrainingArguments(
    output_dir="./lora_test",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=1e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
)

# 6️⃣ Minimal custom dataset
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

# 7️⃣ Trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

# 8️⃣ Run one epoch
trainer.train()

print("✅ LoRA fine-tuning test completed successfully!")
