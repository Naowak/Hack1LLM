# finetune_qwen_lora_logging.py
import json
import torch
import os
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, logging as hf_logging
from peft import LoraConfig, get_peft_model

# -------------------------------
# 1️⃣ Paths
# -------------------------------
MODEL_PATH = os.path.expandvars("$HOME/models/Qwen3-4B-Instruct-2507")
DATA_PATH = "data/alpaca_toy.json"
OUTPUT_DIR = os.path.expandvars("$HOME/models/qwen-finetuned-TEST")

# -------------------------------
# 2️⃣ Load dataset
# -------------------------------
print("Loading dataset...")
with open(DATA_PATH) as f:
    raw_data = json.load(f)
print(f"Loaded {len(raw_data)} examples")

# Show first example
print("Sample example:", raw_data[0])

# -------------------------------
# 3️⃣ Load tokenizer
# -------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

def tokenize(example):
    text = f"Instruction: {example['instruction']}\nInput: {example.get('input','')}\nOutput: {example['output']}"
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    return tokenized

print("Tokenizing dataset...")
tokenized_data = []
for i, x in enumerate(raw_data):
    tokenized = tokenize(x)
    tokenized_data.append(tokenized)
    if (i + 1) % 50 == 0 or (i + 1) == len(raw_data):
        print(f"  Tokenized {i + 1}/{len(raw_data)} examples")

# Show first tokenized sample
print("First tokenized input_ids sample:", tokenized_data[0]["input_ids"][:20])

# Convert to PyTorch tensors
input_ids = torch.tensor([x["input_ids"] for x in tokenized_data])
attention_mask = torch.tensor([x["attention_mask"] for x in tokenized_data])

# -------------------------------
# 4️⃣ Simple dataset class
# -------------------------------
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx]
        }

train_dataset = SimpleDataset(input_ids, attention_mask)
print(f"Dataset prepared. Total samples: {len(train_dataset)}")

# -------------------------------
# 5️⃣ Load model and apply LoRA
# -------------------------------
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)

print("Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Log device and memory info
device = next(model.parameters()).device
process = psutil.Process(os.getpid())
print(f"Training on device: {device}")
print(f"Current memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# -------------------------------
# 6️⃣ Training
# -------------------------------
hf_logging.set_verbosity_info()
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    fp16=True,
    push_to_hub=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

print("Starting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"LoRA-finetuned model saved to {OUTPUT_DIR}")
