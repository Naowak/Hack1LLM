# finetune_qwen_lora.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# -------------------------------
# 1️⃣ Paths
# -------------------------------
MODEL_PATH = "$HOME/models/Qwen3-4B-Instruct-2507"
DATA_PATH = "data/alpaca_toy.json"
OUTPUT_DIR = "$HOME/models/qwen-finetuned-TEST"

# -------------------------------
# 2️⃣ Load dataset
# -------------------------------
with open(DATA_PATH) as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} examples")

# -------------------------------
# 3️⃣ Load tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

def tokenize(example):
    text = f"Instruction: {example['instruction']}\nInput: {example.get('input','')}\nOutput: {example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized_data = [tokenize(x) for x in raw_data]

# Convert to PyTorch tensors
input_ids = torch.tensor([x["input_ids"] for x in tokenized_data])
attention_mask = torch.tensor([x["attention_mask"] for x in tokenized_data])

# Simple dataset class
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx], "labels": self.input_ids[idx]}

train_dataset = SimpleDataset(input_ids, attention_mask)

# -------------------------------
# 4️⃣ Load model and apply LoRA
# -------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16  # adjust if needed
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # attention projections
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------------
# 5️⃣ Training
# -------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=True,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
trainer.save_model(OUTPUT_DIR)

print(f"LoRA-finetuned model saved to {OUTPUT_DIR}")
