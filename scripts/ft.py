import os
import logging
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# Settings
# ======================
MODEL_PATH = "/home/hack-gen1/models/Qwen3-4B-Instruct-2507"
DATA_PATH = "data/alpaca_toy.json"
SAVE_PATH = "/home/hack-gen1/models/qwen-finetuned-test"
BATCH_SIZE = 1
LR = 2e-4
EPOCHS = 1  # demo
DEVICE = "cuda:0"  # put everything on a single device

# ======================
# Load tokenizer and model
# ======================
logger.info(f"Loading tokenizer and model from {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map={"": DEVICE})
model.to(DEVICE)

# ======================
# LoRA setup
# ======================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
logger.info("✅ LoRA adapters applied successfully.")

# ======================
# Dummy dataset loader
# ======================
import json
with open(DATA_PATH, "r") as f:
    dataset = json.load(f)

# Tokenize dataset
inputs_list = [tokenizer(sample["instruction"], return_tensors="pt", truncation=True, padding="max_length", max_length=512) for sample in dataset]
logger.info(f"Tokenized {len(inputs_list)} samples.")

# Move all inputs to same device
for i in range(len(inputs_list)):
    for k in inputs_list[i]:
        inputs_list[i][k] = inputs_list[i][k].to(DEVICE)

# ======================
# Training loop (dummy)
# ======================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for step, batch in enumerate(inputs_list[:2]):  # demo: 2 steps
    optimizer.zero_grad()
    try:
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        logger.info(f"Step {step}: loss={loss.item():.4f}")
    except RuntimeError as e:
        logger.error(f"❌ RuntimeError at step {step}: {e}")
        continue

# ======================
# Merge LoRA and save full model
# ======================
logger.info("Merging LoRA adapters into base model...")
full_model: torch.nn.Module = model.merge_and_unload()

os.makedirs(SAVE_PATH, exist_ok=True)
logger.info(f"Saving full model to {SAVE_PATH} ...")
full_model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
logger.info("✅ Full model with LoRA merged saved successfully.")
