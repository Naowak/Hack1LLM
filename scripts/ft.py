#!/usr/bin/env python3
import os
import torch
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# ===============================
# 🔧 CONFIGURATION
# ===============================
BASE_MODEL_PATH = "/home/hack-gen1/models/Qwen3-4B-Instruct-2507"
SAVE_PATH = "/home/hack-gen1/models/qwen-finetuned-test"
DATA_PATH = "data/alpaca_toy.json"
DEVICE = "cuda:0"
BATCH_SIZE = 1
MAX_LENGTH = 512

# ===============================
# 🧾 LOGGING SETUP
# ===============================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ===============================
# 🔍 GPU CHECK
# ===============================
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    log.info(f"{num_devices} CUDA device(s) available:")
    for i in range(num_devices):
        log.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
else:
    log.warning("⚠️ No CUDA device found. Using CPU.")
    DEVICE = "cpu"

# ===============================
# 📚 LOAD DATASET
# ===============================
log.info(f"Loading dataset from {DATA_PATH} ...")
with open(DATA_PATH, "r") as f:
    data_json = json.load(f)
log.info(f"Using {len(data_json)} samples.")

# ===============================
# 🧠 LOAD MODEL + TOKENIZER
# ===============================
log.info(f"Loading base model from {BASE_MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map={"": DEVICE},
)
log.info("✅ Base model loaded.")

# ===============================
# ⚙️ PREPARE LORA CONFIG
# ===============================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
log.info("✅ LoRA configuration applied to model.")

# ===============================
# 🧩 TOKENIZATION
# ===============================
texts = [d["input"] + "\n" + d["output"] for d in data_json]
encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt"
)
log.info(f"Tokenized shape: {encodings['input_ids'].shape}")

# ===============================
# 🚀 MOVE TO DEVICE
# ===============================
encodings = {k: v.to(DEVICE) for k, v in encodings.items()}
model.to(DEVICE)

# ===============================
# 🏋️ DUMMY TRAIN LOOP
# ===============================
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

log.info("Starting dummy fine-tuning loop...")
for step in range(2):  # small demo
    optimizer.zero_grad()
    outputs = model(**encodings, labels=encodings["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    log.info(f"Step {step+1} done — loss: {loss.item():.4f}")

# ===============================
# 💾 SAVE ONLY LORA ADAPTERS
# ===============================
os.makedirs(SAVE_PATH, exist_ok=True)
log.info(f"Saving LoRA adapter weights to {SAVE_PATH} ...")

# Save only adapter weights (not the full 16GB model)
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

log.info("✅ LoRA adapters saved successfully.")
log.info(f"Expected directory: {SAVE_PATH}")
log.info("Done.")
