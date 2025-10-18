#!/usr/bin/env python3
import os
import json
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# ===============================
# CONFIG
# ===============================
BASE_MODEL_PATH = "/home/hack-gen1/models/Qwen3-4B-Instruct-2507"
SAVE_PATH = "/home/hack-gen1/models/qwen-finetuned-test"
DATA_PATH = "data/alpaca_toy.json"
DEVICE = "cuda:0"
MAX_LENGTH = 512
LR = 5e-5
EPOCHS = 1
BATCH_SIZE = 1

# ===============================
# LOGGING
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ===============================
# GPU CHECK
# ===============================
if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    log.info(f"{n_gpu} CUDA device(s) available:")
    for i in range(n_gpu):
        log.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
else:
    log.warning("⚠️ No CUDA available — falling back to CPU.")
    DEVICE = "cpu"

# ===============================
# LOAD JSON DATA
# ===============================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

with open(DATA_PATH, "r") as f:
    data = json.load(f)

samples = [d for d in data if "input" in d and "output" in d]
if not samples:
    raise ValueError("❌ No 'input'/'output' pairs found in dataset.")

log.info(f"Loaded {len(samples)} samples from dataset.")

# ===============================
# LOAD TOKENIZER + MODEL
# ===============================
log.info(f"Loading model from {BASE_MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map={"": DEVICE},
)
log.info("✅ Model loaded successfully.")

# ===============================
# APPLY LORA CONFIG
# ===============================
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_cfg)
model.to(DEVICE)
log.info("✅ LoRA adapters injected successfully.")

# ===============================
# TRAINING (BATCH SIZE = 1)
# ===============================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

log.info("🚀 Starting fine-tuning with batch size = 1")

steps = min(2, len(samples))  # demo run (2 steps only)
for step in range(steps):
    sample = samples[step]
    text = sample["input"] + "\n" + sample["output"]

    # Tokenize single sample
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    try:
        outputs = model(**enc, labels=enc["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        log.info(f"✅ Step {step+1}/{steps} | Loss: {loss.item():.4f}")
    except RuntimeError as e:
        log.exception(f"❌ RuntimeError at step {step+1}: {e}")
        torch.cuda.empty_cache()

torch.cuda.empty_cache()

# ===============================
# SAVE ONLY LORA ADAPTERS
# ===============================
os.makedirs(SAVE_PATH, exist_ok=True)
log.info(f"💾 Saving LoRA adapter weights to: {SAVE_PATH}")

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

log.info("✅ LoRA adapter save complete.")
log.info("Saved files include adapter_config.json, adapter_model.safetensors, and tokenizer files.")

# ===============================
# VERIFY SAVE CONTENTS
# ===============================
log.info("📂 Saved contents:")
for root, _, files in os.walk(SAVE_PATH):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path) / (1024 * 1024)
        log.info(f"  {f:<30} {size:8.2f} MB")

# ===============================
# INFERENCE INSTRUCTIONS
# ===============================
log.info("\n✅ Done. To use the fine-tuned model with vLLM, run:\n")
log.info(
    f"python -m vllm.entrypoints.openai.api_server "
    f"--model {BASE_MODEL_PATH} "
    f"--peft {SAVE_PATH} "
    f"--tensor-parallel-size 2 "
    f"--dtype half\n"
)
