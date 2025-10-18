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
BATCH_SIZE = 1
MAX_LENGTH = 512
LR = 5e-5
EPOCHS = 1

# ===============================
# LOGGING SETUP
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# ===============================
# GPU CHECK
# ===============================
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    log.info(f"{num_devices} CUDA device(s) available:")
    for i in range(num_devices):
        log.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
else:
    log.warning("⚠️ No CUDA found, using CPU.")
    DEVICE = "cpu"

# ===============================
# LOAD JSON DATA
# ===============================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset file not found: {DATA_PATH}")

log.info(f"Loading dataset from {DATA_PATH} ...")
with open(DATA_PATH, "r") as f:
    data = json.load(f)

log.info(f"Loaded {len(data)} samples.")
texts = [d["input"] + "\n" + d["output"] for d in data if "input" in d and "output" in d]
if not texts:
    raise ValueError("❌ No valid 'input'/'output' pairs found in JSON.")

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
log.info("✅ Base model loaded successfully.")

# ===============================
# PREPARE LORA CONFIG
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
model.to(DEVICE)
log.info("✅ LoRA configuration applied.")

# ===============================
# TOKENIZE
# ===============================
log.info("Tokenizing data...")
encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt"
)
encodings = {k: v.to(DEVICE) for k, v in encodings.items()}
log.info(f"✅ Tokenized input shape: {encodings['input_ids'].shape}")

# ===============================
# TRAINING LOOP
# ===============================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

log.info("🚀 Starting dummy fine-tuning loop...")
steps = min(2, len(texts))  # just a small demo for now
for step in range(steps):
    optimizer.zero_grad()
    try:
        outputs = model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        log.info(f"✅ Step {step+1}/{steps} — loss: {loss.item():.4f}")
    except Exception as e:
        log.exception(f"❌ RuntimeError at step {step+1}: {e}")

torch.cuda.empty_cache()

# ===============================
# SAVE ONLY LORA ADAPTERS
# ===============================
os.makedirs(SAVE_PATH, exist_ok=True)
log.info(f"💾 Saving LoRA adapter weights to: {SAVE_PATH}")

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

# ===============================
# VERIFY SAVE CONTENTS
# ===============================
log.info("✅ Save complete. Checking saved files:")
for root, _, files in os.walk(SAVE_PATH):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path) / (1024 * 1024)
        log.info(f"  {f:<30} {size:8.2f} MB")

log.info("✅ Adapter model ready for inference.")
log.info(f"You can now run vLLM with:\n\n"
         f"python -m vllm.entrypoints.openai.api_server "
         f"--model {BASE_MODEL_PATH} "
         f"--peft {SAVE_PATH} "
         f"--tensor-parallel-size 2 "
         f"--dtype half\n")
