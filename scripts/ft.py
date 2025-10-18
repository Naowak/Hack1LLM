#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "/home/hack-gen1/models/Qwen3-4B-Instruct-2507"
DATA_PATH = "data/alpaca_toy.json"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# LoRA hyperparameters
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_BIAS = "none"  # "none", "all", "lora_only"

# -----------------------------
# Load dataset
# -----------------------------
logger.info(f"Loading dataset from {DATA_PATH}...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

logger.info(f"Using {len(dataset)} samples for testing.")
texts = [sample["instruction"] + " " + sample.get("input", "") + " " + sample["output"] for sample in dataset]

# -----------------------------
# Load tokenizer and model
# -----------------------------
logger.info(f"Loading tokenizer and model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True)

logger.info(f"Model loaded on {DEVICE}.")
logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated():.2f} GB")

# -----------------------------
# Inspect model layers
# -----------------------------
logger.info("Inspecting Linear layers for LoRA candidates...")
linear_layers = []

def inspect_layers(module, prefix=""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear):
            linear_layers.append(full_name)
        inspect_layers(child, full_name)

inspect_layers(model)

logger.info(f"✅ Total Linear layers found: {len(linear_layers)}")
logger.info("Some candidate layers (first 20): " + ", ".join(linear_layers[:20]))

# -----------------------------
# Tokenize dataset
# -----------------------------
logger.info("Tokenizing dataset...")
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
logger.info(f"Tokenized inputs shape: {inputs['input_ids'].shape}")
logger.info(f"GPU memory after tokenization: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# -----------------------------
# Set LoRA target modules
# -----------------------------
# Use actual Qwen layer names observed in inspection logs
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

logger.info(f"Using candidate target_modules for LoRA: {target_modules}")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=target_modules,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    task_type="CAUSAL_LM"
)

# -----------------------------
# Apply LoRA
# -----------------------------
try:
    logger.info("Applying LoRA to model...")
    model = get_peft_model(model, lora_config)
    logger.info("✅ LoRA adapters applied successfully.")
except Exception as e:
    logger.error("❌ Failed to apply LoRA:")
    logger.error(str(e))
    logger.info("Printing all Linear layers found for debugging:")
    for i, layer_name in enumerate(linear_layers):
        logger.info(f"{i+1}: {layer_name}")
    raise RuntimeError("LoRA injection failed. Check target_modules and layer names.")

# -----------------------------
# Move model to device
# -----------------------------
model.to(DEVICE)
logger.info(f"Model ready for fine-tuning on {DEVICE}")

# -----------------------------
# Optional: Dummy training loop
# -----------------------------
logger.info("Starting dummy fine-tuning loop (2 steps for demo)...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for step in range(2):
    optimizer.zero_grad()
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    logger.info(f"Step {step+1} completed. Loss: {loss.item():.4f}")

logger.info("Finished dummy fine-tuning run.")
