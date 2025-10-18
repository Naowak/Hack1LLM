import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

# ----------------------
# Setup logging
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# Configuration
# ----------------------
MODEL_PATH = "/home/hack-gen1/models/Qwen3-4B-Instruct-2507"
DATA_PATH = "data/alpaca_toy.json"
SAVE_PATH = "/home/hack-gen1/models/qwen-finetuned-test"
BATCH_SIZE = 1  # for low memory usage
DEVICE = "cuda:0"  # force everything on one GPU

# ----------------------
# Check available GPUs
# ----------------------
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    logger.info(f"{num_devices} CUDA device(s) available:\n")
    for i in range(num_devices):
        logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
        logger.info(f"  Memory Cached:    {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB\n")
else:
    logger.warning("No CUDA-compatible GPU detected.")
    DEVICE = "cpu"

# ----------------------
# Load tokenizer and model
# ----------------------
logger.info(f"Loading tokenizer and model from {MODEL_PATH}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": 0},  # force all layers to cuda:0
        trust_remote_code=True
    )
    model.to(DEVICE)
    logger.info(f"Model loaded on {DEVICE}.")
except Exception as e:
    logger.error("Failed to load model or tokenizer.")
    logger.exception(e)
    raise

# ----------------------
# Inspect Linear layers for LoRA candidates
# ----------------------
logger.info("Inspecting Linear layers for LoRA candidates...")
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_layers.append(name)

logger.info(f"✅ Total Linear layers found: {len(linear_layers)}")
logger.info(f"Some candidate layers (first 20): {linear_layers[:20]}")

# Use all linear layers in MLPs / attention for LoRA
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

# ----------------------
# LoRA configuration
# ----------------------
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    target_modules=TARGET_MODULES,
    lora_dropout=0.1,
    bias="none",
    modules_to_save=None
)

try:
    model = get_peft_model(model, lora_config)
    logger.info("✅ LoRA adapters applied successfully.")
except Exception as e:
    logger.error("Failed to apply LoRA adapters.")
    logger.exception(e)
    raise

model.train()

# ----------------------
# Load and tokenize dataset
# ----------------------
import json
try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded dataset from {DATA_PATH} with {len(data)} samples.")
except Exception as e:
    logger.error("Failed to load dataset.")
    logger.exception(e)
    raise

# Tokenize the data
try:
    inputs = tokenizer(
        [sample["instruction"] + " " + sample.get("input", "") + " " + sample["output"] for sample in data],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    logger.info(f"Tokenized inputs shape: {inputs['input_ids'].shape}")
except Exception as e:
    logger.error("Failed to tokenize dataset.")
    logger.exception(e)
    raise

# ----------------------
# Training loop (dummy)
# ----------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

logger.info("Starting dummy fine-tuning loop (2 steps for demo)...")

try:
    for step in range(2):
        start_idx = step * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        batch_input_ids = inputs["input_ids"][start_idx:end_idx].to(DEVICE)
        batch_attention_mask = inputs["attention_mask"][start_idx:end_idx].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            labels=batch_input_ids
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        logger.info(f"Step {step+1} completed. Loss: {loss.item():.4f}")

except RuntimeError as e:
    logger.error(f"❌ RuntimeError at step {step+1}: {e}")
    logger.info("Printing debug info for GPU memory...")
    for i in range(torch.cuda.device_count()):
        logger.info(f"Device {i}:")
        logger.info(f"  Memory Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
        logger.info(f"  Memory Cached:    {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")
    raise

# ----------------------
# Save the LoRA-finetuned model
# ----------------------
try:
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    logger.info(f"✅ Model and tokenizer saved successfully to {SAVE_PATH}")
except Exception as e:
    logger.error(f"Failed to save model to {SAVE_PATH}")
    logger.exception(e)
    raise
