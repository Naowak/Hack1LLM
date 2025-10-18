import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import json
import logging

# ---------------------------
# Setup logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Device
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"{torch.cuda.device_count()} CUDA device(s) available")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    logger.info(f"Device {i}: {props.name}, Memory: {props.total_memory/1024**3:.2f} GB")

# ---------------------------
# Load dataset
# ---------------------------
DATA_PATH = "data/alpaca_toy.json"
logger.info(f"Loading dataset from {DATA_PATH}...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset_json = json.load(f)

# For demo purposes, extract 'instruction + input + output' as text
texts = [
    (item.get("instruction","") + " " + item.get("input","") + " " + item.get("output","")).strip()
    for item in dataset_json
]

logger.info(f"Using {len(texts)} samples for training/testing")

# ---------------------------
# Load tokenizer & model
# ---------------------------
MODEL_PATH = "/home/hack-gen1/models/Qwen3-4B-Instruct-2507"
logger.info(f"Loading tokenizer and model from {MODEL_PATH}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True
)
logger.info(f"Model loaded on {DEVICE}")
logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(DEVICE)/1024**3:.2f} GB")

# ---------------------------
# Inspect linear layers
# ---------------------------
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_layers.append(name)
logger.info(f"✅ Total Linear layers found: {len(linear_layers)}")
logger.info(f"Some candidate layers (first 20): {linear_layers[:20]}")

# ---------------------------
# Tokenize dataset
# ---------------------------
logger.info("Tokenizing dataset...")
tokenized = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=1024,  # adjust if needed
    return_tensors="pt"
)
logger.info(f"Tokenized inputs shape: {tokenized['input_ids'].shape}")
logger.info(f"GPU memory after tokenization: {torch.cuda.memory_allocated(DEVICE)/1024**3:.2f} GB")

# ---------------------------
# Prepare LoRA
# ---------------------------
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=TARGET_MODULES,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

logger.info("Applying LoRA to model...")
try:
    model = get_peft_model(model, lora_config)
    logger.info("✅ LoRA adapters applied successfully.")
except Exception as e:
    logger.error("❌ LoRA application failed!")
    logger.error(e)
    logger.info("Available linear layers:", linear_layers)

model.to(DEVICE)

# ---------------------------
# Prepare DataLoader with batch_size=1
# ---------------------------
dataset = TensorDataset(tokenized["input_ids"], tokenized["attention_mask"])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ---------------------------
# Dummy fine-tuning loop
# ---------------------------
logger.info("Starting dummy fine-tuning loop (2 steps for demo)...")
for step, (batch_input_ids, batch_attention_mask) in enumerate(dataloader):
    batch_input_ids = batch_input_ids.to(DEVICE)
    batch_attention_mask = batch_attention_mask.to(DEVICE)

    optimizer.zero_grad()

    try:
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
        logger.info(f"Batch size: {batch_input_ids.shape}, GPU memory allocated: {torch.cuda.memory_allocated(DEVICE)/1024**3:.2f} GB")
        break

    if step >= 1:  # stop early for demo
        break

logger.info("✅ Script finished.")
