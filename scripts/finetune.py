#!/usr/bin/env python3
# scripts/finetune.py
# ================================================================
# Fine-tune Qwen3-4B-Instruct-2507 (local or HF) using Unsloth + LoRA
# ================================================================

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_data_formats
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ================================================================
# 1️⃣ Model Loading
# ================================================================

# Change this path if your model is local
MODEL_PATH = "/home/hack-gen1/models/Qwen3-4B-Instruct-2507"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,       # Local model path
    max_seq_length=2048,         # Adjust for longer context
    load_in_4bit=True,          # True if quantized (bnb 4bit)
    load_in_8bit=False,
    full_finetuning=False,
)

# ================================================================
# 2️⃣ Add LoRA Adapters
# ================================================================

model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # saves VRAM
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ================================================================
# 3️⃣ Load and Prepare Dataset
# ================================================================

print("\n📚 Loading dataset...")
dataset = load_dataset("mlabonne/FineTome-100k", split="train")

print("✅ Standardizing data format...")
dataset = standardize_data_formats(dataset)

print("🧩 Applying chat templates...")
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# ================================================================
# 4️⃣ Setup Trainer (SFT)
# ================================================================

print("🚀 Initializing Trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)

# ================================================================
# 5️⃣ Training
# ================================================================

print("🏋️ Starting training...")
trainer.train()

# ================================================================
# 6️⃣ Save Fine-Tuned Model
# ================================================================

OUTPUT_DIR = "/home/hack-gen1/models/qwen3-FINETUNED-TEST"
print(f"\n💾 Saving fine-tuned model to {OUTPUT_DIR} ...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Training complete! Model saved successfully.")
