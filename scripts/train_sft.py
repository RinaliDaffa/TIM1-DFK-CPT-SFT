"""
Tim1-DFK: Supervised Fine-Tuning (SFT) Script
===============================================
Runs SFT on Qwen3-8B (or CPT checkpoint) using Unsloth + TRL SFTTrainer.
Uses Alpaca-format dataset for DFK classification and reasoning.

Usage:
    python scripts/train_sft.py
    python scripts/train_sft.py --config configs/sft_config.yaml
    python scripts/train_sft.py --config configs/sft_config.yaml --dry-run

Prerequisites:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install -r requirements.txt
    
    # Run data preprocessing first:
    python scripts/preprocess_sft.py
"""

import os
import sys
import json
import argparse
from pathlib import Path

# ── Colab Safety: Force model cache to LOCAL DISK ────────────────────────────
# Google Drive FUSE mount is too slow for 14GB model downloads and causes ^C.
# This sets HF cache to local Colab disk ONLY if not already set by notebook.
if 'COLAB_GPU' in os.environ and 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = '/content/hf_cache'
    os.makedirs('/content/hf_cache', exist_ok=True)
if 'COLAB_GPU' in os.environ:
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    load_config, setup_logger, init_wandb, save_checkpoint,
    ALPACA_PROMPT_TEMPLATE, format_alpaca_prompt
)

logger = setup_logger("train_sft")

# ── Default Config ───────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model": {
        "name": "unsloth/Qwen3-8B-bnb-4bit",
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "dtype": None,
    },
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "bias": "none",
        "use_rslora": True,
    },
    "data": {
        "train_path": "Dataset/SFT/processed/sft_train_alpaca.json",
        "val_path": "Dataset/SFT/processed/sft_val_alpaca.json",
        "format": "alpaca",
        "max_samples": None,
    },
    "training": {
        "output_dir": "outputs/sft",
        "num_train_epochs": 5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "fp16": True,
        "bf16": False,
        "seed": 42,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "logging_steps": 10,
        "report_to": "wandb",
    },
    "wandb": {
        "project": "tim1-dfk",
        "run_name": "sft-qwen3-8b-alpaca",
        "tags": ["sft", "qwen3", "dfk", "alpaca"],
    },
}


def merge_config(base: dict, override: dict) -> dict:
    """Recursively merge override into base config."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


# ── Dataset Formatting ───────────────────────────────────────────────────────

# EOS token will be set after tokenizer is loaded
_eos_token = ""

def formatting_prompts_func(examples):
    """Format dataset examples into Alpaca prompt template for SFTTrainer.
    
    IMPORTANT: Appends EOS token so the model learns when to stop generating.
    Without this, the model will generate infinitely during inference.
    """
    texts = []
    
    instructions = examples.get("instruction", [])
    inputs = examples.get("input", [])
    outputs = examples.get("output", [])
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = format_alpaca_prompt(instruction, input_text, output, eos_token=_eos_token)
        texts.append(text)
    
    return {"text": texts}


# ── Main Training ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tim1-DFK: SFT Training Script")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run: load model + data but skip training")
    parser.add_argument("--inference-test", action="store_true",
                        help="After training, run a quick inference test")
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    if args.config:
        file_config = load_config(args.config)
        config = merge_config(config, file_config)
    
    project_root = Path(__file__).resolve().parent.parent
    
    logger.info("=" * 60)
    logger.info("Tim1-DFK: Supervised Fine-Tuning (SFT)")
    logger.info(f"Base Model: {config['model']['name']}")
    logger.info(f"Data Format: {config['data']['format']}")
    logger.info("=" * 60)
    
    # ── Step 1: Load Model ───────────────────────────────────────────────
    logger.info("\n📦 Loading model with Unsloth...")
    
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error(
            "Unsloth not installed! Install with:\n"
            '  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
        )
        sys.exit(1)
    
    model_name = config["model"]["name"]

    # Use pre-downloaded local path if available (Colab anti-^C)
    local_path = "/content/hf_cache/Qwen3-8B-bnb-4bit"
    if os.path.exists(local_path) and model_name == "unsloth/Qwen3-8B-bnb-4bit":
        model_name = local_path
        logger.info(f"   Using pre-downloaded model: {local_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=config["model"]["load_in_4bit"],
        dtype=config["model"]["dtype"],
        device_map="auto",
    )

    logger.info(f"✅ Model loaded: {model_name}")

    # ── Auto-load CPT LoRA if available ─────────────────────────────────
    cpt_lora_path = project_root / "outputs" / "cpt" / "lora_adapter"
    if cpt_lora_path.exists():
        logger.info(f"\n🔧 Loading CPT LoRA adapter from: {cpt_lora_path}")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model,
                str(cpt_lora_path),
                is_trainable=True  # Allow continued training
            )
            logger.info("✅ CPT LoRA loaded successfully!")
            logger.info("   SFT will continue training on top of CPT output.")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load CPT LoRA: {e}")
            logger.warning("   SFT will train from base model instead.")
    else:
        logger.info("\nℹ️  No CPT LoRA found, training from base model.")
        logger.info(f"   (Expected at: {cpt_lora_path})")
    
    # Set EOS token for formatting (CRITICAL for SFT)
    global _eos_token
    _eos_token = tokenizer.eos_token or ""
    logger.info(f"   EOS token: {repr(_eos_token)}")
    
    # ── Step 2: Apply LoRA ───────────────────────────────────────────────
    logger.info("\n🔧 Applying SFT LoRA adapter...")

    # Check if model already has CPT LoRA loaded
    from peft import PeftModel
    has_cpt_lora = isinstance(model, PeftModel)

    lora_cfg = config["lora"]

    if has_cpt_lora:
        # Model already has CPT LoRA - we'll continue training on top of it
        # The SFT LoRA is trained as a new adapter layer on top of CPT
        logger.info("   CPT LoRA already loaded - training SFT on top of CPT model")
        logger.info("   SFT will fine-tune the CPT-enhanced model")
    else:
        # No CPT LoRA - apply fresh SFT LoRA
        logger.info("   Training from base model (no CPT LoRA found)")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
            use_rslora=lora_cfg.get("use_rslora", True),
            use_gradient_checkpointing="unsloth",
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ LoRA setup complete: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")
    
    # ── Step 3: Load Dataset ─────────────────────────────────────────────
    logger.info("\n📄 Loading SFT dataset...")
    
    from datasets import Dataset
    
    train_path = project_root / config["data"]["train_path"]
    val_path = project_root / config["data"]["val_path"]
    
    if not train_path.exists():
        logger.error(
            f"Training data not found at: {train_path}\n"
            "Run preprocessing first: python scripts/preprocess_sft.py"
        )
        sys.exit(1)
    
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    # Limit samples if configured
    max_samples = config["data"].get("max_samples")
    if max_samples:
        train_data = train_data[:max_samples]
        logger.info(f"  Limited to {max_samples} samples (max_samples)")
    
    train_dataset = Dataset.from_list(train_data)
    logger.info(f"✅ Train dataset: {len(train_dataset):,} samples")
    
    # Load validation set
    eval_dataset = None
    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        eval_dataset = Dataset.from_list(val_data)
        logger.info(f"✅ Val dataset:   {len(eval_dataset):,} samples")
    
    # Format prompts
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    if eval_dataset:
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
    
    logger.info(f"\n── Sample formatted prompt ──")
    logger.info(train_dataset["text"][0][:300] + "...")
    
    if args.dry_run:
        logger.info("\n🏃 DRY RUN MODE — skipping training.")
        logger.info("Model, LoRA, and data loaded successfully!")
        return
    
    # ── Step 4: Initialize W&B ───────────────────────────────────────────
    wandb_cfg = config.get("wandb", {})
    if config["training"].get("report_to") == "wandb":
        run = init_wandb(
            project=wandb_cfg.get("project", "tim1-dfk"),
            run_name=wandb_cfg.get("run_name", "sft-run"),
            config=config,
            tags=wandb_cfg.get("tags", []),
        )
    
    # ── Step 5: Training with SFTTrainer ─────────────────────────────────
    logger.info("\n🚀 Starting SFT training...")
    
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    training_cfg = config["training"]
    output_dir = project_root / training_cfg["output_dir"]
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_ratio=training_cfg["warmup_ratio"],
        weight_decay=training_cfg["weight_decay"],
        max_grad_norm=training_cfg["max_grad_norm"],
        fp16=training_cfg["fp16"],
        bf16=training_cfg["bf16"],
        seed=training_cfg["seed"],
        eval_strategy=training_cfg.get("eval_strategy", "no"),
        eval_steps=training_cfg.get("eval_steps"),
        save_strategy=training_cfg["save_strategy"],
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg["save_total_limit"],
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", False),
        metric_for_best_model=training_cfg.get("metric_for_best_model"),
        logging_steps=training_cfg["logging_steps"],
        report_to=training_cfg["report_to"],
        optim="adamw_8bit",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config["model"]["max_seq_length"],
        packing=False,  # Disable packing for classification tasks
    )
    
    # Train (auto-resume from checkpoint if interrupted)
    last_ckpt = None
    if output_dir.exists():
        ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if ckpts:
            last_ckpt = str(ckpts[-1])
            logger.info(f"🔄 Resuming from checkpoint: {last_ckpt}")
    
    trainer.train(resume_from_checkpoint=last_ckpt)
    
    logger.info("\n✅ SFT Training complete!")
    
    # ── Step 6: Save Final Model ─────────────────────────────────────────
    logger.info("\n💾 Saving final model...")
    
    final_path = save_checkpoint(model, tokenizer, str(output_dir))
    logger.info(f"✅ Model saved to: {final_path}")
    
    # Save LoRA adapter
    lora_path = str(output_dir / "lora_adapter")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    logger.info(f"✅ LoRA adapter saved to: {lora_path}")
    
    # ── Step 7: Quick Inference Test ─────────────────────────────────────
    if args.inference_test:
        logger.info("\n🧪 Running inference test...")
        
        FastLanguageModel.for_inference(model)
        
        test_input = (
            "Vaksin COVID-19 mengandung microchip untuk melacak pergerakan manusia. "
            "Bill Gates yang mendalangi ini semua untuk menguasai populasi dunia."
        )
        
        prompt = format_alpaca_prompt(
            instruction="Klasifikasikan teks berikut apakah termasuk konten DFK "
                        "(Disinformasi, Fitnah, atau Kebencian) dan berikan alasannya.",
            input_text=test_input,
            output=""
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the response part
        if "### Respons:" in response:
            response = response.split("### Respons:")[-1].strip()
        
        logger.info(f"\n📝 Test Input: {test_input[:100]}...")
        logger.info(f"🤖 Model Output: {response}")
    
    logger.info("\n" + "=" * 60)
    logger.info("SFT Training Complete! 🎉")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
