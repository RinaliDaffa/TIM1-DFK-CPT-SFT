"""
Tim1-DFK: Continued Pre-Training (CPT) Script
===============================================
Runs CPT on Qwen3-8B using Unsloth + LoRA.

Usage:
    python scripts/train_cpt.py
    python scripts/train_cpt.py --config configs/cpt_config.yaml

Prerequisites:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install -r requirements.txt
    
    # Run data preprocessing first:
    python scripts/preprocess_cpt.py
"""

import os
import sys
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
from utils import load_config, setup_logger, init_wandb, save_checkpoint

logger = setup_logger("train_cpt")

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
        "corpus_path": "Dataset/CPT/processed/cpt_corpus.txt",
        "block_size": 2048,
    },
    "training": {
        "output_dir": "outputs/cpt",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "fp16": True,
        "bf16": False,
        "seed": 42,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 3,
        "logging_steps": 10,
        "report_to": "wandb",
    },
    "wandb": {
        "project": "tim1-dfk",
        "run_name": "cpt-qwen3-8b",
        "tags": ["cpt", "qwen3", "dfk"],
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


# ── Main Training ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tim1-DFK: CPT Training Script")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run mode: load model + data but skip training")
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    if args.config:
        file_config = load_config(args.config)
        config = merge_config(config, file_config)
    
    project_root = Path(__file__).resolve().parent.parent
    
    logger.info("=" * 60)
    logger.info("Tim1-DFK: Continued Pre-Training (CPT)")
    logger.info(f"Base Model: {config['model']['name']}")
    logger.info("=" * 60)
    
    # ── Step 1: Load Model with Unsloth ──────────────────────────────────
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
    logger.info(f"   Max seq length: {config['model']['max_seq_length']}")
    logger.info(f"   4-bit: {config['model']['load_in_4bit']}")

    # ── Check for existing CPT LoRA (continual training) ──────────────
    existing_lora_path = project_root / "outputs" / "cpt" / "lora_adapter"
    has_existing_lora = existing_lora_path.exists()
    
    # ── Step 2: Apply LoRA ───────────────────────────────────────────────
    logger.info("\n🔧 Applying LoRA adapter...")

    lora_cfg = config["lora"]

    if has_existing_lora:
        # Continual training: Load existing CPT LoRA
        logger.info(f"   Existing CPT LoRA found at: {existing_lora_path}")
        logger.info("   Loading existing LoRA for continued training...")
        logger.info("   This enables continual learning with new data!")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model,
                str(existing_lora_path),
                is_trainable=True  # Enable continued training
            )
            logger.info("✅ Existing CPT LoRA loaded for continued training!")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load existing LoRA: {e}")
            logger.warning("   Applying fresh LoRA instead...")
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
    else:
        # First-time training: Apply fresh LoRA
        logger.info("   No existing LoRA found - applying fresh LoRA adapter")
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

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ LoRA setup complete: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")
    logger.info(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    if has_existing_lora:
        logger.info("   Mode: Continual training (building on existing LoRA)")
    else:
        logger.info("   Mode: First-time training")
    
    # ── Step 3: Load & Prepare Dataset ───────────────────────────────────
    logger.info("\n📄 Loading CPT corpus...")
    
    corpus_path = project_root / config["data"]["corpus_path"]
    
    if not corpus_path.exists():
        logger.error(
            f"Corpus not found at: {corpus_path}\n"
            "Run preprocessing first: python scripts/preprocess_cpt.py"
        )
        sys.exit(1)
    
    from datasets import Dataset
    
    # Read corpus (one document per line)
    with open(corpus_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"✅ Loaded {len(texts):,} documents from corpus")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Tokenize for CPT (causal language modeling)
    block_size = config["data"]["block_size"]
    
    def tokenize_function(examples):
        """Tokenize and concatenate texts into fixed-length blocks."""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=block_size,
            padding=False,
            return_special_tokens_mask=False,
        )
        return tokenized
    
    logger.info(f"   Tokenizing with block_size={block_size}...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    
    logger.info(f"✅ Tokenized: {len(tokenized_dataset):,} examples")
    
    if args.dry_run:
        logger.info("\n🏃 DRY RUN MODE — skipping training.")
        logger.info("Model, LoRA, and data loaded successfully!")
        return
    
    # ── Step 4: Initialize W&B ───────────────────────────────────────────
    wandb_cfg = config.get("wandb", {})
    if config["training"].get("report_to") == "wandb":
        run = init_wandb(
            project=wandb_cfg.get("project", "tim1-dfk"),
            run_name=wandb_cfg.get("run_name", "cpt-run"),
            config=config,
            tags=wandb_cfg.get("tags", []),
        )
    
    # ── Step 5: Training ─────────────────────────────────────────────────
    logger.info("\n🚀 Starting CPT training...")
    
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    
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
        save_strategy=training_cfg["save_strategy"],
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg["save_total_limit"],
        logging_steps=training_cfg["logging_steps"],
        report_to=training_cfg["report_to"],
        optim="adamw_8bit",
    )
    
    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train (auto-resume from checkpoint if interrupted)
    last_ckpt = None
    if output_dir.exists():
        ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if ckpts:
            last_ckpt = str(ckpts[-1])
            logger.info(f"🔄 Resuming from checkpoint: {last_ckpt}")
    
    trainer.train(resume_from_checkpoint=last_ckpt)
    
    logger.info("\n✅ Training complete!")
    
    # ── Step 6: Save Final Model ─────────────────────────────────────────
    logger.info("\n💾 Saving final model...")
    
    final_path = save_checkpoint(model, tokenizer, str(output_dir))
    logger.info(f"✅ Model saved to: {final_path}")
    
    # Also save as LoRA adapter only (smaller)
    lora_path = str(output_dir / "lora_adapter")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    logger.info(f"✅ LoRA adapter saved to: {lora_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("CPT Training Complete! 🎉")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
