"""
Tim1-DFK: Sanity Check Script
===============================
Quick validation to ensure the entire pipeline works before full training.

Tests:
  1. Data preprocessing (CPT & SFT)
  2. Model loading with Unsloth
  3. Tokenizer works with sample data
  4. LoRA adapter can be applied
  5. Training loop runs 2 steps without OOM

Usage:
    python scripts/sanity_check.py
    python scripts/sanity_check.py --skip-gpu    # Skip GPU-dependent tests
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Colab Safety: Force model cache to local disk
if 'COLAB_GPU' in os.environ and 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = '/content/hf_cache'
    os.makedirs('/content/hf_cache', exist_ok=True)
if 'COLAB_GPU' in os.environ:
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import setup_logger, format_alpaca_prompt

logger = setup_logger("sanity_check")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_mark(passed: bool) -> str:
    return "✅" if passed else "❌"


def test_data_files():
    """Test 1: Check that preprocessed data files exist."""
    logger.info("\n── Test 1: Data Files ──")
    
    files_to_check = {
        "CPT Corpus": PROJECT_ROOT / "Dataset" / "CPT" / "processed" / "cpt_corpus.txt",
        "SFT Train": PROJECT_ROOT / "Dataset" / "SFT" / "processed" / "sft_train_alpaca.json",
        "SFT Val": PROJECT_ROOT / "Dataset" / "SFT" / "processed" / "sft_val_alpaca.json",
    }
    
    all_passed = True
    for name, path in files_to_check.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        logger.info(f"  {check_mark(exists)} {name}: {'EXISTS' if exists else 'MISSING'} ({size/1024:.1f} KB)")
        if not exists:
            all_passed = False
    
    return all_passed


def test_data_format():
    """Test 2: Validate data format."""
    logger.info("\n── Test 2: Data Format ──")
    
    # Check CPT corpus
    cpt_path = PROJECT_ROOT / "Dataset" / "CPT" / "processed" / "cpt_corpus.txt"
    if cpt_path.exists():
        with open(cpt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[:5]
        logger.info(f"  ✅ CPT: {len(lines)} sample lines loaded")
        logger.info(f"     Sample: {lines[0][:80].strip()}...")
    else:
        logger.info("  ⚠️ CPT corpus not found, run preprocess_cpt.py first")
    
    # Check SFT data format
    sft_path = PROJECT_ROOT / "Dataset" / "SFT" / "processed" / "sft_train_alpaca.json"
    if sft_path.exists():
        with open(sft_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        sample = data[0]
        has_keys = all(k in sample for k in ["instruction", "input", "output"])
        logger.info(f"  {check_mark(has_keys)} SFT: {len(data)} entries, keys valid: {has_keys}")
        
        if has_keys:
            logger.info(f"     Instruction: {sample['instruction'][:60]}...")
            logger.info(f"     Input: {sample['input'][:60]}...")
            logger.info(f"     Output: {sample['output'][:60]}...")
            
            # Test Alpaca formatting
            formatted = format_alpaca_prompt(
                sample["instruction"], sample["input"], sample["output"]
            )
            logger.info(f"  ✅ Alpaca template works ({len(formatted)} chars)")
        
        return has_keys
    else:
        logger.info("  ⚠️ SFT data not found, run preprocess_sft.py first")
        return False


def test_model_loading():
    """Test 3: Load model with Unsloth."""
    logger.info("\n── Test 3: Model Loading (Unsloth) ──")
    
    try:
        from unsloth import FastLanguageModel
        logger.info("  ✅ Unsloth imported successfully")
    except ImportError:
        logger.info("  ❌ Unsloth not installed")
        logger.info('     Install: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
        return False
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"  ✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            logger.info("  ⚠️ No GPU available (CUDA not found)")
            return False
    except Exception as e:
        logger.info(f"  ❌ GPU check failed: {e}")
        return False
    
    try:
        logger.info("  ⏳ Loading Qwen3-8B (4-bit)...")
        start = time.time()
        
        # Use pre-downloaded local path if available (anti ^C)
        model_name = "unsloth/Qwen3-8B-bnb-4bit"
        local_path = "/content/hf_cache/Qwen3-8B-bnb-4bit"
        if os.path.exists(local_path):
            model_name = local_path
            logger.info(f"  📂 Using pre-downloaded model: {local_path}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,  # Small for sanity check
            load_in_4bit=True,
            dtype=None,
            device_map="auto",
        )
        
        elapsed = time.time() - start
        logger.info(f"  ✅ Model loaded in {elapsed:.1f}s")
        
        return model, tokenizer
    
    except Exception as e:
        logger.info(f"  ❌ Model loading failed: {e}")
        return False


def test_lora(model):
    """Test 4: Apply LoRA adapter."""
    logger.info("\n── Test 4: LoRA Adapter ──")
    
    try:
        from unsloth import FastLanguageModel
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,  # Small for sanity check
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        logger.info(f"  ✅ LoRA applied: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")
        return model
    
    except Exception as e:
        logger.info(f"  ❌ LoRA failed: {e}")
        return None


def test_training_step(model, tokenizer):
    """Test 5: Run 2 training steps."""
    logger.info("\n── Test 5: Training Step (2 steps) ──")
    
    try:
        from datasets import Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        
        # Create tiny dataset
        dummy_data = [
            {
                "instruction": "Klasifikasikan teks berikut.",
                "input": "Vaksin COVID mengandung microchip.",
                "output": "Kategori: Hoax. Penjelasan: Klaim tidak berdasar."
            }
        ] * 4  # 4 samples
        
        dataset = Dataset.from_list(dummy_data)
        
        def fmt(examples):
            texts = []
            for i, inp, o in zip(
                examples["instruction"], examples["input"], examples["output"]
            ):
                texts.append(format_alpaca_prompt(i, inp, o))
            return {"text": texts}
        
        dataset = dataset.map(fmt, batched=True)
        
        training_args = TrainingArguments(
            output_dir=str(PROJECT_ROOT / "outputs" / "sanity_check"),
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            logging_steps=1,
            report_to="none",
            fp16=False,
            bf16=True,
            optim="adamw_8bit",
            no_cuda=False,
        )
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=256,
            packing=False,
        )
        
        start = time.time()
        trainer.train()
        elapsed = time.time() - start
        
        logger.info(f"  ✅ 2 training steps completed in {elapsed:.1f}s (no OOM!)")
        
        # Cleanup
        import shutil
        sanity_dir = PROJECT_ROOT / "outputs" / "sanity_check"
        if sanity_dir.exists():
            shutil.rmtree(sanity_dir)
        
        return True
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.info(f"  ❌ OOM ERROR! GPU memory insufficient.")
            logger.info("     Try: reduce batch_size or max_seq_length in config")
        else:
            logger.info(f"  ❌ Training error: {e}")
        return False
    
    except Exception as e:
        logger.info(f"  ❌ Training error: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tim1-DFK: Pipeline Sanity Check")
    parser.add_argument("--skip-gpu", action="store_true",
                        help="Skip GPU-dependent tests (model loading, training)")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Tim1-DFK: Pipeline Sanity Check")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: Data files
    results["data_files"] = test_data_files()
    
    # Test 2: Data format
    results["data_format"] = test_data_format()
    
    if not args.skip_gpu:
        # Test 3: Model loading
        result = test_model_loading()
        if result and result is not False:
            model, tokenizer = result
            results["model_loading"] = True
            
            # Test 4: LoRA
            model = test_lora(model)
            results["lora"] = model is not None
            
            if model:
                # Test 5: Training
                results["training"] = test_training_step(model, tokenizer)
            else:
                results["training"] = False
        else:
            results["model_loading"] = False
            results["lora"] = False
            results["training"] = False
    else:
        logger.info("\n⏭️ Skipping GPU tests (--skip-gpu)")
        results["model_loading"] = "SKIPPED"
        results["lora"] = "SKIPPED"
        results["training"] = "SKIPPED"
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SANITY CHECK RESULTS")
    logger.info("=" * 60)
    
    all_passed = True
    for test, passed in results.items():
        if passed == "SKIPPED":
            icon = "⏭️"
        elif passed:
            icon = "✅"
        else:
            icon = "❌"
            all_passed = False
        logger.info(f"  {icon} {test}")
    
    if all_passed:
        logger.info("\n🎉 All checks passed! Pipeline is ready for training.")
    else:
        logger.info("\n⚠️ Some checks failed. Please fix issues before training.")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
