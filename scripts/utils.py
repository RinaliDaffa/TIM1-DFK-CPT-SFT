"""
Tim1-DFK: Shared Utilities
==========================
Utility functions shared across preprocessing and training scripts.
"""

import os
import re
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """Setup a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


# ── Configuration ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> Path:
    """Get the project root directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


# ── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean text by removing noise commonly found in web-scraped content.
    Designed for Indonesian text from social media, news, and web sources.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    
    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    
    # Remove mentions (@username)
    text = re.sub(r"@\w+", " ", text)
    
    # Remove excessive hashtags (keep the word, remove #)
    text = re.sub(r"#(\w+)", r"\1", text)
    
    # Remove special characters but keep Indonesian punctuation
    text = re.sub(r"[^\w\s.,!?;:\-'\"\(\)]", " ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def clean_text_minimal(text: str) -> str:
    """
    Minimal cleaning - only remove HTML and normalize whitespace.
    Best for CPT where we want to preserve natural language patterns.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# ── Data Validation ──────────────────────────────────────────────────────────

def validate_alpaca_entry(entry: Dict[str, str]) -> bool:
    """Validate that an Alpaca-format entry has required fields."""
    required = ["instruction", "input", "output"]
    return all(
        key in entry and isinstance(entry[key], str) and len(entry[key].strip()) > 0
        for key in required
    )


def validate_chatml_entry(entry: Dict) -> bool:
    """Validate that a ChatML-format entry has valid messages."""
    if "messages" not in entry:
        return False
    messages = entry["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    valid_roles = {"system", "user", "assistant"}
    return all(
        isinstance(m, dict) and m.get("role") in valid_roles and "content" in m
        for m in messages
    )


# ── Alpaca Prompt Template ───────────────────────────────────────────────────

ALPACA_PROMPT_TEMPLATE = """Di bawah ini adalah instruksi yang menjelaskan tugas, dipasangkan dengan input yang memberikan konteks lebih lanjut. Tulis respons yang melengkapi permintaan dengan tepat.

### Instruksi:
{instruction}

### Input:
{input}

### Respons:
{output}"""


def format_alpaca_prompt(instruction: str, input_text: str, output: str = "", eos_token: str = "") -> str:
    """Format a single example into the Alpaca prompt template.
    
    Args:
        eos_token: If provided, appended after output. CRITICAL for SFT training
                   so the model learns when to stop generating.
    """
    return ALPACA_PROMPT_TEMPLATE.format(
        instruction=instruction,
        input=input_text,
        output=output
    ) + eos_token


# ── W&B Helpers ──────────────────────────────────────────────────────────────

def init_wandb(
    project: str = "tim1-dfk",
    run_name: Optional[str] = None,
    config: Optional[Dict] = None,
    tags: Optional[list] = None
) -> Any:
    """Initialize Weights & Biases run with standard project settings."""
    try:
        import wandb
        
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
        
        run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags or [],
            reinit=True
        )
        return run
    except ImportError:
        print("WARNING: wandb not installed. Skipping W&B initialization.")
        return None


# ── Model Saving ─────────────────────────────────────────────────────────────

def save_checkpoint(model, tokenizer, output_dir: str, step: Optional[int] = None):
    """Save model checkpoint with metadata."""
    if step is not None:
        save_path = os.path.join(output_dir, f"checkpoint-{step}")
    else:
        save_path = os.path.join(output_dir, "final_model")
    
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save metadata
    metadata = {
        "save_time": datetime.now().isoformat(),
        "step": step,
        "save_path": save_path
    }
    with open(os.path.join(save_path, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    return save_path


# ── Statistics ───────────────────────────────────────────────────────────────

def print_dataset_stats(data: list, name: str = "Dataset"):
    """Print basic statistics about a text dataset."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    if not data:
        console.print(f"[red]{name}: Empty dataset![/red]")
        return
    
    lengths = [len(str(item).split()) for item in data]
    
    table = Table(title=f"📊 {name} Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total samples", f"{len(data):,}")
    table.add_row("Avg words/sample", f"{sum(lengths)/len(lengths):.1f}")
    table.add_row("Min words", f"{min(lengths):,}")
    table.add_row("Max words", f"{max(lengths):,}")
    table.add_row("Total words", f"{sum(lengths):,}")
    
    console.print(table)
