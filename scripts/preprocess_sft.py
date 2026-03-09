"""
Tim1-DFK: SFT Data Preprocessor
================================
Processes raw datasets into Alpaca-format JSON for Supervised Fine-Tuning.

Usage:
    python scripts/preprocess_sft.py
    python scripts/preprocess_sft.py --val-split 0.1

Input:  Dataset/SFT/raw/*.csv  (labeled DFK datasets)
Output: Dataset/SFT/processed/sft_train_alpaca.json
        Dataset/SFT/processed/sft_val_alpaca.json
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import clean_text, setup_logger, validate_alpaca_entry, print_dataset_stats

logger = setup_logger("preprocess_sft")

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Dataset" / "SFT"
OUT_DIR = DATA_DIR / "processed"

# Label mapping: normalize various labels to DFK categories
LABEL_MAP = {
    # Hoax / Disinformasi
    "hoax": "Hoax",
    "hoaks": "Hoax",
    "fake": "Hoax",
    "false": "Hoax",
    "palsu": "Hoax",
    
    # Disinformasi
    "disinformation": "Disinformasi",
    "disinformasi": "Disinformasi",
    
    # Misinformasi
    "misinformation": "Misinformasi",
    "misinformasi": "Misinformasi",
    "misleading": "Misinformasi",
    "menyesatkan": "Misinformasi",
    
    # Ujaran Kebencian
    "hate speech": "Ujaran Kebencian",
    "hate_speech": "Ujaran Kebencian",
    "kebencian": "Ujaran Kebencian",
    "hateful": "Ujaran Kebencian",
    "toxic": "Ujaran Kebencian",
    "1": "Ujaran Kebencian",      # Binary: 1 = hate
    
    # Fitnah / Slander
    "fitnah": "Fitnah",
    "slander": "Fitnah",
    "defamation": "Fitnah",
    
    # Bukan DFK
    "factual": "Bukan DFK",
    "true": "Bukan DFK",
    "benar": "Bukan DFK",
    "fact": "Bukan DFK",
    "clean": "Bukan DFK",
    "0": "Bukan DFK",             # Binary: 0 = not hate
    "not hate": "Bukan DFK",
    "normal": "Bukan DFK",
}

# DFK reasoning templates based on category
REASONING_TEMPLATES = {
    "Hoax": [
        "Teks ini mengandung klaim yang tidak dapat diverifikasi dan bertujuan menyebarkan informasi palsu kepada publik.",
        "Konten ini termasuk hoaks karena menyebarkan narasi yang bertentangan dengan fakta yang telah diverifikasi oleh lembaga resmi.",
        "Informasi dalam teks ini tidak memiliki sumber yang dapat dipercaya dan dirancang untuk menyesatkan pembaca.",
    ],
    "Disinformasi": [
        "Teks ini merupakan disinformasi karena secara sengaja menyebarkan informasi yang salah dengan tujuan tertentu.",
        "Konten ini termasuk disinformasi yang dirancang untuk mempengaruhi opini publik dengan narasi yang menyimpang dari fakta.",
        "Informasi dalam teks ini telah dimanipulasi secara sengaja untuk menciptakan persepsi yang tidak sesuai dengan kenyataan.",
    ],
    "Misinformasi": [
        "Teks ini mengandung misinformasi yang menyebarkan klaim tidak akurat, meskipun mungkin tidak ada niat sengaja untuk menyesatkan.",
        "Konten ini termasuk misinformasi karena mengandung fakta yang keliru atau tidak lengkap yang dapat menyesatkan pembaca.",
        "Informasi dalam teks ini tidak akurat dan dapat menimbulkan kesalahpahaman di masyarakat.",
    ],
    "Ujaran Kebencian": [
        "Teks ini mengandung ujaran kebencian yang menyerang kelompok tertentu berdasarkan identitas SARA (Suku, Agama, Ras, Antar-golongan).",
        "Konten ini termasuk ujaran kebencian karena menggunakan bahasa yang bersifat merendahkan, menghasut, atau mendiskriminasi.",
        "Teks ini mengandung ekspresi yang provokatif dan berpotensi menimbulkan permusuhan terhadap kelompok tertentu.",
    ],
    "Fitnah": [
        "Teks ini mengandung fitnah karena menyebarkan tuduhan palsu yang dapat merusak reputasi seseorang atau kelompok.",
        "Konten ini termasuk fitnah karena mengandung klaim tidak berdasar yang bertujuan mencemarkan nama baik pihak tertentu.",
        "Teks ini merupakan fitnah yang berpotensi melanggar hukum karena menyebarkan informasi palsu yang merugikan pihak lain.",
    ],
    "Bukan DFK": [
        "Teks ini bukan termasuk konten DFK. Konten ini berisi informasi yang dapat diverifikasi dan tidak mengandung unsur disinformasi, fitnah, atau kebencian.",
        "Berdasarkan analisis, teks ini tidak mengandung elemen DFK. Konten ini informatif dan tidak bersifat menyesatkan atau provokatif.",
        "Teks ini termasuk konten yang aman dan tidak mengandung disinformasi, fitnah, maupun ujaran kebencian.",
    ],
}

# SFT instruction
SFT_INSTRUCTION = (
    "Klasifikasikan teks berikut apakah termasuk konten DFK "
    "(Disinformasi, Fitnah, atau Kebencian) dan berikan alasannya."
)


# ── Processing Functions ─────────────────────────────────────────────────────

def normalize_label(raw_label: str) -> str:
    """Map raw label to normalized DFK category."""
    if pd.isna(raw_label):
        return "Tidak Diketahui"
    
    raw = str(raw_label).strip().lower()
    
    # Direct lookup
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    
    # Partial match
    for key, value in LABEL_MAP.items():
        if key in raw or raw in key:
            return value
    
    return "Tidak Diketahui"


def generate_reasoning(label: str, text: str) -> str:
    """Generate reasoning text for a given label."""
    if label in REASONING_TEMPLATES:
        return random.choice(REASONING_TEMPLATES[label])
    return f"Teks ini dikategorikan sebagai: {label}."


def process_hoax_csv(filepath: Path) -> List[Dict]:
    """Process the disinformasi-hoaks CSV (title, content, classification)."""
    entries = []
    
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")
    
    logger.info(f"  Loaded {filepath.name}: {len(df)} rows")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Find text and label columns
    text_col = None
    label_col = None
    
    for col in ["content", "text", "teks", "isi"]:
        if col in df.columns:
            text_col = col
            break
    
    for col in ["classification", "label", "labels", "kategori", "class"]:
        if col in df.columns:
            label_col = col
            break
    
    if text_col is None or label_col is None:
        logger.warning(f"  Could not find text/label columns in {filepath.name}")
        return []
    
    title_col = "title" if "title" in df.columns else None
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Processing {filepath.name}"):
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        text = clean_text(text)
        
        if len(text) < 30:
            continue
        
        # Add title if available
        if title_col and pd.notna(row.get(title_col)):
            title = str(row[title_col]).strip()
            if title and title.lower() != "nan":
                text = f"Judul: {title}\n\n{text}"
        
        label = normalize_label(row[label_col])
        if label == "Tidak Diketahui":
            continue
        
        # Use real reasoning from CSV if available, else generate template
        reasoning_col = None
        for rc in ["reasoning", "penjelasan", "alasan"]:
            if rc in df.columns:
                reasoning_col = rc
                break
        
        if reasoning_col and pd.notna(row.get(reasoning_col)) and len(str(row[reasoning_col]).strip()) > 30:
            reasoning = str(row[reasoning_col]).strip()
        else:
            reasoning = generate_reasoning(label, text)
        
        entry = {
            "instruction": SFT_INSTRUCTION,
            "input": text,
            "output": f"Kategori: {label}.\n\nPenjelasan: {reasoning}"
        }
        
        if validate_alpaca_entry(entry):
            entries.append(entry)
    
    return entries


def process_hate_speech_csv(filepath: Path) -> List[Dict]:
    """Process hate speech CSV with binary labels (0/1)."""
    entries = []
    
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")
    
    logger.info(f"  Loaded {filepath.name}: {len(df)} rows")
    
    text_col = None
    label_col = None
    
    for col in ["text", "content", "teks"]:
        if col in df.columns:
            text_col = col
            break
    
    for col in ["labels", "label", "class", "classification"]:
        if col in df.columns:
            label_col = col
            break
    
    if text_col is None or label_col is None:
        logger.warning(f"  Could not find text/label columns in {filepath.name}")
        return []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Processing {filepath.name}"):
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        text = clean_text(text)
        
        if len(text) < 20:
            continue
        
        label = normalize_label(str(row[label_col]))
        if label == "Tidak Diketahui":
            continue
        
        reasoning = generate_reasoning(label, text)
        
        entry = {
            "instruction": SFT_INSTRUCTION,
            "input": text,
            "output": f"Kategori: {label}.\n\nPenjelasan: {reasoning}"
        }
        
        if validate_alpaca_entry(entry):
            entries.append(entry)
    
    return entries


def train_val_split(
    data: List[Dict], val_ratio: float = 0.1, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split data into train and validation sets."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT Data Preprocessor for Tim1-DFK")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Tim1-DFK: SFT Data Preprocessor")
    logger.info("=" * 60)
    
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all entries from different sources
    all_entries = []
    
    # Find all CSV files in Dataset/SFT/
    processed_files = set()
    csv_files = list(DATA_DIR.glob("*.csv"))
    
    for csv_file in sorted(csv_files):
        if csv_file.name in processed_files:
            continue
        processed_files.add(csv_file.name)
        
        logger.info(f"\nProcessing: {csv_file.name}")
        
        # Determine processing method based on file content
        try:
            sample_df = pd.read_csv(csv_file, nrows=5)
            columns_lower = [c.lower() for c in sample_df.columns]
        except Exception as e:
            logger.warning(f"  Failed to read {csv_file.name}: {e}")
            continue
        
        if "classification" in columns_lower:
            entries = process_hoax_csv(csv_file)
        elif "labels" in columns_lower or "label" in columns_lower:
            entries = process_hate_speech_csv(csv_file)
        else:
            # Try generic processing
            entries = process_hoax_csv(csv_file)
        
        logger.info(f"  → Generated {len(entries)} Alpaca entries")
        all_entries.extend(entries)
    
    if not all_entries:
        logger.error("No data was processed! Please check your CSV files.")
        sys.exit(1)
    
    logger.info(f"\nTotal Alpaca entries: {len(all_entries)}")
    
    # Print label distribution
    label_counts = {}
    for entry in all_entries:
        # Extract label from output
        output = entry["output"]
        if "Kategori:" in output:
            label = output.split("Kategori:")[1].split(".")[0].strip()
            label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info("\n📊 Label Distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_entries) * 100
        logger.info(f"  {label}: {count:,} ({pct:.1f}%)")
    
    # Split train/val
    train_data, val_data = train_val_split(all_entries, args.val_split, args.seed)
    logger.info(f"\nTrain: {len(train_data):,} | Val: {len(val_data):,}")
    
    # Save outputs
    train_path = OUT_DIR / "sft_train_alpaca.json"
    val_path = OUT_DIR / "sft_val_alpaca.json"
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ Train data saved to: {train_path}")
    logger.info(f"✅ Val data saved to:   {val_path}")
    
    # Save stats
    stats = {
        "total_entries": len(all_entries),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "val_ratio": args.val_split,
        "label_distribution": label_counts,
        "processed_files": list(processed_files),
        "format": "alpaca",
    }
    
    stats_path = OUT_DIR / "sft_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📊 Stats saved to: {stats_path}")
    
    # Print sample
    logger.info("\n── Sample Entry ──")
    sample = all_entries[0]
    logger.info(f"Instruction: {sample['instruction'][:100]}...")
    logger.info(f"Input: {sample['input'][:150]}...")
    logger.info(f"Output: {sample['output'][:200]}...")
    
    logger.info("\n" + "=" * 60)
    logger.info("SFT preprocessing complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
