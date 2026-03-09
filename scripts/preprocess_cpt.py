"""
Tim1-DFK: CPT Data Preprocessor
================================
Processes raw datasets into clean text corpus for Continued Pre-Training.

Usage:
    python scripts/preprocess_cpt.py
    python scripts/preprocess_cpt.py --config configs/cpt_config.yaml

Input:  Dataset/CPT/raw/*.csv  (raw text datasets)
Output: Dataset/CPT/processed/cpt_corpus.txt  (one document per line)
"""

import os
import sys
import re
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Set

import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import clean_text_minimal, setup_logger, print_dataset_stats

logger = setup_logger("preprocess_cpt")

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Dataset" / "CPT"
OUT_DIR = DATA_DIR / "processed"
OUTPUT_FILE = OUT_DIR / "cpt_corpus.txt"
STATS_FILE = OUT_DIR / "cpt_stats.json"

# Minimum text length (in characters) to keep a document
MIN_TEXT_LENGTH = 50


# ── Processing Functions ─────────────────────────────────────────────────────

def extract_texts_from_csv(filepath: Path) -> List[str]:
    """Extract text column from a CSV file."""
    texts = []
    
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")
    
    logger.info(f"  Loaded {filepath.name}: {len(df)} rows, columns: {list(df.columns)}")
    
    # Try common text column names
    text_columns = ["text", "content", "teks", "isi", "artikel", "body"]
    text_col = None
    
    for col in text_columns:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        # Fall back to the largest string column
        str_cols = df.select_dtypes(include=["object"]).columns
        if len(str_cols) > 0:
            avg_lens = {c: df[c].dropna().astype(str).str.len().mean() for c in str_cols}
            text_col = max(avg_lens, key=avg_lens.get)
            logger.info(f"  Auto-detected text column: '{text_col}' (avg length: {avg_lens[text_col]:.0f})")
        else:
            logger.warning(f"  No text column found in {filepath.name}, skipping.")
            return []
    
    for _, row in df.iterrows():
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        text = clean_text_minimal(text)
        if len(text) >= MIN_TEXT_LENGTH:
            texts.append(text)
    
    return texts


def deduplicate_texts(texts: List[str]) -> List[str]:
    """Remove duplicate texts using hash-based deduplication."""
    seen_hashes: Set[str] = set()
    unique_texts = []
    
    for text in texts:
        # Normalize for dedup: lowercase, strip whitespace
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        text_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_texts.append(text)
    
    return unique_texts


def compute_stats(texts: List[str]) -> dict:
    """Compute corpus statistics."""
    word_counts = [len(t.split()) for t in texts]
    char_counts = [len(t) for t in texts]
    
    stats = {
        "total_documents": len(texts),
        "total_words": sum(word_counts),
        "total_characters": sum(char_counts),
        "avg_words_per_doc": round(sum(word_counts) / max(len(texts), 1), 1),
        "avg_chars_per_doc": round(sum(char_counts) / max(len(texts), 1), 1),
        "min_words": min(word_counts) if word_counts else 0,
        "max_words": max(word_counts) if word_counts else 0,
    }
    return stats


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    global MIN_TEXT_LENGTH
    
    parser = argparse.ArgumentParser(description="CPT Data Preprocessor for Tim1-DFK")
    parser.add_argument("--min-length", type=int, default=MIN_TEXT_LENGTH,
                        help="Minimum text length in characters (default: 50)")
    args = parser.parse_args()
    
    MIN_TEXT_LENGTH = args.min_length
    
    logger.info("=" * 60)
    logger.info("Tim1-DFK: CPT Data Preprocessor")
    logger.info("=" * 60)
    
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files in Dataset/CPT/
    csv_files = [f for f in DATA_DIR.glob("*.csv") if f.parent == DATA_DIR]
    
    if not csv_files:
        logger.error("No CSV files found in Dataset/CPT/")
        logger.info("Please place your raw CSV data files in: Dataset/CPT/")
        sys.exit(1)
    
    # Process each CSV
    all_texts = []
    for csv_file in sorted(csv_files):
        logger.info(f"\nProcessing: {csv_file.name}")
        texts = extract_texts_from_csv(csv_file)
        logger.info(f"  Extracted {len(texts)} valid texts")
        all_texts.extend(texts)
    
    logger.info(f"\nTotal texts before dedup: {len(all_texts)}")
    
    # Deduplicate
    unique_texts = deduplicate_texts(all_texts)
    removed = len(all_texts) - len(unique_texts)
    logger.info(f"Removed {removed} duplicates")
    logger.info(f"Total texts after dedup: {len(unique_texts)}")
    
    # Write output corpus (one document per line)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for text in unique_texts:
            # Ensure single line per document
            one_line = text.replace("\n", " ").replace("\r", " ")
            f.write(one_line + "\n")
    
    logger.info(f"\n✅ Corpus saved to: {OUTPUT_FILE}")
    logger.info(f"   File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    
    # Compute and save stats
    stats = compute_stats(unique_texts)
    stats["source_files"] = [f.name for f in csv_files]
    stats["duplicates_removed"] = removed
    
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📊 Stats saved to: {STATS_FILE}")
    
    # Print stats
    print_dataset_stats(unique_texts, "CPT Corpus")
    
    logger.info("\n" + "=" * 60)
    logger.info("CPT preprocessing complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
