"""
Convert 'Konten DFK Terverifikasi.xlsx' → CSV for SFT pipeline.
Extracts key columns, normalizes labels, and merges into a clean CSV.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "Dataset" / "SFT" / "Konten DFK Terverifikasi.xlsx"
OUT_CSV = ROOT / "Dataset" / "SFT" / "konten_dfk_terverifikasi.csv"

# Read the main sheet
df = pd.read_excel(XLSX, sheet_name="Detail data recap")
print(f"Raw rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# Show sample content
print("\n=== 3 Sample Rows ===")
for i in [0, 100, 500]:
    if i < len(df):
        row = df.iloc[i]
        print(f"\n--- Row {i} ---")
        for col in ["KATEGORI", "ANALISIS PELANGGARAN", "DASAR HUKUM", "ANALISIS DAMPAK", "PLATFORM"]:
            val = str(row.get(col, ""))[:200]
            print(f"  {col}: {val}")

# ── Normalize kategori → DFK labels ──
KATEGORI_MAP = {
    # Ujaran Kebencian
    "sara": "Ujaran Kebencian",
    "ujaran kebencian": "Ujaran Kebencian",
    "sara atau ujaran kebencian": "Ujaran Kebencian",
    "ujaran kebencian dan sara": "Ujaran Kebencian",
    "ujaran kebencian & penghinaan": "Ujaran Kebencian",
    # Provokatif → Ujaran Kebencian (provokatif = inciting hatred)
    "provokatif": "Ujaran Kebencian",
    "provokasi": "Ujaran Kebencian",
    "provokator": "Ujaran Kebencian",
    "provokatfif": "Ujaran Kebencian",  # typo in data
    # Combined provokatif
    "provokatif dan ujaran kebencian": "Ujaran Kebencian",
    "provokasi dan ujaran kebencian": "Ujaran Kebencian",
    "provokatif dan sara": "Ujaran Kebencian",
    "provokasi dan sara": "Ujaran Kebencian",
    "provokatif, ujaran kebencian dan menyerang kehormatan": "Ujaran Kebencian",
    "provokatif dan\nujaran kebencian": "Ujaran Kebencian",
    "provokatif dan ujaran kebencian": "Ujaran Kebencian",
    "ancaman dan provokatif": "Ujaran Kebencian",
    # Disinformasi
    "disinformasi": "Disinformasi",
    "disinformasi dan provokatif": "Disinformasi",
    "provokatif dan disinformasi": "Disinformasi",
    "provokasi dan disinformasi": "Disinformasi",
    "disinformasi dan provokasi": "Disinformasi",
    "disinformasi dan ujaran kebencian": "Disinformasi",
    "disinformasi, fitnah dan ujaran kebencian": "Disinformasi",
    "disinformasi, hoax": "Hoax",
    "provokatif, disinformasi, dan separatisme": "Disinformasi",
    "provokatif, disinformasi, dan sara": "Disinformasi",
    "provokatif, disinformasi dan separatisme": "Disinformasi",
    "provokasi, disinformasi, dan separatisme": "Disinformasi",
    # Hoax
    "hoaks": "Hoax",
    "hoax": "Hoax",
    "berita bohong": "Hoax",
    "pencemaran dan berita bohong": "Hoax",
    # Fitnah
    "fitnah dan ujaran kebencian": "Fitnah",
    "provokatif dan fitnah": "Fitnah",
    # Separatisme → Ujaran Kebencian (separatisme = inciting division)
    "separatisme": "Ujaran Kebencian",
    "sparatisme": "Ujaran Kebencian",  # typo in data
    "provokatif dan separatisme": "Ujaran Kebencian",
    "provokasi dan separatisme": "Ujaran Kebencian",
    "provokator dan separatisme": "Ujaran Kebencian",
    "provokatif dan sparatisme": "Ujaran Kebencian",
    "provokatif\ndan\nseparatisme": "Ujaran Kebencian",
    "provokatif\ndan separatisme": "Ujaran Kebencian",
    "provokatif dan separatisme\n": "Ujaran Kebencian",
    "provokasi dan separatisme\n\n": "Ujaran Kebencian",
    "provokasi - separatisme": "Ujaran Kebencian",
    "separatisme dan ujaran kebencian": "Ujaran Kebencian",
    "separatisme dan provokatif": "Ujaran Kebencian",
    "sparatisme dan provokatif": "Ujaran Kebencian",
    "provokatif, separatisme, dan sara": "Ujaran Kebencian",
    "provokatif, sara, dan separatisme": "Ujaran Kebencian",
    "provokasi, separatisme, dan sara": "Ujaran Kebencian",
    "provokasi, sara, dan separatisme": "Ujaran Kebencian",
    "provokatif, separatisme, dan disinformasi": "Disinformasi",
    "disinformasi dan separatisme": "Disinformasi",
    "disinformasi dan sparatisme": "Disinformasi",
    "provokasi, disinformasi": "Disinformasi",
    # Makar
    "makar": "Ujaran Kebencian",
    # Additional typos
    "provoaktif": "Ujaran Kebencian",
    "pelanggaran keamanan informasi": "Disinformasi",
    # Platform names (errors in data)
    "tiktok": None,
    "facebook": None,
    # Provokasi variants
    "provokasi, separatisme": "Ujaran Kebencian",
    "provokasi\n": "Ujaran Kebencian",
}


def normalize_kategori(raw):
    """Normalize raw KATEGORI to DFK label."""
    if pd.isna(raw):
        return None
    clean = str(raw).strip().lower().replace("\r", "")
    
    # Direct match
    if clean in KATEGORI_MAP:
        return KATEGORI_MAP[clean]
    
    # Fuzzy match
    for key, val in KATEGORI_MAP.items():
        if key in clean or clean in key:
            return val
    
    return None  # Skip unknown


# ── Build clean CSV ──
records = []
skipped = 0
unknown_labels = {}

for _, row in df.iterrows():
    label = normalize_kategori(row.get("KATEGORI"))
    analisis = str(row.get("ANALISIS PELANGGARAN", "")).strip()
    dasar_hukum = str(row.get("DASAR HUKUM", "")).strip()
    dampak = str(row.get("ANALISIS DAMPAK", "")).strip()
    
    if label is None:
        raw = str(row.get("KATEGORI", "")).strip().lower()
        unknown_labels[raw] = unknown_labels.get(raw, 0) + 1
        skipped += 1
        continue
    
    # Build content text from analisis pelanggaran (main reasoning)
    if analisis in ("", "nan", "NaN"):
        skipped += 1
        continue
    
    # Build enriched reasoning with dasar hukum
    reasoning = analisis
    if dasar_hukum not in ("", "nan", "NaN"):
        reasoning += f"\n\nDasar Hukum: {dasar_hukum}"
    
    records.append({
        "content": analisis,
        "classification": label,
        "reasoning": reasoning,
    })

out_df = pd.DataFrame(records)
out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

print(f"\n{'='*60}")
print(f"✅ Saved: {OUT_CSV.name}")
print(f"   Total: {len(out_df):,} rows")
print(f"   Skipped: {skipped:,} (empty/unknown)")
print(f"\n📊 Label distribution:")
print(out_df["classification"].value_counts().to_string())

if unknown_labels:
    print(f"\n⚠️  Unknown labels skipped:")
    for lbl, cnt in sorted(unknown_labels.items(), key=lambda x: -x[1]):
        print(f"   {lbl}: {cnt}")
