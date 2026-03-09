/# Tim1-DFK: Deteksi Disinformasi, Fitnah, dan Kebencian

Pipeline **Continued Pre-Training (CPT)** dan **Supervised Fine-Tuning (SFT)** untuk model LLM deteksi konten DFK di Indonesia.

---

## Daftar Isi

1. [Ringkasan Proyek](#ringkasan-proyek)
2. [Arsitektur Pipeline](#arsitektur-pipeline)
3. [Pemilihan Base Model](#pemilihan-base-model)
4. [Struktur Proyek](#struktur-proyek)
5. [Strategi Data](#strategi-data)
6. [Cara Menjalankan](#cara-menjalankan)
7. [Penjelasan Setiap Komponen](#penjelasan-setiap-komponen)
8. [Monitoring dengan W&B](#monitoring-dengan-wb)
9. [Menambahkan Data Baru](#menambahkan-data-baru)
10. [Kebutuhan Infrastruktur](#kebutuhan-infrastruktur)
11. [Pemetaan Tugas ke Implementasi](#pemetaan-tugas-ke-implementasi)

---

## Ringkasan Proyek

Tim1-DFK membangun model LLM yang mampu **mengklasifikasikan** dan **memberikan reasoning** terhadap konten Disinformasi, Fitnah, dan Kebencian (DFK) dalam bahasa Indonesia. Model dilatih melalui dua tahap:

1. **CPT (Continued Pre-Training)** — adaptasi domain: model belajar pola bahasa DFK Indonesia secara unsupervised
2. **SFT (Supervised Fine-Tuning)** — tugas klasifikasi: model belajar mengklasifikasikan teks ke 6 kategori DFK dan memberikan penjelasan

Label klasifikasi:

| Label | Deskripsi |
|-------|-----------|
| Hoax | Informasi palsu yang disebarkan secara sengaja |
| Disinformasi | Informasi palsu dengan niat manipulatif |
| Misinformasi | Informasi tidak akurat tanpa niat jahat |
| Ujaran Kebencian | Konten menyerang berdasarkan SARA |
| Fitnah | Tuduhan palsu untuk merusak reputasi |
| Bukan DFK | Konten aman, terverifikasi |

---

## Arsitektur Pipeline

```
┌─────────────────────────────────────────────┐
│     Base Model: Qwen3-8B                    │
│     (8B parameter, pre-quantized 4-bit)     │
└──────────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  Tahap 1: CPT               │
    │  scripts/preprocess_cpt.py  │  ← Bersihkan & deduplikasi teks
    │  scripts/train_cpt.py       │  ← Latih LoRA adapter (unsupervised)
    │  configs/cpt_config.yaml    │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  Tahap 2: SFT               │
    │  scripts/preprocess_sft.py  │  ← Konversi ke format Alpaca
    │  scripts/train_sft.py       │  ← Latih klasifikasi + reasoning
    │  configs/sft_config.yaml    │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  Model Final (DFK           │
    │  Classifier + Reasoner)     │
    │  outputs/sft/lora_adapter/  │
    └─────────────────────────────┘
```

### Alur Data

```
Dataset/CPT/*.csv ──► preprocess_cpt.py ──► cpt_corpus.txt ──► train_cpt.py ──► outputs/cpt/lora_adapter/
Dataset/SFT/*.csv ──► preprocess_sft.py ──► sft_train_alpaca.json ──► train_sft.py ──► outputs/sft/lora_adapter/
```

---

## Pemilihan Base Model

**Model terpilih: unsloth/Qwen3-8B-bnb-4bit**

| Metrik | Qwen3-8B | Qwen2.5-7B | Llama 3.1-8B |
|--------|----------|------------|---------------|
| **MMLU** | **75.8** | 74.2 | 73.0 |
| **HumanEval** | **68.9** | 61.0 | 62.2 |
| **MGSM (ID)** | **78.0** | 66.4 | - |
| **VRAM (4-bit)** | ~5 GB | ~5 GB | ~5 GB |

**Alasan utama:**
- **Performa terbaik di kelasnya**: Qwen3-8B unggul di benchmark reasoning dan multilingual
- **Dukungan multilingal**: Tokenizer mendukung bahasa Indonesia dengan efisien
- **Pre-quantized**: Tersedia versi 4-bit dari Unsloth, langsung siap pakai tanpa manual quantization
- **Efisien**: 4-bit quantization hanya butuh ~5 GB VRAM, bisa jalan di Colab T4

Justifikasi lengkap: [docs/model_justification.md](docs/model_justification.md)

---

## Struktur Proyek

```
Tim1-DFK/
├── Dataset/                           # Semua data (raw + processed)
│   ├── CPT/
│   │   ├── 14k-manueltonneau-*.csv   # Dataset hate speech Indonesia (14.3K baris)
│   │   └── processed/
│   │       ├── cpt_corpus.txt         # Corpus bersih (1 dokumen per baris)
│   │       └── cpt_stats.json         # Statistik corpus
│   └── SFT/
│       ├── data disinformasi-hoaks.csv       # Dataset hoax TurnBackHoax (~2K baris)
│       ├── konten_dfk_komdigi.csv            # Data DFK Komdigi
│       ├── konten_dfk_terverifikasi.csv      # Konten DFK terverifikasi (dari XLSX)
│       ├── Konten DFK Terverifikasi.xlsx     # File asli XLSX dari Komdigi
│       └── processed/
│           ├── sft_train_alpaca.json         # Data training (90%)
│           ├── sft_val_alpaca.json           # Data validasi (10%)
│           └── sft_stats.json                # Distribusi label
│
├── scripts/                           # Semua script pipeline
│   ├── preprocess_cpt.py              # Preprocessing data CPT
│   ├── preprocess_sft.py              # Preprocessing data SFT
│   ├── train_cpt.py                   # Training CPT (Unsloth + LoRA)
│   ├── train_sft.py                   # Training SFT (TRL SFTTrainer)
│   ├── sanity_check.py                # Validasi pipeline end-to-end
│   ├── convert_xlsx.py                # Konversi XLSX Komdigi → CSV
│   └── utils.py                       # Fungsi utilitas bersama
│
├── configs/                           # Konfigurasi training
│   ├── cpt_config.yaml                # Hyperparameter CPT
│   └── sft_config.yaml                # Hyperparameter SFT
│
├── notebooks/                         # Notebook Google Colab
│   ├── 1_CPT_Colab.ipynb             # CPT training (jalankan pertama)
│   └── 2_SFT_Colab.ipynb             # SFT training (jalankan setelah CPT)
│
├── docs/                              # Dokumentasi
│   ├── model_justification.md         # Justifikasi pemilihan model
│   ├── data_strategy.md               # Strategi komposisi data
│   └── kebutuhan_infrastruktur.md     # Kebutuhan infrastruktur lengkap
│
├── outputs/                           # Output training (tidak di-commit)
│   ├── cpt/lora_adapter/             # LoRA adapter CPT (~150 MB)
│   └── sft/lora_adapter/             # LoRA adapter SFT (~150 MB)
│
├── requirements.txt                   # Dependensi Python
├── .gitignore                         # File yang tidak di-commit
└── README.md                          # File ini
```

---

## Strategi Data

### Data CPT (Unsupervised — teks mentah domain DFK)

| Kategori | Proporsi Target | Sumber | Status |
|----------|----------------|--------|--------|
| Konten DFK | 50% | Dataset hate speech, hoaks TurnBackHoax, konten Komdigi | Mulai terkumpul |
| Fakta/Resmi | 40% | Press release kementerian, ANTARA, portal berita | Perlu crawling |
| Umum | 10% | Wikipedia ID, CC-100, OSCAR corpus | Belum mulai |

**Dataset CPT yang sudah ada:**
- `14k-manueltonneau-indonesian_hate_speech.csv` — 14.306 baris teks hate speech Indonesia

### Data SFT (Supervised — teks berlabel)

| Kategori | Proporsi Target | Sumber |
|----------|----------------|--------|
| Hoax/Disinformasi | 30% | turnbackhoax.id |
| Ujaran Kebencian | 30% | manueltonneau hate speech dataset |
| Fitnah | 15% | Konten DFK Terverifikasi (Komdigi) |
| Misinformasi | 10% | Berita misleading dari portal fact-check |
| Bukan DFK (Negatif) | 15% | Berita resmi, konten informatif |

**Dataset SFT yang sudah ada:**
- `data disinformasi-hoaks.csv` — ~2.000+ baris (hoax, disinformasi, misinformasi)
- `konten_dfk_terverifikasi.csv` — konten DFK dari Komdigi (ujaran kebencian, disinformasi, dll.)
- `konten_dfk_komdigi.csv` — data tambahan DFK Komdigi

**Format output SFT (Alpaca):**
```json
{
  "instruction": "Klasifikasikan teks berikut apakah termasuk konten DFK...",
  "input": "<teks konten>",
  "output": "Kategori: <label>.\n\nPenjelasan: <reasoning>"
}
```

Strategi lengkap: [docs/data_strategy.md](docs/data_strategy.md)

---

## Cara Menjalankan

### Prasyarat

- Google Colab dengan GPU T4 (minimum), disarankan Colab Pro
- Akun Google Drive (untuk menyimpan dataset dan output)
- Akun Weights & Biases gratis (untuk monitoring training)
- Akun HuggingFace gratis (untuk download model)

### Langkah-langkah (Google Colab)

#### Persiapan Awal (sekali saja)

1. **Upload folder `Tim1-DFK/` ke Google Drive**
   ```
   Google Drive/
   └── Tim1-DFK/        ← upload seluruh folder ini
       ├── Dataset/
       ├── scripts/
       ├── configs/
       ├── notebooks/
       └── ...
   ```

2. **Buat akun W&B** di https://wandb.ai (gratis) — untuk monitoring training real-time

#### Tahap 1: CPT (Continued Pre-Training)

1. Buka `notebooks/1_CPT_Colab.ipynb` di Google Colab
2. Pilih Runtime → Change runtime type → **T4 GPU**
3. Jalankan cell secara berurutan:
   - **Configuration** — sesuaikan `MODEL_NAME`, `WANDB_PROJECT` jika perlu
   - **Step 1: Setup** — mount Google Drive, navigasi ke folder proyek
   - **Step 2: W&B** — login dengan API key dari https://wandb.ai/authorize
   - **Step 3: Check Status** — cek cache model dan LoRA yang sudah ada
   - **Step 4: Check Data** — verifikasi dataset CPT tersedia
   - **Step 5: Install** — install Unsloth dan dependensi
   - **Step 6: Preprocess** — bersihkan dan deduplikasi teks CPT
   - **Step 7: Train** — jalankan CPT training (~1.5-3 jam di T4)
   - **Step 8-9: Verify** — verifikasi output dan cek W&B dashboard

**Output:** `outputs/cpt/lora_adapter/` (LoRA adapter ~150 MB)

#### Tahap 2: SFT (Supervised Fine-Tuning)

1. Buka `notebooks/2_SFT_Colab.ipynb` di Google Colab
2. Pilih Runtime → Change runtime type → **T4 GPU**
3. Jalankan cell secara berurutan:
   - **Configuration** — pastikan `USE_CPT_LORA = True`
   - **Step 1: Setup** — mount Google Drive
   - **Step 2: W&B** — login
   - **Step 3: Check Model** — verifikasi CPT LoRA tersedia
   - **Step 4: Check Data** — verifikasi dataset SFT dan distribusi label
   - **Step 5: Install** — install dependensi
   - **Step 6: Preprocess** — konversi CSV → format Alpaca
   - **Step 7: Train** — jalankan SFT training (~1-4 jam di T4)
   - **Step 9: Inference Test** — tes model dengan sampel teks DFK
   - **Step 10: Verify** — verifikasi output

**Output:** `outputs/sft/lora_adapter/` (LoRA adapter ~150 MB)

### Alternatif: Jalankan via Command Line (lokal dengan GPU)

```bash
# Install dependensi
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Preprocessing
python scripts/preprocess_cpt.py
python scripts/preprocess_sft.py

# Sanity check (opsional — validasi pipeline)
python scripts/sanity_check.py

# Training CPT
python scripts/train_cpt.py --config configs/cpt_config.yaml

# Training SFT
python scripts/train_sft.py --config configs/sft_config.yaml

# Dry run (tes tanpa training penuh)
python scripts/train_cpt.py --config configs/cpt_config.yaml --dry-run
python scripts/train_sft.py --config configs/sft_config.yaml --dry-run
```

### Kapan Menjalankan Notebook Mana?

| Situasi | Jalankan |
|---------|----------|
| Pertama kali (belum pernah training) | 1_CPT → 2_SFT |
| Ada data CPT baru (teks unlabeled) | 1_CPT → 2_SFT |
| Ada data SFT baru (teks berlabel) | 2_SFT saja |
| Mau mulai ulang dari awal | Hapus `outputs/`, lalu 1_CPT → 2_SFT |

---

## Penjelasan Setiap Komponen

### 1. `scripts/preprocess_cpt.py` — Preprocessing CPT

**Input:** Semua `*.csv` di `Dataset/CPT/`
**Output:** `Dataset/CPT/processed/cpt_corpus.txt`

Proses:
1. Baca semua CSV dari `Dataset/CPT/`
2. Auto-deteksi kolom teks (cari `text`, `content`, `teks`, atau kolom string terpanjang)
3. Pembersihan minimal (hapus HTML, normalisasi whitespace) — mempertahankan pola bahasa alami
4. Deduplikasi menggunakan MD5 hash
5. Filter minimum 50 karakter
6. Output: 1 dokumen per baris di `cpt_corpus.txt`

### 2. `scripts/preprocess_sft.py` — Preprocessing SFT

**Input:** Semua `*.csv` di `Dataset/SFT/`
**Output:** `Dataset/SFT/processed/sft_train_alpaca.json`, `sft_val_alpaca.json`

Proses:
1. Baca semua CSV dari `Dataset/SFT/`
2. Auto-deteksi kolom teks dan label
3. Normalisasi label ke 6 kategori DFK (mapping dari berbagai format: `0/1`, `hoax`, `hate_speech`, dll.)
4. Konversi ke format Alpaca (`instruction` / `input` / `output`)
5. Jika tidak ada kolom reasoning, assign template reasoning per kategori
6. Split 90/10 (train/val)
7. Simpan ke JSON

### 3. `scripts/train_cpt.py` — Training CPT

**Input:** Corpus dari `Dataset/CPT/processed/cpt_corpus.txt`
**Output:** LoRA adapter di `outputs/cpt/lora_adapter/`

Proses:
1. Load base model Qwen3-8B dengan 4-bit quantization (QLoRA via Unsloth)
2. Cek apakah ada LoRA adapter lama → jika ada, load untuk continual training
3. Jika baru pertama kali, apply fresh LoRA (r=16, alpha=32)
4. Tokenisasi corpus
5. Training causal language modeling (prediksi token berikutnya)
6. Simpan LoRA adapter

**Hyperparameter CPT:**

| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| Learning Rate | 2e-4 | Standar untuk LoRA |
| Epochs | 3 | Balance antara learning dan overfitting |
| Effective Batch Size | 16 | 2 per device × 8 gradient accumulation |
| LoRA Rank (r) | 16 | Cukup untuk domain adaptation |
| Max Seq Length | 2048 | Cakup mayoritas konten DFK |
| Warmup Ratio | 0.1 | Warmup 10% untuk stabilitas |

### 4. `scripts/train_sft.py` — Training SFT

**Input:** Data Alpaca dari `Dataset/SFT/processed/sft_train_alpaca.json`
**Output:** LoRA adapter di `outputs/sft/lora_adapter/`

Proses:
1. Load base model Qwen3-8B
2. Auto-load CPT LoRA jika tersedia di `outputs/cpt/lora_adapter/`
3. Training menggunakan TRL SFTTrainer dengan format Alpaca
4. Validasi setiap 50 steps
5. Best model disimpan berdasarkan validation loss terendah
6. Simpan LoRA adapter final

**Hyperparameter SFT:**

| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| Learning Rate | 2e-4 | Standar SFT dengan LoRA |
| Epochs | 5 | Lebih banyak epoch untuk task supervised |
| Effective Batch Size | 8 | 2 per device × 4 gradient accumulation |
| LoRA Rank (r) | 16 | Konsisten dengan CPT |
| Max Seq Length | 2048 | Sama dengan CPT |
| Eval Steps | 50 | Validasi tiap 50 steps |

### 5. `scripts/sanity_check.py` — Validasi Pipeline

Menjalankan 5 tes untuk memastikan pipeline siap:
1. Cek file data preprocessed ada
2. Validasi format data (Alpaca keys valid)
3. Load model dengan Unsloth (cek GPU)
4. Apply LoRA adapter (cek memori)
5. Jalankan 2 training steps (cek tidak OOM)

```bash
python scripts/sanity_check.py           # Full check
python scripts/sanity_check.py --skip-gpu  # Skip GPU tests
```

### 6. `scripts/convert_xlsx.py` — Konversi XLSX

Mengkonversi `Konten DFK Terverifikasi.xlsx` dari Komdigi ke CSV yang bisa diproses pipeline:
- Normalisasi label KATEGORI ke format DFK standar
- Extract kolom ANALISIS PELANGGARAN sebagai konten
- Include DASAR HUKUM sebagai bagian reasoning

### 7. `scripts/utils.py` — Utilitas Bersama

Fungsi yang digunakan oleh semua script:
- `clean_text()` — pembersihan teks (hapus HTML, URL, mention, normalize whitespace)
- `clean_text_minimal()` — pembersihan minimal untuk CPT
- `validate_alpaca_entry()` — validasi format Alpaca
- `format_alpaca_prompt()` — format ke template Alpaca Indonesia
- `init_wandb()` — inisialisasi W&B monitoring
- `save_checkpoint()` — simpan model dengan metadata
- `load_config()` — load YAML config
- `setup_logger()` — setup logging

### 8. Konfigurasi YAML

**`configs/cpt_config.yaml`** — konfigurasi CPT training
**`configs/sft_config.yaml`** — konfigurasi SFT training

Kedua config menggunakan:
- QLoRA 4-bit quantization
- LoRA r=16, alpha=32
- fp16 precision (kompatibel T4)
- Cosine learning rate schedule
- W&B logging

---

## Monitoring dengan W&B

Setiap training run otomatis ter-track di Weights & Biases:

**Metrik CPT:**
- `train/loss` — training loss (harus menurun)
- `train/learning_rate` — jadwal learning rate
- `train/grad_norm` — gradient norm (harus stabil)

**Metrik SFT:**
- `train/loss` — training loss
- `eval/loss` — validation loss (harus < train loss)
- `train/learning_rate` — jadwal learning rate

**Cara akses:**
1. Login di https://wandb.ai
2. Buka project `tim1-dfk`
3. Bandingkan run berbeda untuk eksperimen hyperparameter

**Tanda training sehat:**
- Loss menurun secara smooth
- Gradient norm stabil, tidak ada spike
- eval/loss < train/loss (tidak overfitting)

**Tanda masalah:**
- Loss meningkat → learning rate terlalu tinggi
- Loss tidak turun → data atau konfigurasi bermasalah
- NaN/Inf → cek data preprocessing

---

## Menambahkan Data Baru

### Data CPT Baru (teks unlabeled)

1. Taruh file CSV baru di `Dataset/CPT/`
2. Jalankan preprocessing:
   ```bash
   python scripts/preprocess_cpt.py
   ```
3. Jalankan training (otomatis lanjut dari checkpoint):
   ```bash
   python scripts/train_cpt.py --config configs/cpt_config.yaml
   ```
4. Jalankan SFT ulang jika perlu

### Data SFT Baru (teks berlabel)

1. Taruh file CSV baru di `Dataset/SFT/`
   - Harus punya kolom teks (`content`, `text`, `teks`) dan label (`classification`, `label`, `labels`)
2. Jalankan preprocessing:
   ```bash
   python scripts/preprocess_sft.py
   ```
3. Jalankan training:
   ```bash
   python scripts/train_sft.py --config configs/sft_config.yaml
   ```

### Data XLSX dari Komdigi

1. Taruh file XLSX di `Dataset/SFT/`
2. Jalankan konversi:
   ```bash
   python scripts/convert_xlsx.py
   ```
3. Lanjutkan dengan preprocessing SFT dan training

### Mekanisme Incremental Learning

Training script otomatis mendeteksi LoRA adapter yang sudah ada:
- Jika ada → load adapter lama, lanjut training dengan data baru + lama
- Jika tidak ada → mulai fresh training
- Pengetahuan model terakumulasi antar sesi training

---

## Kebutuhan Infrastruktur

| Komponen | Minimum | Rekomendasi |
|----------|---------|-------------|
| GPU | Tesla T4 (15 GB VRAM) | L4/A100 |
| RAM | 8 GB | 16 GB |
| Storage (Drive) | 500 MB | 2 GB |
| Colab | Free (sering timeout) | **Colab Pro** |

**Estimasi waktu training (T4):**

| Fase | Waktu |
|------|-------|
| Download model | 5-10 menit |
| CPT (3 epoch) | 1.5-3 jam |
| SFT (5 epoch) | 1-4 jam |
| **Total** | **3-7 jam** |

**Yang perlu disiapkan:**
- Akun W&B gratis (monitoring)
- Akun HuggingFace gratis (download model)
- Google Drive 2 TB (sudah ada)

Detail lengkap: [docs/kebutuhan_infrastruktur.md](docs/kebutuhan_infrastruktur.md)

---

## Pemetaan Tugas ke Implementasi

### Fase 1 — Sub-tim A: Infrastruktur & Pipeline

| Tugas | Implementasi | File |
|-------|-------------|------|
| Pilih base model + justifikasi | Qwen3-8B (via Unsloth), benchmark comparison | [docs/model_justification.md](docs/model_justification.md) |
| Pipeline pretraining (CPT) | Script preprocessing + training + config | `scripts/preprocess_cpt.py`, `scripts/train_cpt.py`, `configs/cpt_config.yaml` |
| Pipeline fine-tuning (SFT) | Script preprocessing + training + config | `scripts/preprocess_sft.py`, `scripts/train_sft.py`, `configs/sft_config.yaml` |
| Dashboard monitoring | W&B integration di semua training script | `scripts/utils.py` (init_wandb), config `report_to: wandb` |
| Uji coba skala kecil | Sanity check + dry run mode | `scripts/sanity_check.py`, flag `--dry-run` |

### Fase 1 — Sub-tim B: Kurasi & Strategi Data

| Tugas | Implementasi | File |
|-------|-------------|------|
| Strategi komposisi data | Dokumen proporsi CPT (50/40/10) dan SFT | [docs/data_strategy.md](docs/data_strategy.md) |
| Kumpulkan data mentah DFK | 3 dataset: hate speech, hoax, konten Komdigi | `Dataset/CPT/*.csv`, `Dataset/SFT/*.csv` |
| Konversi data Komdigi | Script XLSX → CSV dengan normalisasi label | `scripts/convert_xlsx.py` |

### Fase 2 — Sub-tim A: Continual Pretraining

| Tugas | Implementasi | File |
|-------|-------------|------|
| Jalankan CPT | Notebook Colab end-to-end | `notebooks/1_CPT_Colab.ipynb` |
| Eksperimen hyperparameter | Configurable via YAML + W&B tracking | `configs/cpt_config.yaml`, W&B dashboard |
| Preprocessing skala besar | Auto-detect kolom, deduplikasi, cleaning | `scripts/preprocess_cpt.py` |
| Tokenizer check | Menggunakan tokenizer Qwen3 (mendukung bahasa Indonesia) | `scripts/train_cpt.py` (tokenizer dari model) |

### Fase 2 — Sub-tim B: Supervised Fine-Tuning

| Tugas | Implementasi | File |
|-------|-------------|------|
| Dataset klasifikasi + reasoning | Format Alpaca dengan 6 label DFK + template reasoning | `scripts/preprocess_sft.py` |
| Jalankan SFT setelah CPT | Auto-load CPT LoRA, notebook end-to-end | `notebooks/2_SFT_Colab.ipynb`, `scripts/train_sft.py` |
| Inference test | Test sampel DFK di notebook | Cell inference di `2_SFT_Colab.ipynb` |

---

*Tim1-DFK — AITF 2026*
