# Tim1-DFK: Deteksi Disinformasi, Fitnah, dan Kebencian

Pipeline **Continued Pre-Training (CPT)** dan **Supervised Fine-Tuning (SFT)** untuk model LLM deteksi konten DFK di Indonesia.

---

## Daftar Isi

1. [Ringkasan Proyek](#ringkasan-proyek)
2. [Arsitektur Pipeline](#arsitektur-pipeline)
3. [Pemilihan Base Model](#pemilihan-base-model)
4. [Struktur Proyek](#struktur-proyek)
5. [Strategi Data](#strategi-data)
6. [Cara Menjalankan](#cara-menjalankan)
7. [Test Mode](#test-mode)
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

**Semua logic ada di dalam notebook — tidak butuh file `.py` eksternal.**

```
┌─────────────────────────────────────────────┐
│     Base Model: Qwen3-8B                    │
│     (8B parameter, pre-quantized 4-bit)     │
└──────────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  Tahap 1: CPT               │
    │  1_CPT_Colab.ipynb          │
    │  ├ Preprocess CSV → corpus  │
    │  └ Train LoRA (unsupervised)│
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  Tahap 2: SFT               │
    │  2_SFT_Colab.ipynb          │
    │  ├ Preprocess CSV → Alpaca  │
    │  ├ Train LoRA (supervised)  │
    │  └ Inference test           │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼─────────────────┐
    │  Tahap 3: Merge & Upload       │
    │  2_SFT_Colab.ipynb Step 9   │
    │  ├ Merge LoRA → model 16-bit│
    │  └ Upload ke HuggingFace    │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  Model Final (16-bit)       │
    │  HuggingFace Hub            │
    │  + outputs/sft/merged_model/│
    └─────────────────────────────┘
```

### Alur Data

```
Dataset/CPT/*.csv ──► [1_CPT_Colab] preprocess ──► cpt_corpus.txt ──► training ──► outputs/cpt/lora_adapter/
Dataset/SFT/*.csv ──► [2_SFT_Colab] preprocess ──► sft_train_alpaca.json ──► training ──► merge ──► outputs/sft/merged_model/ ──► HuggingFace Hub
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
- **Dukungan multilingual**: Tokenizer mendukung bahasa Indonesia dengan efisien
- **Pre-quantized**: Tersedia versi 4-bit dari Unsloth, langsung siap pakai
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
│       ├── konten_dfk_terverifikasi.csv      # Konten DFK terverifikasi
│       └── processed/
│           ├── sft_train_alpaca.json         # Data training (90%)
│           ├── sft_val_alpaca.json           # Data validasi (10%)
│           └── sft_stats.json                # Distribusi label
│
├── notebooks/                         # Notebook Google Colab (SELF-CONTAINED)
│   ├── 1_CPT_Colab.ipynb             # CPT: preprocess + training (jalankan pertama)
│   └── 2_SFT_Colab.ipynb             # SFT: preprocess + training + inference (setelah CPT)
│
├── docs/                              # Dokumentasi
│   ├── model_justification.md         # Justifikasi pemilihan model
│   ├── data_strategy.md               # Strategi komposisi data
│   └── kebutuhan_infrastruktur.md     # Kebutuhan infrastruktur lengkap
│
├── outputs/                           # Output training (tidak di-commit)
│   ├── cpt/lora_adapter/             # LoRA adapter CPT (~150 MB)
│   └── sft/
│       ├── lora_adapter/             # LoRA adapter SFT (~150 MB)
│       └── merged_model/             # Model final 16-bit (~16 GB) → upload ke HF
│
├── requirements.txt                   # Dependensi Python
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
- Akun HuggingFace gratis (untuk download model + upload model final)

### Langkah-langkah (Google Colab)

#### Persiapan Awal (sekali saja)

1. **Upload folder `Tim1-DFK/` ke Google Drive**
   ```
   Google Drive/
   └── Tim1-DFK/
       ├── Dataset/
       ├── notebooks/
       └── ...
   ```

2. **Buat akun W&B** di https://wandb.ai (gratis) — untuk monitoring training real-time

#### Tahap 1: CPT (Continued Pre-Training)

1. Buka `notebooks/1_CPT_Colab.ipynb` di Google Colab
2. Pilih Runtime → Change runtime type → **T4 GPU**
3. **Edit Step 1 Configuration:**
   - `TEST_MODE = True` untuk tes pipeline cepat (~5 menit)
   - `TEST_MODE = False` untuk full training (~1.5-3 jam)
4. Jalankan semua cell secara berurutan (Step 1 → Step 8)

**Output:** `outputs/cpt/lora_adapter/`

#### Tahap 2: SFT (Supervised Fine-Tuning)

1. Buka `notebooks/2_SFT_Colab.ipynb` di Google Colab
2. Pilih Runtime → Change runtime type → **T4 GPU**
3. **Edit Step 1 Configuration:**
   - `TEST_MODE = True` untuk tes pipeline cepat (~5 menit)
   - `TEST_MODE = False` untuk full training (~2-4 jam)
   - `USE_CPT_LORA = True` (otomatis load CPT LoRA)
   - `HF_REPO_ID = "username/repo"` (repo HuggingFace untuk upload)
   - `PUSH_TO_HUB = True` (upload model final ke HuggingFace)
4. Jalankan semua cell secara berurutan (Step 1 → Step 10)

**Output:**
- `outputs/sft/lora_adapter/` — LoRA adapter (untuk incremental training)
- `outputs/sft/merged_model/` — Model final 16-bit (~16 GB)
- **HuggingFace Hub** — Model di-upload otomatis

### Kapan Menjalankan Notebook Mana?

| Situasi | Jalankan |
|---------|----------|
| Pertama kali (belum pernah training) | 1_CPT → 2_SFT |
| Ada data CPT baru (teks unlabeled) | 1_CPT → 2_SFT |
| Ada data SFT baru (teks berlabel) | 2_SFT saja |
| Mau mulai ulang dari awal | Hapus `outputs/`, lalu 1_CPT → 2_SFT |

---

## Test Mode

Kedua notebook memiliki **TEST_MODE** di cell Configuration (Step 1). Ini berguna untuk menguji seluruh pipeline tanpa menunggu berjam-jam.

### Cara Pakai

```python
# Di cell Configuration (Step 1):
TEST_MODE = True      # ← aktifkan untuk tes cepat
MAX_ROWS = 1000       # ← jumlah data yang dipakai
TEST_EPOCHS = 1       # ← hanya 1 epoch
```

### Perbandingan

| Setting | Test Mode | Full Training |
|---------|-----------|---------------|
| `TEST_MODE` | `True` | `False` |
| Data dipakai | 1.000 rows | Semua data |
| Epochs CPT | 1 | 3 |
| Epochs SFT | 1 | 5 |
| Waktu CPT | ~5 menit | 1.5-3 jam |
| Waktu SFT | ~5 menit | 2-4 jam |

### Workflow yang Disarankan

1. **Test dulu**: `TEST_MODE = True` → jalankan semua → pastikan tidak ada error
2. **Full training**: `TEST_MODE = False` → jalankan ulang untuk training sesungguhnya

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
2. Buka `1_CPT_Colab.ipynb`, jalankan dari Step 6 (preprocess) lalu Step 7 (training)
3. Model otomatis lanjut dari LoRA yang sudah ada

### Data SFT Baru (teks berlabel)

1. Taruh file CSV baru di `Dataset/SFT/`
   - Harus punya kolom teks (`content`, `text`, `teks`) dan label (`classification`, `label`, `labels`)
2. Buka `2_SFT_Colab.ipynb`, jalankan dari Step 6 (preprocess) lalu Step 7 (training)

### Mekanisme Incremental Learning

Training otomatis mendeteksi LoRA adapter yang sudah ada:
- Jika ada → load adapter lama, lanjut training dengan data baru + lama
- Jika tidak ada → mulai fresh training
- Pengetahuan model terakumulasi antar sesi training

---

## Kebutuhan Infrastruktur

| Komponen | Minimum | Rekomendasi |
|----------|---------|-------------|
| GPU | Tesla T4 (15 GB VRAM) | L4/A100 |
| RAM | 8 GB | 16 GB |
| Storage (Drive) | 500 MB | 20 GB (termasuk merged model) |
| Colab | Free (sering timeout) | **Colab Pro** |

**Estimasi waktu training (T4):**

| Fase | Test Mode | Full Training |
|------|-----------|---------------|
| Download model | 5-10 menit | 5-10 menit |
| CPT | ~5 menit | 1.5-3 jam |
| SFT | ~5 menit | 2-4 jam |
| Merge + Upload HF | ~2 menit | 15-30 menit |
| **Total** | **~15 menit** | **3.5-7.5 jam** |

**Yang perlu disiapkan:**
- Akun W&B gratis (monitoring)
- Akun HuggingFace gratis (download model + upload model final)
- Google Drive (sudah ada, disarankan ~20 GB kosong untuk merged model)

Detail lengkap: [docs/kebutuhan_infrastruktur.md](docs/kebutuhan_infrastruktur.md)

---

## Pemetaan Tugas ke Implementasi

### Fase 1 — Sub-tim A: Infrastruktur & Pipeline

| Tugas | Implementasi | File |
|-------|-------------|------|
| Pilih base model + justifikasi | Qwen3-8B (via Unsloth), benchmark comparison | [docs/model_justification.md](docs/model_justification.md) |
| Pipeline pretraining (CPT) | Preprocessing + training (self-contained) | `notebooks/1_CPT_Colab.ipynb` |
| Pipeline fine-tuning (SFT) | Preprocessing + training + inference (self-contained) | `notebooks/2_SFT_Colab.ipynb` |
| Dashboard monitoring | W&B integration di kedua notebook | W&B Step 4 di setiap notebook |
| Uji coba skala kecil | TEST_MODE dengan MAX_ROWS | Step 1 Configuration di setiap notebook |

### Fase 1 — Sub-tim B: Kurasi & Strategi Data

| Tugas | Implementasi | File |
|-------|-------------|------|
| Strategi komposisi data | Dokumen proporsi CPT (50/40/10) dan SFT | [docs/data_strategy.md](docs/data_strategy.md) |
| Kumpulkan data mentah DFK | 3 dataset: hate speech, hoax, konten Komdigi | `Dataset/CPT/*.csv`, `Dataset/SFT/*.csv` |

### Fase 2 — Sub-tim A: Continual Pretraining

| Tugas | Implementasi | File |
|-------|-------------|------|
| Jalankan CPT | Notebook Colab end-to-end | `notebooks/1_CPT_Colab.ipynb` |
| Eksperimen hyperparameter | Configurable di Step 1 + W&B tracking | Step 1 Configuration + W&B |

### Fase 2 — Sub-tim B: Supervised Fine-Tuning

| Tugas | Implementasi | File |
|-------|-------------|------|
| Dataset klasifikasi + reasoning | Format Alpaca dengan 6 label DFK + template reasoning | Step 6 di `notebooks/2_SFT_Colab.ipynb` |
| Jalankan SFT setelah CPT | Auto-load CPT LoRA, notebook end-to-end | `notebooks/2_SFT_Colab.ipynb` |
| Inference test | Test sampel DFK di notebook | Step 8 di `notebooks/2_SFT_Colab.ipynb` |
| Merge & Upload HF | Merge LoRA → model 16-bit, upload ke HuggingFace | Step 9 di `notebooks/2_SFT_Colab.ipynb` |

---

*Tim1-DFK — AITF 2026*
