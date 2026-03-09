# Justifikasi Pemilihan Base Model: Qwen3-8B

## Ringkasan Keputusan

**Model terpilih: unsloth/Qwen3-8B-bnb-4bit** (Alibaba Cloud / Unsloth pre-quantized)

Qwen3-8B dipilih sebagai base model untuk proyek deteksi DFK berdasarkan evaluasi performa, efisiensi, dan kemudahan deployment.

## Perbandingan Benchmark

| Metrik              | Qwen3-8B      | Qwen2.5-7B | Llama 3.1-8B |
| ------------------- | ------------- | ----------- | ------------ |
| **MMLU**            | **75.8**      | 74.2        | 73.0         |
| **HumanEval**       | **68.9**      | 61.0        | 62.2         |
| **MGSM (ID)**       | **78.0**      | 66.4        | -            |
| **VRAM (4-bit)**    | ~5 GB         | ~5 GB       | ~5 GB        |

## Alasan Pemilihan

### 1. Performa Reasoning Terbaik di Kelasnya

Qwen3-8B unggul di benchmark reasoning dan coding. Fitur "Thinking Mode" memungkinkan step-by-step reasoning yang sangat berguna untuk analisis disinformasi — model bisa menjelaskan *mengapa* suatu konten diklasifikasikan sebagai DFK.

### 2. Dukungan Multilingual Kuat

- Tokenizer mendukung bahasa Indonesia dengan baik
- Performa MGSM (ID) 78.0 — tertinggi di kelasnya untuk reasoning bahasa Indonesia
- Dilatih pada corpus multilingual yang luas

### 3. Efisiensi Komputasi

Varian 4-bit pre-quantized dari Unsloth hanya membutuhkan **~5 GB VRAM**, memungkinkan:

- Training di Google Colab T4 (16 GB)
- Inference di laptop dengan GPU 8 GB
- Download cepat (~5 GB vs ~16 GB model full precision)

### 4. Pre-Quantized via Unsloth

- Versi `unsloth/Qwen3-8B-bnb-4bit` sudah di-quantize, langsung siap pakai
- Tidak perlu manual quantization atau konfigurasi BitsAndBytes
- Terintegrasi native dengan Unsloth untuk training yang 2-5x lebih cepat

## Roadmap Scaling

| Fase   | Model              | Kebutuhan           | Kapan                      |
| ------ | ------------------ | ------------------- | -------------------------- |
| Fase 1 | Qwen3-8B (4-bit)  | Colab T4 / GPU 8GB+ | Sekarang                   |
| Fase 2 | Qwen3-32B          | RTX 5090 / A100     | Setelah infrastruktur siap |
| Fase 3 | Model 70B+         | Multi-GPU / Cloud   | Scaling production         |

## Kandidat Alternatif

### SeaLLMs-v3-7B-Chat (DAMO Academy)

- **Kekuatan**: Optimisasi khusus bahasa Asia Tenggara, Refusal-F1 tinggi
- **Kelemahan**: Download besar (~16 GB), sering gagal di Colab karena timeout

### Sahabat-AI (GoTo & Indosat)

- **Kekuatan**: Dukungan 5 bahasa daerah (Jawa, Sunda, Batak Toba, Bali)
- **Kelemahan**: Model 70B butuh 35-40 GB VRAM (4-bit)

## Referensi Lengkap

Lihat dokumen riset detail:

- `Risetmodel_LLM for Indonesian Disinformation Analysis.pdf`
- `Strategic Assessment of Foundation Models for Moderating Indonesian Disinformation and Sociocultural Harm.pdf`
- `Continued Pre-Training & SFT Fokus DFK.pdf`
