# Kebutuhan Infrastruktur Tim1-DFK

## Deteksi Disinformasi, Fitnah, dan Kebencian (DFK) di Indonesia

---

### 1. Aktivitas teknis yang paling memakan resource

**Continued Pre-Training (CPT)** dan **Supervised Fine-Tuning (SFT)** model LLM 7B parameter menggunakan teknik QLoRA (4-bit quantization + LoRA adapter) melalui library Unsloth.

- **CPT**: Melatih model agar mengenal konteks bahasa Indonesia sehari-hari, slang, dan pola teks DFK (ujaran kebencian, hoaks, disinformasi) dari ~11.456 dokumen teks mentah.
- **SFT**: Melatih model agar bisa mengklasifikasikan dan memberikan reasoning terhadap konten DFK menggunakan ~12.657 data instruksi format Alpaca.

---

### 2. Model yang digunakan dan ukurannya

**unsloth/Qwen3-8B-bnb-4bit** (8 Billion parameters, pre-quantized 4-bit) dari HuggingFace/Unsloth.

Dipilih berdasarkan benchmark perbandingan berikut:

| Metrik          | Qwen3-8B      | Qwen2.5-7B | Llama 3.1-8B |
| --------------- | ------------- | ----------- | ------------ |
| MMLU            | **75.8**      | 74.2        | 73.0         |
| HumanEval       | **68.9**      | 61.0        | 62.2         |
| MGSM (ID)       | **78.0**      | 66.4        | -            |

Qwen3-8B unggul di benchmark reasoning dan multilingual, dengan keunggulan signifikan pada MGSM bahasa Indonesia untuk penalaran logis — kritis untuk analisis disinformasi.

---

### 3. Kapasitas VRAM yang dibutuhkan

**Minimal 16 GB VRAM** (GPU NVIDIA).

Dengan teknik QLoRA 4-bit, model 7B hanya membutuhkan **~5.5 GB VRAM** untuk inferensi. Saat training (CPT/SFT), puncak penggunaan VRAM mencapai **~12-14 GB** karena ada overhead gradien dan optimizer. Tesla T4 (15 GB VRAM) sudah cukup.

---

### 4. Apakah butuh GPU spesifik?

**Ya, butuh GPU NVIDIA dengan minimal 15 GB VRAM.**

| GPU       | VRAM  | Kecepatan Training    | Ketersediaan        |
| --------- | ----- | --------------------- | ------------------- |
| Tesla T4  | 15 GB | Cukup (~6 jam)        | Colab Free/Pro      |
| L4        | 24 GB | Bagus (~3 jam)        | Colab Pro/Pro+      |
| A100 40GB | 40 GB | Sangat Cepat (~1 jam) | Colab Pro+ / Server |

**Alasan:** GPU dibutuhkan untuk proses forward/backward pass saat melatih model LLM 7B parameter. Tanpa GPU, proses ini tidak memungkinkan secara waktu (bisa berbulan-bulan jika hanya CPU).

---

### 5. Berapa lama GPU bekerja nonstop dalam satu siklus eksperimen

Estimasi menggunakan **Tesla T4** (Colab):

| Fase      | Dataset          | Estimasi Waktu |
| --------- | ---------------- | -------------- |
| CPT       | 11.456 dokumen   | 1-3 jam        |
| SFT       | 12.657 instruksi | 2-4 jam        |
| **Total** |                  | **3-7 jam**    |

Jika menggunakan GPU L4/A100, total bisa berkurang menjadi **1-3 jam**.

---

### 6. RAM komputer dan CPU untuk data preprocessing

Preprocessing data Tim1-DFK **sangat ringan** karena dataset berupa file teks/CSV berukuran kecil:

| Resource | Kebutuhan Minimal | Keterangan                                 |
| -------- | ----------------- | ------------------------------------------ |
| RAM      | **8 GB**          | Dataset terbesar hanya ~14 MB (CSV)        |
| CPU Core | **2-4 Core**      | Preprocessing selesai dalam hitungan detik |

Kami **TIDAK memerlukan** RAM 64 GB atau CPU 16 Core untuk tahap preprocessing, berbeda dengan tim lain yang mungkin memproses dataset visual/multimedia.

---

### 7. Level Google Colab yang dibutuhkan

**Rekomendasi: Google Colab Pro** (Rp ~150.000/bulan).

| Level         | GPU          | Durasi Sesi     | Cocok untuk Tim1-DFK?                   |
| ------------- | ------------ | --------------- | --------------------------------------- |
| Colab Free    | T4 (15 GB)   | Max ~3 jam      | ⚠️ Sering timeout saat training panjang |
| **Colab Pro** | **T4/L4**    | **Max ~12 jam** | **✅ Ideal untuk CPT+SFT**              |
| Colab Pro+    | A100 (40 GB) | Max ~24 jam     | Overkill untuk Fase 1                   |

**Alasan Pro direkomendasikan:** Total training CPT+SFT bisa mencapai 6 jam. Colab Free sering memutus koneksi setelah 3 jam inaktif, yang menyebabkan training gagal di tengah jalan. Colab Pro memberikan sesi lebih panjang dan opsi GPU lebih cepat.

**Dedicated Server belum diperlukan** di Fase 1. Baru dibutuhkan di Fase 3 saat model harus di-deploy sebagai API service.

---

### 8. Estimasi total penyimpanan

| Komponen                           | Ukuran                                  |
| ---------------------------------- | --------------------------------------- |
| Dataset mentah (CSV, XLSX)         | ~20 MB                                  |
| Dataset terproses (JSON, TXT)      | ~16 MB                                  |
| Base Model (download sementara)    | ~15 GB (ephemeral, di local disk Colab) |
| Checkpoint LoRA Adapter CPT        | ~150 MB                                 |
| Checkpoint LoRA Adapter SFT        | ~150 MB                                 |
| W&B Logs                           | ~50 MB                                  |
| **Total permanen di Google Drive** | **< 500 MB**                            |

Model base 15 GB didownload ke disk lokal Colab (ephemeral) dan **tidak perlu disimpan** permanen karena bisa didownload ulang kapan saja. Yang disimpan permanen hanya adapter LoRA yang berukuran sangat kecil.

---

### 9. Jenis Database yang dibutuhkan

**Fase 1-2 (saat ini): TIDAK BUTUH database apapun.**

Semua data disimpan dalam flat-file:

- Dataset SFT: `JSON` (format Alpaca)
- Corpus CPT: `TXT` (plain text)
- Konfigurasi: `YAML`

**Fase 3 (masa depan, jika integrasi RAG):** Baru dipertimbangkan penggunaan **Vector DB** (contoh: Qdrant / ChromaDB) untuk menyimpan knowledge base fakta-fakta kontra-hoaks.

---

### 10. Estimasi jumlah dokumen yang di-index

| Fase      | Jenis Dokumen                  | Jumlah              |
| --------- | ------------------------------ | ------------------- |
| CPT       | Teks mentah domain DFK         | 11.456 dokumen      |
| SFT       | Instruksi Alpaca (train + val) | 14.064 entri        |
| **Total** |                                | **~25.520 dokumen** |

---

### 11. Rencana akses model (API/Deployment)

**Fase 1-2 (saat ini):** Model diakses secara langsung melalui skrip Python di Colab. Belum butuh API endpoint, IP publik, atau Docker.

**Fase 3 (deployment):**

- Framework: **FastAPI** + **vLLM** (untuk serving LLM secara efisien)
- Container: **Docker** (untuk portabilitas antar server)
- Networking: **IP Publik + SSL** (agar bisa diakses oleh tim frontend/Tim2)
- GPU Server: Diperlukan dedicated server dengan GPU untuk menjalankan inference secara real-time

---

### 12. Apakah butuh API Key dari Google/Claude/ChatGPT/dll?

**TIDAK BUTUH.**

Kami menggunakan model open-source (Qwen3-8B) yang dijalankan secara mandiri di atas GPU sendiri. Tidak ada dependensi terhadap API berbayar pihak ketiga (OpenAI, Google, Anthropic, dll). Seluruh proses CPT, SFT, dan inferensi berjalan 100% offline/self-hosted.

---

### 13. Apakah butuh akun Gemini/Claude/ChatGPT Pro?

**TIDAK BUTUH** untuk proses training maupun inference.

Seluruh pipeline Tim1-DFK berjalan mandiri dengan model open-source. Akun AI berbayar hanya diperlukan jika tim ingin:

- Men-generate dataset sintetis secara massal untuk augmentasi data (opsional, bukan kebutuhan utama)
- Melakukan evaluasi benchmark perbandingan dengan model komersial (opsional)

---

### 14. Kebutuhan tambahan lain

| Kebutuhan                          | Status       | Keterangan                                                                                |
| ---------------------------------- | ------------ | ----------------------------------------------------------------------------------------- |
| **W&B (Weights & Biases) Account** | ✅ Wajib     | Gratis. Untuk monitoring grafik training loss secara real-time via web browser            |
| **HuggingFace Account**            | ✅ Wajib     | Gratis. Untuk download model Qwen3 dan upload model hasil training                      |
| **Google Drive**                   | ✅ Sudah ada | User memiliki 2 TB. Digunakan untuk menyimpan dataset, skrip, dan checkpoint LoRA adapter |
| Akses VPN                          | ❌ Tidak     | Tidak diperlukan                                                                          |
| Integrasi antar server tim         | ⏳ Fase 3    | Diperlukan nanti saat integrasi dengan Tim2 (Agentic AI)                                  |
