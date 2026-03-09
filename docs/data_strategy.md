# Strategi Komposisi Data Tim1-DFK

## Target Data

| Tahap | Target Minimum          | Status                             |
| ----- | ----------------------- | ---------------------------------- |
| CPT   | 100K+ dokumen raw text  | ⏳ 14K (perlu crawling tambahan)   |
| SFT   | 10K+ pasangan instruksi | ⏳ ~2K (perlu augmentasi/tambahan) |

## Komposisi Data CPT

Sesuai notulensi rapat 24 Feb 2026:

| Kategori    | Proporsi | Sumber                                                                | Status             |
| ----------- | -------- | --------------------------------------------------------------------- | ------------------ |
| Konten DFK  | 50%      | Dataset hate speech, hoaks TurnBackHoax, konten terverifikasi Komdigi | ✅ Mulai terkumpul |
| Fakta/Resmi | 40%      | Press release kementerian, ANTARA, ANRI, portal berita terpercaya     | ⏳ Perlu crawling  |
| Umum        | 10%      | Wikipedia ID, CC-100, OSCAR corpus                                    | ⏳ Belum mulai     |

## Komposisi Data SFT

| Kategori            | Proporsi | Sumber                                               |
| ------------------- | -------- | ---------------------------------------------------- |
| Hoax/Disinformasi   | 30%      | turnbackhoax.id, data disinformasi-hoaks.csv         |
| Ujaran Kebencian    | 30%      | manueltonneau hate speech dataset, Instagram dataset |
| Fitnah              | 15%      | Konten DFK Terverifikasi (Komdigi)                   |
| Misinformasi        | 10%      | Berita misleading dari portal fact-check             |
| Bukan DFK (Negatif) | 15%      | Berita resmi, konten informatif                      |

## Dataset yang Sudah Ada

### Folder CPT (`Dataset/CPT/`)

1. **14k-manueltonneau-indonesian_hate_speech.csv** — 14,306 baris
   - Kolom: text, labels (0/1), source (Instagram/Twitter)
   - Sumber: [HuggingFace](https://huggingface.co/datasets/manueltonneau/indonesian-hate-speech-superset)

### Folder SFT (`Dataset/SFT/`)

1. **data disinformasi-hoaks.csv** — ~2,000+ baris
   - Kolom: title, content, classification (hoax/disinformation/misinformation)
   - Sumber: turnbackhoax.id
2. **Konten DFK Terverifikasi.xlsx** — Metadata dari Komdigi
   - Dataset SFT resmi dari Kementerian Komdigi

## Sumber Data Tambahan (Perlu Crawling)

### Portal Berita & Fact-Check

- https://turnbackhoax.id/
- https://www.anri.go.id/en/publications/news
- https://korporat.antaranews.com/publikasi/artikel

### Dataset Publik

- [Indonesian Hate Speech Superset](https://huggingface.co/datasets/manueltonneau/indonesian-hate-speech-superset)
- [Instagram Hate Speech Dataset](https://github.com/nurindahpratiwi/dataset-hate-speech-instagram)
- [TurnBackHoax Dataset](https://github.com/jibranfawaid/turnbackhoax-dataset)

### Link Pengumpulan Dataset CPT Tim

- [Google Drive](https://drive.google.com/drive/folders/1vwJ593UCQIODMtg-MzFNk9dES1kSyqM1?usp=sharing)

## Format Output

### CPT

```
Raw text satu dokumen per baris (cpt_corpus.txt)
```

### SFT (Alpaca Format)

```json
{
  "instruction": "Klasifikasikan teks berikut apakah termasuk konten DFK...",
  "input": "<teks konten>",
  "output": "Kategori: <label>.\n\nPenjelasan: <reasoning>"
}
```

## Catatan Penting

- Dataset SFT resmi dari Komdigi akan diberikan terpisah
- CPT dataset dikumpulkan mandiri oleh tim melalui crawling
- Fase 1-2 awal menggunakan single-turn (Alpaca style)
- Multi-turn (ChatML) dipertimbangkan setelah single-turn terbukti bagus
