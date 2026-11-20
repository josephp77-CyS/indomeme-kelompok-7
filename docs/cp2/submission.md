# Reproduksi dan verifikasi Checkpoint 2

Model pengumpulan adalah **B2 seed 42**, dipilih berdasarkan Macro-F1 raw rata-rata pada validation. Masukan terdiri dari caption, OCR, dan gambar. Keluaran `hate=1` berarti hateful dan `appropriate=1` berarti inappropriate. Skor 77,37% adalah rata-rata Macro-F1 kedua tugas pada 1.005 meme official test. [Protokol B2](protokol-full-seed42.md)

## Paket weight

Untuk pengumpulan, gunakan `INDOMEME_CP2_B2_seed42.tar`. Arsip berisi `best.pt`, `config.json`, label map, metrik validation/test, provenance, dan `SHA256SUMS`.

| Berkas | SHA-256 |
| --- | --- |
| `INDOMEME_CP2_B2_seed42.tar` | `e9bbdc5f995613f3401372f59319812e32e243cf54e0733a25fa799b3a62b20e` |
| `best.pt` | `71dbdb3c538410b860ffb63dd2632b1af8fecf6c730c91431c33b02db008ef99` |

Hash arsip dan semua isinya diverifikasi. Weight dimuat pada CPU dengan `weights_only=True`, hash config cocok, dan delapan prediksi raw untuk empat meme validation cocok dengan prediksi tersimpan.

## Menjalankan verifikasi

Gunakan Python 3.12 dengan PyTorch yang sesuai dengan perangkat dan Transformers 4.46.3 dari [persyaratan CP2](../../requirements-cp2-nlp.txt). Siapkan akses ke [INDOMEME](https://huggingface.co/datasets/aiatums/INDOMEME), tokenizer IndoBERTweet, dan backbone ViT yang revisinya dipatok di kode.

```bash
python -m pip install -r requirements.txt -r requirements-cp2-nlp.txt
export PYTHONPATH=src
python -m pytest -q
python -m cp2.nlp_runner init-data --workspace "$CP2_WORKSPACE"
python -m cp2.nlp_cv_eval --variant B2 --seed 42 --workspace "$CP2_WORKSPACE" \
  --checkpoint "$CP2_MODEL_DIR/best.pt" --output "$CP2_CHECK_DIR"
```

Tetapkan ketiga direktori di luar checkout. Ekstrak arsip ke `CP2_MODEL_DIR` dan pakai lokasi baru untuk output evaluasi. Evaluator memeriksa config, manifest, dan checkpoint, lalu menjalankan robustness pada validation. [Runner dan panduan CP2](../../README_CP2_NLP.md)

## Hasil dan reproduksi eksperimen

- [Router S1, audit caption, dan classifier S3](caption-repair-results.md)
- [Protokol regenerasi caption](caption-repair-study.md)
- [Router tiga aksi dan uji kuota 5%, 10%, dan 20%](three-actions.md)
- [Agregat eksperimen](../../results/cp2/) dan [konfigurasi terkunci](../../configs/)

Hasil robustness memakai caption donor sintetis pada data yang sudah dipakai dalam studi sebelumnya. Biaya kesalahan memberi bobot dua untuk hate yang terlewat dan satu untuk setiap kesalahan lain. Audit manusia atas kesalahan caption alami terpisah dari evaluasi utama.
