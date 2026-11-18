# CP2 NLP05 — CPU-only aggregate analysis

Tujuan: memanfaatkan ringkasan robustness B0 seed 42 dari branch NLP04 untuk menyiapkan analisis tanpa menunggu GPU. CLI tidak melatih model, memilih checkpoint, atau mengevaluasi official test.

## 1. Analisis dari ringkasan GitHub

Dari root repository pada branch NLP05 yang memiliki file JSON dan skrip:

    conda activate cp2-indomeme
    CUDA_VISIBLE_DEVICES='' python scripts/cp2_nlp05_analyze.py --summary docs/cp2/metrics/b0_seed42_robustness_summary.json

Output Markdown berisi mean raw validation Macro-F1, delta terhadap normal, dan rescue/hurt per task. Macro-F1 per task dan confusion matrix ditandai belum tersedia karena ringkasan yang dipublikasikan hanya bersumber dari output terminal, bukan JSON lengkap.

## 2. Jika CSV prediksi privat tersedia

Di terminal yang memiliki akses resmi ke tiga validation_*_predictions_PRIVATE.csv dari B0_full_seed42:

    CUDA_VISIBLE_DEVICES='' python scripts/cp2_nlp05_analyze.py --summary docs/cp2/metrics/b0_seed42_robustness_summary.json --predictions-dir "$HOME/kelompok-7-private/cp2_runs/B0_robustness_seed42" --output "$HOME/kelompok-7-private/cp2_analysis/nlp05_b0_seed42.md"

Program memeriksa run, seed, split, pasangan tugas dan ID, konsistensi Macro-F1, dan rescue/hurt. Jika cocok, laporan menambahkan confusion matrix agregat dan Macro-F1 per tugas. Jika tidak cocok, CLI berhenti tanpa menerbitkan laporan.

Folder predictions-dir tersebut hanya contoh pada server Joseph. Pada server lain gunakan lokasi privat masing-masing. Jangan commit CSV privat, checkpoint, atau data individual ke GitHub.

## 3. Pekerjaan tersisa

- Seed 123: ringkasan dua seed tersedia di dokumen lama, tetapi JSON dan CSV sumber Colab/Drive perlu diaudit sebelum publikasi lengkap.
- B1/B2/M1/M2/M3: tunggu checkpoint full dan prediksi validation; pilot/smoke bukan skor final.
- Rescue/hurt lintas model membutuhkan dua set prediksi ID yang berpasangan dan belum dikerjakan CLI ini.
- Hasil evaluasi hanya validation, tidak menggunakan official test.

Tes baru tanpa GPU:

    CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_cp2_nlp05_analysis.py
