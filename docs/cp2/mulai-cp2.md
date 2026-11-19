# CP2 INDOMEME — integrasi CV dan NLP

Status 18 November 2026. [CP1](../../README.md) adalah referensi historis. Eksperimen CP2 memakai input dan aturan pemilihan checkpoint yang berbeda, sehingga skor CP1 dan CP2 tidak menjadi perbandingan langsung.

| Kondisi | Arsitektur CV01 | Bukti tersedia |
| --- | --- | --- |
| B0 | IndoBERTweet, teks Caption+OCR | Full seed 42 diaudit di JupyterHub (mean raw validation F1 0,782072); full seed 123 dilaporkan di Colab (0,771214); robustness B0 dua seed dilaporkan |
| B1 | IndoBERTweet + ViT concatenation | Pilot seed 42 diaudit: 0,767186, 18,08 menit/epoch; full belum ada |
| B2 | Pooled projection + shared scalar gate + residual teks | Pilot seed 42 diaudit: 0,754206, 17,41 menit/epoch; full belum ada |
| M1 | Query token teks, key/value 49 patch ViT, masked pooling + residual | Smoke pretrained di CPU JupyterHub lulus; kalibrasi GPU dan full belum ada |
| M2 | M1 + shared scalar gate | Smoke pretrained di CPU JupyterHub lulus; kalibrasi GPU dan full belum ada |
| M3 | M2 + visual dropout per meme dengan peluang 0,1 | Smoke pretrained di CPU JupyterHub lulus; kalibrasi GPU dan full belum ada |

Pilot berbeda dari full dalam jumlah epoch dan jadwal learning rate. Angka pilot bukan ranking model. Referensi sintetis yang lama tetap dipisahkan dari model pretrained resmi di `src/cp2/mm_models.py`.

Smoke di JupyterHub pada commit `42ce32f`: masing-masing 8 fit dan 8 validation, seed 42, satu langkah optimizer, data asli dan bobot pinned dari cache. M1/M2/M3 masing-masing memerlukan 19,0/19,3/20,2 detik untuk train + validation CPU. Tiap `best.pt` dan `last.pt` cocok dengan hash config; checkpoint terbaik dimuat ulang dan mereproduksi prediksi serta probabilitas validation dalam toleransi FP32. Checkpoint dan prediksi disimpan privat. Ini bukti alur, bukan estimasi GPU atau bukti keunggulan metode. Tes sintetis tambahan menguji resume M3 hingga bobot sama persis dengan run tak terputus.

Commit `6e4c212` lulus 74 tes di laptop dan 74 tes di CPU JupyterHub dalam 5,74 detik. Verifier CP1 juga lulus pada source tersebut. Pemeriksaan GPU tanggal 18 November berhasil memilih UUID perangkat yang sama dengan PyTorch, lalu menolak training karena utilisasi 100% dengan sekitar 39 GiB VRAM bebas.

Pada `6fe0da7`, kedua skrip operator lulus 74 tes setelah isolasi PYTHONPATH diperbaiki. Notebook operator selesai menjalankan tujuh sel kode dengan kernel CP2. B0 seed 42 juga dievaluasi ulang di CPU pada 804 meme validation. Prediksi raw untuk normal, no-caption, dan no-OCR cocok dengan hasil sebelumnya, dengan mean raw Macro-F1 masing-masing 0,782072 / 0,692749 / 0,756750. Rescue/hurt juga cocok. CLI awal berhenti pada batas 900 detik setelah dua kondisi tersimpan. Kondisi no-OCR diselesaikan dengan batch evaluasi 8, lalu seluruh prediksi dipasangkan ulang berdasarkan ID dan task. Ketiga kondisi menghasilkan nol perubahan label prediksi. Checkpoint, CSV individual, dan JSON verifikasi tersimpan privat.

## Keputusan training CP2 seed 42

[PR desain CV #2](https://github.com/josephp77-CyS/indomeme-kelompok-7/pull/2), [PR protokol NLP #3](https://github.com/josephp77-CyS/indomeme-kelompok-7/pull/3), dan [PR implementasi NLP #4](https://github.com/josephp77-CyS/indomeme-kelompok-7/pull/4) tetap menjadi riwayat kontribusi masing-masing. Cabang integrasi menggabungkan source dari keduanya untuk review; tidak memutuskan approval atas nama Rafa.

- Pertahankan split historis 3.214 fit / 804 validation / 1.005 official test. Semua model memakai caption lalu OCR tanpa topic; urutan label `hate`, `appropriate` dengan kelas positif `inappropriate`.
- Dua CE berbobot 0,5. Rata-rata **raw** validation Macro-F1 memilih checkpoint dan epoch lebih awal menang pada seri persis. Skor constrained tetap sekunder. Bandingkan model pada seed yang sama dan laporkan biaya secara terpisah.
- Model dipilih dari validation. Batas 256 token, pinned backbone dan seed split 42 tetap. B0 seed 42/123 sudah dilatih sebelum freeze formal; pertahankan sebagai bukti yang ada dan dokumentasikan jika keputusan baru mengubah keterbandingannya.
- Paket eksploratif untuk tenggat 20 November, 00.00 WIB memakai seed 42. Pakai ulang B0 yang kompatibel dan jalankan lima full run multimodal. Laporkan CP1 dan CP2, raw/constrained, normal/ablasi, dan biaya secara terpisah.
- [Keputusan protokol 19 November](protokol-full-seed42.md) mencatat otorisasi Joseph setelah telaah teknis. [Config CP2](../../configs/cp2_nlp_protocol.json) berstatus `FROZEN`. Runner menolak source/protokol yang belum di-commit dan menyimpan hash config pada setiap full run.

## Menjalankan tahap berikutnya

Aktifkan environment CP2 yang sudah diuji. Gunakan checkout bersih pada SHA kandidat dan folder privat yang sama untuk seluruh worker kelompok. Gunakan output baru agar artefak lama tidak tertimpa. Data tidak disalin ke dalam checkout.

[Notebook operator](../../notebooks/00_cp2_nlp_operator.ipynb) dapat dibuka dari folder notebooks. Sel awal menemukan root checkout. Sel tes menjalankan seluruh suite CP2, dan kontrol multimodal menyediakan smoke/pilot B1–M3. [Panduan kontribusi](../../CONTRIBUTING.md) memuat perintah pemeriksaan PR.

```bash
export CP2_WORKSPACE="$HOME/kelompok-7-private/cp2_data"
export CP2_OUT="$HOME/kelompok-7-private/cp2_runs_integrasi"
bash scripts/cp2_mm_run.sh test
python verify_submission.py
bash scripts/cp2_mm_run.sh preflight
# Setelah GPU kelompok dialokasikan dan kapasitas lolos:
for model in m1 m2 m3; do bash scripts/cp2_mm_run.sh "smoke-$model" || break; done
bash scripts/cp2_mm_run.sh pilot-m3
```

Smoke menguji alur dengan sampel terbatas. Pilot mengukur satu epoch pada seluruh fit/validation. Catat VRAM, disk, waktu, parameter, dan hasil muat ulang checkpoint smoke. Pengaman memeriksa GPU yang dipilih PyTorch via UUID, minimal 10 GiB VRAM bebas, RAM Linux 8 GiB, 50 GiB untuk workspace tanpa cache Parquet, dan ruang cadangan untuk checkpoint. Ambang utilisasi standar adalah 10%. Untuk sesi GPU bersama yang diizinkan, set `CP2_ALLOW_SHARED_GPU=YES` pada sesi tersebut. Catat kondisi GPU dan ukur kembali durasi pilot saat berbagi GPU. Jalankan satu worker CP2 pada workspace fisik yang sama.

Setelah GPU dialokasikan, jalankan smoke dan pilot M3. Setelah lolos, jalankan full **B1 → M1 → M2 → M3 → B2** pada seed 42 dari bobot pretrained. Skrip full meminta `CP2_CONFIRM_MM_FULL=YES` sebagai tanda operator sudah memeriksa syarat. Gunakan `resume-<varian>` untuk melanjutkan checkpoint full parsial dengan SHA/config/environment yang sama.

Gunakan `python -m cp2.nlp_cv_eval --help` dengan `PYTHONPATH=src` untuk robustness validation setelah full checkpoint tersedia. Visual-off meniadakan kontribusi langsung ViT; caption tetap membawa informasi turunan gambar. Official test CP2 tetap menunggu checkpoint serta aturan evaluasi yang sudah dikunci. Simpan prediksi per-ID dan checkpoint di luar Git.

## Keluaran yang masih menunggu data

Setelah full checkpoint tersedia, pasangkan prediksi validation dan hitung rescue/hurt per seed dan task. Catat perbedaan perangkat seed 42/123. Setelah daftar kandidat dikunci, evaluasi test resmi sekali sesuai protokol dan laporkan sensitivitas duplikat secara terpisah. Isi notebook, laporan, slide, dan manifest backup dari artefak hasil, dengan status eksperimen parsial jika paket tidak selesai.
