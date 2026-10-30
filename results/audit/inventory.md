# Inventaris lengkap hasil empiris INDOMEME

Inventaris ini membedakan test inti, validation baseline, hasil historis tambahan, kalibrasi, kegagalan, dan percobaan RQ3 tidak lengkap. Tidak ada training, inferensi, atau API baru. Berkas sumber tidak diubah.

## Empat belas baseline konvensional validation

Seluruh trial memakai fit 3.214 dan validation 804 dari train resmi, seed 42. Test resmi tidak dievaluasi. Baseline bukan hasil deep learning paper. Kosakata TF-IDF dipelajari pada fit saja.

| Trial | Tugas | Input | Macro P | Macro R | Macro F1 | Accuracy |
|---|---|---|---:|---:|---:|---:|
| B01 | hate | Majority | 0.330224 | 0.500000 | 0.397753 | 0.660448 |
| B02 | hate | OCR | 0.726356 | 0.731690 | 0.728749 | 0.753731 |
| B03 | hate | Caption | 0.731496 | 0.735457 | 0.733334 | 0.758706 |
| B04 | hate | Caption+OCR | 0.756037 | 0.754186 | 0.755087 | 0.781095 |
| B05 | hate | Caption+Topic | 0.779324 | 0.781762 | 0.780504 | 0.802239 |
| B06 | hate | OCR+Topic | 0.767555 | 0.762608 | 0.764925 | 0.791045 |
| B07 | hate | Caption+OCR+Topic | 0.788077 | 0.786521 | 0.787285 | 0.809701 |
| B08 | appropriate | Majority | 0.282338 | 0.500000 | 0.360890 | 0.564677 |
| B09 | appropriate | OCR | 0.682376 | 0.682008 | 0.682182 | 0.687811 |
| B10 | appropriate | Caption | 0.715294 | 0.714242 | 0.714712 | 0.720149 |
| B11 | appropriate | Caption+OCR | 0.738914 | 0.734065 | 0.735734 | 0.742537 |
| B12 | appropriate | Caption+Topic | 0.745771 | 0.744928 | 0.745319 | 0.750000 |
| B13 | appropriate | OCR+Topic | 0.751668 | 0.747042 | 0.748705 | 0.754975 |
| B14 | appropriate | Caption+OCR+Topic | 0.757422 | 0.755167 | 0.756117 | 0.761194 |

Prediksi gabungan: evidence/baseline_validation_2026-10-18/predictions.csv. Gunakan filter task dan variant; tiap trial berisi tepat 804 baris. Config, split manifest, metrics JSON/CSV, serta completion disertakan. Dua re-execution notebook 18 dan 23 Oktober menghasilkan metrik yang sama dan disertakan sebagai replay, bukan konfigurasi independen tambahan.

## Hasil supervised tambahan di luar delapan core

- E001 SwinV2 hate menyelesaikan empat epoch, memiliki test macro-F1 0,6269661825 dan artefak prediksi/metrik. Hasil ini historis tambahan, tidak mengubah delapan konfigurasi core yang telah disepakati.
- E002 SwinV2 inappropriate memiliki tiga epoch history. Validation terbaik 0,6193252967 pada epoch 2. Hasil ditunda karena perubahan cakupan. Tidak ada result/test metrics, sehingga tidak boleh dilaporkan sebagai eksperimen test selesai.

## Kalibrasi dan kegagalan

- colab_vit_received_2026-10-18: GPU Tesla T4, loss 0.749701, satu langkah 2.571 detik, peak allocated 1690.82 MiB. Kalibrasi saja.
- local_vit_2026-10-23: GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU, loss 0.737444, satu langkah 5.801 detik, peak allocated 1698.54 MiB. Kalibrasi saja.
- local_fusion_E059_2026-10-23: GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU, loss 0.590871, satu langkah 7.729 detik, peak allocated 3839.67 MiB. Kalibrasi saja.

Kalibrasi Colab berasal dari kiriman pengguna, dengan schema/revisi diverifikasi; eksekusi tidak diamati langsung. Dua cek runtime forward/backward pascapembersihan juga disertakan sebagai diagnostik lingkungan.

- E059 dan E101 lokal asli gagal pada backward dengan pesan nyata RuntimeError: CUDA error: out of memory. Tidak ada metrik test untuk percobaan gagal. Keduanya kemudian memiliki run lab yang berhasil, dipisahkan identitasnya.

## Percobaan RQ3 historis

- LLaVA OneVision 0.5B full__d8cfc2726036: scope full, request 1005, baris prediksi 2, label valid 1, status incomplete_partial_not_research_result.
- LLaVA OneVision 0.5B smoke__6a3ae35e666b: scope smoke, request 1, baris prediksi 1, label valid 1, status smoke_complete_not_research_result.
- LLaVA OneVision 0.5B smoke__cb5abfb7aee3: scope smoke, request 3, baris prediksi 3, label valid 2, status incomplete_partial_not_research_result.
- Qwen2-VL 2B smoke__8e71801efce7: scope smoke, request 3, baris prediksi 0, label valid 0, status prepared_without_response_evidence.
- Qwen2-VL 2B smoke__aec1c2daa729: scope smoke, request 1, baris prediksi 0, label valid 0, status prepared_without_response_evidence.

LLaVA memiliki bukti ModuleNotFoundError yang kemudian diikuti retry satu sampel, ValueError pada percobaan test penuh yang berhenti setelah dua baris, dan keluaran tidak sesuai label pada smoke tiga sampel. Tidak ada skor test RQ3 yang lengkap. Dua direktori Qwen hanya memuat config/request, tanpa response atau completion; tidak ada dasar untuk mengarang jenis error atau skor. RQ3 tetap di luar cakupan aktif dan tidak dipanggil ulang.

## Audit data tambahan yang pernah selesai

- Screening pHash 64 bit DCT dengan Hamming <= 6 pada 5.023 gambar menghasilkan 197 pasangan kandidat, termasuk 73 lintas split resmi, 5 pasangan exact yang ditemukan kembali, dan 192 nonexact. Manual review belum selesai dan split tidak diubah. Jangan menulis bahwa screening near-duplicate belum pernah dilakukan.
- Audit tokenizer berisi 18 kombinasi, tiga tokenizer dan enam varian, pada 4.018 train resmi. IndoBERTweet kombinasi lengkap memiliki median 48, p95 83, maksimum 210 token dan nol di atas 256. XLM-R kombinasi lengkap memiliki maksimum 257 dengan satu baris di atas 256. Jangan mengubah pengamatan train menjadi klaim test.

## Cakupan dan risiko

Semua source path, staged path, ukuran dan SHA-256 tersedia dalam inventory.json. Tidak ada gambar, checkpoint, environment atau request penuh 1.005 sampel yang disalin. JSON response ringkas mempertahankan tipe error dan contoh keluaran tanpa kredensial. Angka kalibrasi per-kondisi lab belum tersedia sebagai JSON lokal di inventaris ini; penambahan memerlukan bukti dari folder kelompok sendiri.

Validasi metrik dalam inventaris menghitung ulang macro P/R/F1 dan accuracy dari prediksi untuk 14 baseline, E001 test/validation dan E002 validation. Pemeriksaan ini bukan eksekusi ulang model dan tidak memperbaiki protokol historis.
