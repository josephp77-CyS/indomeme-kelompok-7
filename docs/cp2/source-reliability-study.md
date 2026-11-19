# Studi reliabilitas caption dan OCR INDOMEME

## Pertanyaan dan batas

Kontras utama adalah **S2 minus S1 pada caption_swap**, dihitung berpasangan pada meme dan seed yang sama. Metrik adalah mean raw Macro-F1 dari hate dan inappropriate. Nilai normal dan kondisi lain ikut dilaporkan. Semua angka tahap ini berasal dari 804 validation; pemilihan checkpoint memakai kondisi normal. [Config studi](../../configs/cp2_source_reliability.json)

| Kondisi | Input yang diubah |
| --- | --- |
| normal | Caption, OCR, dan gambar asli |
| no_caption | Caption dikosongkan bila OCR tersedia |
| caption_swap | Caption donor yang berbeda dari split yang sama, dipilih dekat dari panjangnya, tanpa membaca label |
| no_ocr | OCR dikosongkan bila caption tersedia |
| ocr_noise | Sekitar 20% posisi karakter yang memenuhi syarat diganti dengan karakter mirip, minimal satu bila ada posisi |

Caption donor menghasilkan gangguan sintetis. Kecocokan semantik belum dinilai manusia. Setiap kondisi mempertahankan gambar, ID, dan kedua label asli. Donor training berasal dari fit, donor validation dari validation, dan OCR diubah dengan seed per ID. S1 dan S2 mendapat kondisi yang sama per ID dan epoch. Manifest historis dan bobot backbone dipatok seperti CP2.

## Model dan pembanding

- B0–M3 seed 42 sebagai pembanding awal. B2 dan M2 juga dilatih pada seed 123 dan 2026 untuk pembanding tertanding.
- S0, S1, dan S2 berbagi arsitektur dan inisialisasi pada setiap seed. Caption dan OCR masuk ke IndoBERTweet yang sama dengan alokasi 128 token per sumber. ViT memberi representasi gambar. Gate memberi bobot pada ketiga sumber, dan bobot sumber yang kosong dipaksa nol.
- S0 dilatih dengan input bersih. S1 dan S2 dilatih dengan campuran kondisi yang sama. S2 menambah loss BCE kualitas caption/OCR sebesar 0,1. Target asli atau terganggu hanya masuk ke loss saat training. Gate pada inference membaca representasi dan ketersediaan input.
- Jalankan smoke sebelum full. Gunakan maksimal lima epoch, patience dua, dan seed 42, 123, dan 2026. Laporkan skor masing-masing seed dan interval bootstrap berpasangan yang bersyarat pada seed tersebut.
- Simulasi review S2 membandingkan urutan acak, skor positif, skor plus ketidakpastian, dan penambahan indikasi kualitas sumber. Semua strategi melihat anggaran antrean yang sama. Keluaran mencatat recall label positif, recall kasus yang model salah klasifikasikan, dan jumlah kasus negatif pada kedua tugas yang masuk antrean. Simulasi menggunakan gold sebagai penilaian offline dan tidak mengukur waktu atau keputusan moderator nyata.

## Hasil baseline seed 42

Enam checkpoint CP2 dievaluasi ulang pada lima kondisi dan 804 ID yang sama. Kondisi normal cocok dengan hasil full training. Pada setiap model, caption donor sintetis memberi skor lebih rendah daripada caption yang dihapus.

| Model | Normal | Tanpa caption | Caption ditukar | Selisih terhadap tanpa caption, pp | Interval 95% |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0,7821 | 0,6927 | 0,6452 | −4,75 | −8,14 s.d. −1,37 |
| B1 | 0,7885 | 0,7071 | 0,6447 | −6,24 | −9,77 s.d. −2,75 |
| B2 | 0,7887 | 0,7007 | 0,6555 | −4,51 | −8,02 s.d. −0,99 |
| M1 | 0,7853 | 0,7027 | 0,6486 | −5,41 | −8,84 s.d. −2,01 |
| M2 | 0,7871 | 0,7088 | 0,6480 | −6,08 | −9,53 s.d. −2,65 |
| M3 | 0,7862 | 0,6926 | 0,6606 | −3,20 | −6,50 s.d. +0,08 |

Interval berasal dari 10.000 bootstrap berpasangan pada meme validation, dengan skor yang dikondisikan pada satu seed model. Pada B2, penggantian caption dibanding penghapusan memberi 79 rescue dan 145 hurt pada kebencian, serta 99 rescue dan 160 hurt pada ketidakpantasan. [Bukti agregat baseline dan simulasi antrean](https://node580-ai-hub.ub.ac.id/user/josephplase77/lab/tree/kelompok-7-private/cp2-source-noise-study/baseline_analysis.json)

Pada kondisi caption ditukar dan budget 161 dari 804 item, mengurutkan kasus berdasarkan skor positif B2 menjangkau 36,6% meme dengan label kebencian. Urutan acak menjangkau 17,2%. Menambahkan sinyal ketidakpastian menjangkau 32,0% kesalahan klasifikasi kebencian, dibanding 24,4% dengan skor positif saja. Ini mengilustrasikan perbedaan antara mendahulukan kasus positif dan mendahulukan kesalahan model.

## Eksekusi

Jalankan dari root checkout pada lingkungan CP2 yang sudah disiapkan. Set `CP2_ALLOW_SHARED_GPU=YES` hanya untuk penggunaan bersama yang diizinkan. Simpan prediksi per ID dan checkpoint pada folder privat.

```bash
export CP2_WORKSPACE=/path/to/private/cp2_data
export CP2_STUDY_ROOT=/path/to/private/source_study
export PYTHONPATH="$PWD/src"

# Setelah melengkapi daftar checkpoint B0, B1, B2, M1, M2, dan M3 seed 42.
python -m cp2.noise_eval --workspace "$CP2_WORKSPACE" --output "$CP2_STUDY_ROOT/benchmark" \
  --require-gpu /path/to/B0_full_seed42/best.pt /path/to/B1_full_seed42/best.pt \
  /path/to/B2_full_seed42/best.pt /path/to/M1_full_seed42/best.pt \
  /path/to/M2_full_seed42/best.pt /path/to/M3_full_seed42/best.pt

# Catat kode keluar aktual benchmark di baseline_exit sebelum memulai training.
bash scripts/cp2_source_study.sh
```

## Official INDOMEME test

Setelah ketiga seed selesai dan analisis validation tersedia, [protokol official](../../configs/cp2_official_source_study.json) menetapkan B0–M3 seed 42 dan set berpasangan B2/M2/S0/S1/S2 pada tiga seed. Evaluator mencatat SHA-256 checkpoint dan config serta pilihan CP2 berdasarkan validation pada lock privat **sebelum** membaca gambar test atau menjalankan inferensi. Jalankan kondisi teks yang sama pada 1.005 test; visual-off ditambahkan untuk B1–M3 seed 42. Skor normal juga dihitung pada 1.003 test setelah mengecualikan dua exact RGB duplicate yang cocok dengan fit atau validation. Hasil official tidak digunakan untuk memilih checkpoint.

```bash
python -m cp2.official_source_eval --workspace "$CP2_WORKSPACE" --study "$CP2_STUDY_ROOT" \
  --b0-checkpoint /path/to/B0_full_seed42/best.pt --cp2-runs /path/to/cp2_full_runs \
  --output "$CP2_STUDY_ROOT/official" --require-gpu
python -m cp2.reliability_summary --benchmark "$CP2_STUDY_ROOT/official" \
  --output "$CP2_STUDY_ROOT/official_summary.json" --split test --seeds 42 123 2026
```

## Protokol untuk domain dan anotator baru

Sumber dan anotator eksternal belum tersedia. Siapkan satu domain dengan izin penggunaan dan pisahkan dari INDOMEME maupun template yang hampir identik. Beri setiap item ID privat. Simpan metadata sumber, kondisi input, dua label, dan penilaian kualitas sumber. Dua penilai menulis caption deskriptif atau menandai tidak yakin tanpa melihat prediksi atau label model. Untuk kualitas caption, tandai fakta yang bertentangan, informasi penting yang hilang, dan inferensi yang tidak dapat didukung oleh bukti yang tampak. Catat kesepakatan dan adjudikasi. Simpan sampel asli sesuai izin sumber. Evaluasi held-out lintas domain dan sesi moderator dilakukan setelah data tersedia.
