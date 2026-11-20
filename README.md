# Reproduksi Eksperimen Inti INDOMEME

**Kelompok 7 — Joseph Pratama Lase dan Rafa Maritza Hanasaputri**  
Intelligent Systems Lab FILKOM 2026

Jika meme menggabungkan gambar dan teks, apakah tambahan gambar selalu membantu model menilainya? Kami membandingkan gambar, teks yang diperkaya, dan gabungannya untuk dua tugas: mendeteksi kebencian dan ketidakpantasan menurut label INDOMEME.

Delapan konfigurasi inti menghasilkan sepuluh hasil tugas. Pada satu seed yang digunakan, fusi lebih baik untuk ketidakpantasan, sedangkan deteksi kebencian tidak membaik terhadap model teks. Teks sudah memuat OCR, deskripsi gambar, dan topik anotasi; hasil ini tidak membuktikan gambar tidak penting.

## Checkpoint 2

CP2 menggunakan gambar, OCR, dan caption untuk mendeteksi kebencian dan ketidakpantasan. Anotasi topik dipakai untuk audit per kelompok setelah inferensi. Semua varian utama selesai dievaluasi pada split historis 3.214 fit, 804 validation, dan 1.005 official test.

| Hasil | Temuan |
| --- | --- |
| **Weight pengumpulan B2 seed 42** | Macro-F1 raw rata-rata **78,87%** pada validation dan **77,37%** pada official test. Skor test per tugas **77,74%** hate dan **77,00%** inappropriate. |
| **Ketahanan terhadap caption donor sintetis** | S1 unggul **4,73 poin Macro-F1** atas S0 pada tiga seed. IK 95% **+3,57 hingga +5,85**. |
| **Regenerasi caption selektif** | Router S1 pada kuota 20% mencapai **75,88%** Macro-F1 dibanding **74,75%** untuk ketidakpastian. Biaya kesalahan turun **2,41 per 100 meme** pada replay dengan 20% caption donor sintetis. |
| **Sensitivitas kuota** | Pada replay yang sama, kuota 10% dan 20% memakai **101 dan 201** permintaan. Selisih biaya per 100 meme **+0,008** (IK 95% **−0,23 hingga +0,26**) adalah estimasi eksploratif. |

[Panduan pengumpulan dan verifikasi model](docs/cp2/submission.md) · [Hasil lengkap dan grafik](docs/cp2/caption-repair-results.md) · [Uji kuota dan router tiga aksi](docs/cp2/three-actions.md)

- [Rancangan model dan kalibrasi CV01](docs/cp2/cv-design.md)
- [Status, keputusan review, dan langkah operator CP2](docs/cp2/mulai-cp2.md)
- [Penjelasan sederhana untuk diskusi](docs/cp2/laporan-sederhana.md) · [Word](docs/cp2/laporan-sederhana.docx)
- [Contoh hasil dan serah terima NLP](docs/cp2/contoh-hasil-nlp.md) — panduan, belum menjadi hasil pekerjaan Rafa

## Materi Checkpoint 1

- [Presentasi utama tim](docs/INDOMEME_Checkpoint1_PitchDeck.pptx)
- [Ringkasan dengan bahasa sederhana](docs/ringkasan.md)
- [Laporan lengkap dengan rumus dan analisis](docs/laporan.md) · [Word](docs/laporan.docx) · [HTML untuk diunduh](docs/laporan.html)
- [Notebook lengkap dengan output](notebooks/eksperimen_indomeme.ipynb)
- [Panduan data dan reproduksi](docs/panduan_reproduksi.md)

## Alur penelitian

Dataset rilis → audit dan preprocessing → latihan 3.214 meme → pemilihan checkpoint dengan 804 meme validation → evaluasi 1.005 meme test → analisis hasil dan kesalahan.

## Hasil reproduksi Checkpoint 1

Semua angka berikut adalah **Macro-F1 pada test**, bukan persentase jawaban benar. Kelas 1 pada tugas internal `appropriate` berarti **inappropriate**. Angka ketidakpantasan multitask memakai aturan constrained.

| Masukan dan cara belajar | Kebencian | Ketidakpantasan |
| --- | --- | --- |
| Gambar satu tugas | 0,6408 | 0,6315 |
| Teks diperkaya satu tugas | 0,8058 | 0,7984 |
| Teks dan gambar satu tugas | 0,8043 | 0,8125 |
| Teks multitask | 0,8043 | 0,7981 |
| Teks dan gambar multitask | 0,8029 | 0,8109 |

[Tabel sepuluh hasil tugas](results/hasil_utama.csv) dan [empat belas baseline validation](results/hasil_baseline.csv) memisahkan dua bagian evaluasi tersebut. Kandidat berdasarkan validation adalah E059 untuk kebencian dan E042 untuk ketidakpantasan. Skor test tertinggi tidak dijadikan dasar mengganti kandidat.

## Struktur

| Folder | Isi |
| --- | --- |
| `docs/` | Presentasi, laporan, ringkasan, visual, dan panduan |
| `notebooks/` | Satu notebook utama yang sudah memiliki output |
| `src/` | Implementasi, runner analisis, reproduksi training, dan sumber penyusunan dokumen |
| `results/` | Tabel hasil, konfigurasi dan riwayat tiap run, serta bukti audit |

## Memeriksa dan menjalankan

Jalankan `python verify_submission.py` dari root repo untuk memeriksa hasil agregat, source terkunci, notebook beroutput, dan integritas PPT. Pemeriksaan ini memakai standard library Python dan tidak melatih atau mengunduh model.

Untuk menjalankan ulang analisis, ikuti [panduan reproduksi](docs/panduan_reproduksi.md). Gambar dataset, CSV prediksi dan ID individual, environment, serta checkpoint besar tidak dimasukkan ke GitHub. Paket analisis pendamping tersedia melalui [Google Drive](https://drive.google.com/drive/folders/1c7cuGs_ij6ghZxLF21D6GZ1Hk3a8nplH); [arsip model](https://drive.google.com/drive/folders/1xXrSuGL89PohMh4soz6Nt7KnNNVnWxJo) terpisah.

## Batas kesimpulan reproduksi Checkpoint 1

Delapan konfigurasi inti dan notebook analisis telah selesai. Notebook memuat 52 sel kode dengan keluaran dari lab. Baseline, hasil historis tambahan, proses parsial, kalibrasi, dan kegagalan ditandai terpisah. RQ3 lengkap dan metode baru belum selesai dalam cakupan ini.

Hasil memakai satu seed; selisih belum membuktikan signifikansi atau keunggulan umum. Topik anotasi merupakan informasi tambahan, terdapat duplikat lintas split resmi, dan peninjauan manual kandidat kemiripan belum selesai. Perbedaan protokol terhadap penulis dijelaskan pada laporan.

## Sumber

Pamungkas dkk. (2026), [Decoding hate in memes](https://peerj.com/articles/cs-3736/), DOI 10.7717/peerj-cs.3736. [Dataset INDOMEME](https://huggingface.co/datasets/aiatums/INDOMEME), revisi `3e0cb8cae9cd726582f69afdb122163d0eab108b`, dataset card CC BY 4.0. Lisensi dataset tidak otomatis memberikan lisensi baru untuk seluruh isi repo.
