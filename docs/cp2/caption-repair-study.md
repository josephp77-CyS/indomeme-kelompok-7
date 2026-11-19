# Regenerasi caption untuk moderasi meme Indonesia dengan anggaran tetap

[Hasil dan bahan presentasi](caption-repair-results.md)

INDOMEME memakai caption Gemini 2.5 Flash dan OCR Qwen2-VL-2B sebagai informasi tambahan untuk klasifikasi kebencian dan ketidakpantasan. Caption awal mencakup isi visual dan pesan tersirat. Eksperimen ini menguji apakah deskripsi visual baru membantu kedua keputusan, terutama saat caption yang tersedia keliru. Aplikasi yang dituju adalah antrean moderasi komunitas berbahasa Indonesia, dengan pemrosesan caption tambahan dibatasi pada item yang diperkirakan paling terbantu. [Paper INDOMEME](https://doi.org/10.7717/peerj-cs.3736) · [Kartu dataset](https://huggingface.co/datasets/aiatums/INDOMEME/blob/main/README.md)

Posisi kontribusinya adalah perbandingan terkontrol antara strategi memilih intervensi dan training yang membiasakan model dengan caption baru. Pembelajaran kebijakan berbiaya dan pemilihan intervensi dengan kapasitas terbatas sudah memiliki preseden. Di sini pembanding memakai aksi regenerasi caption yang sama dan biaya klasifikasi dilacak pada dua tugas moderasi. [Mozannar dan Sontag, ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html) · [Alves dkk., TMLR 2024](https://arxiv.org/abs/2403.06906)

## Pertanyaan penelitian

1. Saat kuota regenerasi 20%, apakah prediksi manfaat per item menurunkan biaya kesalahan dibanding urutan berdasarkan ketidakpastian?
2. Apakah S3, dengan sebagian caption fit diganti keluaran model generatif, mempertahankan atau memperbaiki hasil dibanding S1 pada caption regenerasi, caption asli, caption tertukar, dan input tanpa caption?
3. Berapa banyak kasus yang membaik dan memburuk setelah regenerasi, dan bagaimana pola itu berubah antara caption asli dan caption donor sintetis?

## Protokol

- Qwen2-VL-2B-Instruct dipatok ke satu revisi dan dijalankan lokal di A100. Generator menerima piksel dan prompt tetap. Caption kosong atau terpotong dianggap tidak tersedia. [Model dan lisensi](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- B2 adalah baseline fusion CP2 dan S1 adalah baseline source gate dengan augmentasi. Tiga seed dipakai untuk masing-masing. Router Ridge belajar dari selisih biaya kesalahan sebelum dan sesudah regenerasi. Input router berupa probabilitas dengan dan tanpa caption yang sedang tampil. Beban satu kesalahan adalah 2 untuk hate yang terlewat, 1 untuk hate positif palsu, dan 1 untuk kesalahan ketidakpantasan.
- Router diuji pada 804 validation dengan prediksi out-of-fold yang menempatkan versi asli dan terganggu dari satu meme di lipatan yang sama. Router kemudian dilatih ulang pada seluruh validation. Sebanyak 1.005 test INDOMEME dinilai secara retrospektif. Dua puluh replay memakai penempatan gangguan dan tie-break yang sama untuk setiap kebijakan. Kondisi utama mengganti 20% caption dengan caption donor. Kondisi tambahan memakai caption asli.
- Random, ketidakpastian, perubahan probabilitas saat caption dihilangkan, dan router terlatih memperoleh kuota identik. Kontrol tambahan meliputi mempertahankan caption, menghapus caption pada pilihan router, dan mencoba regenerasi untuk seluruh item. Caption yang tidak tersedia mempertahankan prediksi awal pada simulasi regenerasi. Kuota mengukur jumlah permintaan caption dan belum menghitung biaya tambahan untuk pass tanpa caption.
- S3 dan S1 berbagi arsitektur, seed, jadwal gangguan, dan pemilihan checkpoint Clean Macro-F1. Untuk S3, sekitar separuh bagian normal mendapat tawaran caption baru. Dengan proporsi normal 60%, ini mencakup hingga sekitar 30% input fit. Semua 3.214 fit dipakai. Konfigurasi memberi maksimum lima epoch dan early stopping setelah dua epoch tanpa perbaikan.
- Selisih berpasangan dan interval bootstrap dihitung berdasarkan meme. Seed dan replay dipertahankan tetap. Dua duplikat piksel yang ditemukan pada audit awal tetap masuk ukuran utama 1.005 test agar perbandingan cocok dengan hasil awal. [Protokol terkunci](../../configs/cp2_caption_repair.json) · [Audit dan studi awal](source-reliability-study.md)
- Audit tambahan membandingkan tumpang tindih kosakata caption lama dan baru terhadap OCR. Karena OCR dan regenerasi memakai keluarga model yang sama, kemiripan keluaran relevan untuk membaca hasil. Audit deskriptif melaporkan perubahan biaya saat seluruh caption yang tersedia dipakai oleh S1, dikelompokkan menurut topik dengan minimal 50 meme. Topik dapat bertumpang tindih dan hanya dipakai untuk analisis setelah inferensi.
- Eksperimen tambahan S1 menguji pemeriksaan setelah caption dibuat. Model Ridge kedua memakai prediksi sebelum dan sesudah regenerasi, dengan lipatan dan target yang sama. Ia menerima caption hasil regenerasi hanya bila prediksi manfaat positif. Evaluasinya membandingkan dua kebijakan pada item terpilih yang persis sama, sehingga kuota regenerasi tetap identik. Ambang nol dibakukan sebelum hasil regenerasi dan replay tersedia.

## Pembacaan hasil

Pilot fase sebelumnya mendapati bahwa ketidaksepakatan probabilitas lebih sering menandai swap daripada ketidakpastian. Memakai prediksi tanpa caption pada item terpilih menurunkan Macro-F1 S1 sekitar 1,43 poin dan B2 sekitar 5,06 poin. Regenerasi menguji apakah aksi pengganti yang memanfaatkan gambar memberi nilai lebih. Perubahan skor juga dapat dipengaruhi oleh cakupan semantik yang berbeda antara prompt awal dan prompt deskripsi visual. [Hasil pilot sebelumnya](source-reliability-study.md#pilot-verifikasi-selektif)

Klaim efektivitas perlu bersandar pada selisih biaya kesalahan S1 terhadap ketidakpastian dan intervalnya. Dampak pada caption asli, false positive, serta missed hate membantu menilai trade-off moderasi. Efek training S3 dibaca dari selisih Macro-F1 berpasangan terhadap S1. Penggunaan lintas platform dan ketepatan caption akan memerlukan data domain lain, anotasi caption, dan pengujian moderator.

## Reproduksi

Jalankan `cp2.recaption` di server setelah bobot revisi terkunci tersedia dalam cache Hugging Face. Urutan split adalah validation, test, lalu fit. Letakkan hasil privat di luar repositori sebagai `<split>_captions_PRIVATE.json`. Worker penyiapan menulis kode keluar ke `generation_exit`. Dari root Git worktree yang bersih, aktifkan lingkungan CP2 lalu jalankan langkah berikut. Gunakan `CP2_ALLOW_SHARED_GPU=YES` dengan izin penggunaan GPU bersama.

```bash
bash scripts/run_cp2_caption_repair.sh /path/to/cp2_data /path/to/source_study /path/to/caption_repair
```

Ringkasan agregat ditulis ke `router_results.json` dan `classifier_results.json`. Prediksi per meme dan caption tetap di penyimpanan privat. `pipeline_stage`, `pipeline.log`, dan `pipeline_exit` menunjukkan status kerja.

Setelah prediksi router tersedia, jalankan `python -m cp2.repair_caption_audit --workspace /path/to/cp2_data --study /path/to/source_study --job /path/to/caption_repair` untuk menghasilkan `caption_audit.json`. `python -m cp2.repair_confirmation --study /path/to/source_study --job /path/to/caption_repair` menghasilkan `confirmation_results.json`.
