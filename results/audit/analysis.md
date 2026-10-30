# Analisis bukti reproduksi inti INDOMEME

Paket pengumpulan 30 Oktober 2026 memuat delapan konfigurasi supervised yang telah ditetapkan sebelum pelaksanaan dan menghasilkan sepuluh hasil tugas. Seluruh angka di bawah berasal dari prediksi tersimpan, dihitung ulang tanpa training atau inferensi baru. Hasil numerik cocok dengan tabel agregat hingga toleransi 1e-12. Satu konfigurasi multitask menghasilkan dua hasil tugas. Ini bukan reproduksi seluruh grid paper dan bukan eksperimen baru untuk meningkatkan skor setelah melihat test.

## Sumber dan status verifikasi

Paper acuan ialah Pamungkas EW, Wahyuni CS, Amal I, Purworini D, Rintyarna BS. 2026. *Decoding hate in memes: multimodal and multitask approaches for low-resource Indonesian social media*. PeerJ Computer Science 12:e3736. DOI [10.7717/peerj-cs.3736](https://doi.org/10.7717/peerj-cs.3736). [Halaman penerbit](https://peerj.com/articles/cs-3736/) dan PDF mengembalikan HTTP 403 pada pemeriksaan final. Teks artikel yang diambil dari browser resmi pada 18 Oktober tersedia dalam workspace dan dipakai bersama arsip materi tambahan yang telah dihash. Tidak ada klaim bahwa halaman penerbit berhasil dibaca ulang malam ini.

[Kartu dataset resmi INDOMEME](https://huggingface.co/datasets/aiatums/INDOMEME) berhasil dibuka saat pemeriksaan ini dan menyatakan lisensi CC BY 4.0. Revisi data yang dipakai proyek ialah `3e0cb8cae9cd726582f69afdb122163d0eab108b`. [CSV tambahan penerbit](https://dfzljdn9uc3pi.cloudfront.net/2026/cs-3736/1/INDOMEME_Dataset.zip) memuat anotasi dan teks. [Kode tambahan penerbit](https://dfzljdn9uc3pi.cloudfront.net/2026/cs-3736/1/Source_Codes.zip) memuat notebook penulis. Angka keluaran yang sudah ada pada notebook penulis tidak diperlakukan sebagai hasil tim.

## Data dan preprocessing yang benar benar dijalankan

Dataset berisi 5.023 meme. Split resmi terdiri atas 4.018 train dan 1.005 test. Dari train, proyek membuat 3.214 fit dan 804 validation melalui fold pertama `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` dengan stratifikasi label gabungan dan grouping tiga pasangan gambar identik dalam train. Seluruh delapan run memakai himpunan ID fit dan validation yang identik. Tidak ada ID yang beririsan antara fit, validation, dan test.

| Kelompok label | Train resmi | Test resmi | Total |
|---|---:|---:|---:|
| Hate dan inappropriate | 1.386 | 347 | 1.733 |
| Not hate dan inappropriate | 345 | 86 | 431 |
| Not hate dan appropriate | 2.287 | 572 | 2.859 |
| Total | 4.018 | 1.005 | 5.023 |

Audit lokal terdahulu berhasil mendekode semua gambar dan mencocokkan data dengan CSV tambahan. Label hate dan appropriateness lengkap. Kolom topik kosong pada 179 train dan 49 test. Nilai ini menjadi string kosong untuk input model, dan menjadi kategori none/others hanya untuk tabulasi statistik. Topik bersifat multilabel, sehingga total kemunculan topik tidak sama dengan jumlah meme.

Proyek memakai OCR, caption, dan anotasi topik rilis. Proyek tidak mengulang koleksi, anotasi, OCR, atau generasi caption. Untuk varian yang dinamai Caption+OCR+Topic, implementasi menggabungkan **topic, caption, OCR** dalam urutan tersebut, memisahkan komponen tidak kosong dengan token `[SEP]` IndoBERTweet. Tokenizer melakukan padding dan truncation hingga 256 token. Pipeline transformer tidak melakukan stemming atau penghapusan stopword. Gambar dikonversi menjadi RGB lalu diproses oleh image processor dari revisi ViT yang dipatok.

Topik rilis merupakan input anotasi yang sudah disediakan. Hasil ini belum menggambarkan sistem produksi yang harus menebak topik meme baru secara otomatis.

Audit byte gambar dan piksel menemukan lima grup duplikat identik. Dua grup melintasi train dan test resmi, mencakup empat baris atau dua meme test. Split resmi dipertahankan untuk keterbandingan. Grouping train/validation tidak menghapus duplikat train/test. Karena itu klaim bebas kebocoran harus dibatasi pada pemisahan ID dan grouping duplikat dalam train/validation. Near-duplicate seperti crop dan recompression belum diuji.

## Implementasi dan protokol

ViT memakai `google/vit-base-patch32-224-in21k` revisi `98cb6f7099d372694c8547bf7002289b36857dc7`. Teks memakai `indolem/indobertweet-base-uncased` revisi `32e28c05b47e33b6675d2670a1745c50a65e987a`. Seluruh parameter encoder ikut fine-tuning. Model gambar dan teks single-task menghasilkan satu head klasifikasi dua kelas. Fusi menggabungkan fitur kedua encoder, kemudian proyeksi linear dan dropout sebelum head. Multitask menggunakan representasi bersama dengan dua head.

Konfigurasi tetap: seed 42, maksimum lima epoch, AdamW dengan learning rate 2e-5 dan weight decay 0,01, dropout 0,3, precision FP32, scheduler linear tanpa warmup, dan early stopping patience dua. Run historis memakai batch fisik 2 dengan akumulasi 4, sedangkan adaptasi lab memakai batch fisik 1 dengan akumulasi 8. Batch efektif nominal sama-sama sekitar delapan; hal ini tidak menjamin urutan pembaruan identik.

Loss single-task ialah cross-entropy. Loss multitask ialah 0,5 kali loss hate ditambah 0,5 kali loss inappropriate. Bobot ini kandidat eksplisit yang belum menjalani tuning lambda seperti uraian paper. Checkpoint single-task dipilih dengan macro-F1 validation tertinggi. Checkpoint multitask dipilih dengan rerata macro-F1 validation hate dan macro-F1 validation inappropriate constrained. Angka `selection_validation_macro_f1` multitask adalah rerata tersebut, bukan skor satu tugas.

Deviasi yang harus dinyatakan: pembentukan validation dari train, early stopping, warmup nol, perubahan batch/akumulasi, revisi encoder yang dipatok sendiri, perbedaan ViT 224 artikel dengan checkpoint 384 pada source penulis, serta detail pooling/proyeksi yang belum sepenuhnya konsisten antara uraian artikel dan source. Ini reproduksi terarah dengan deviasi tercatat, bukan pelaksanaan identik dengan penulis.

## Hasil test

Kelas positif pada tugas hate ialah **hate**. Nama kolom tugas `appropriate` mengikuti implementasi, tetapi kelas positif 1 berarti **inappropriate** dan kelas 0 berarti appropriate. Macro-F1 menghitung rerata F1 kedua kelas. Selisih di tabel merupakan poin persentase, bukan perubahan relatif.

| ID | Tugas | Validation per tugas | Test macro F1 | Paper | Selisih pp | Epoch dipilih dari dilatih |
|---|---|---:|---:|---:|---:|---:|
| E005 | Hate gambar | 0,6407 | 0,6408 | 0,630 | +1,08 | 3 dari 5 |
| E006 | Inappropriate gambar | 0,6348 | 0,6315 | 0,630 | +0,15 | 2 dari 4 |
| E041 | Hate teks | 0,8208 | 0,8058 | 0,800 | +0,58 | 2 dari 4 |
| E042 | Inappropriate teks | 0,7924 | 0,7984 | 0,788 | +1,04 | 2 dari 4 |
| E059 | Hate fusi | 0,8293 | 0,8043 | 0,820 | −1,57 | 2 dari 4 |
| E060 | Inappropriate fusi | 0,7849 | 0,8125 | 0,809 | +0,35 | 2 dari 4 |
| E101 | Hate teks multitask | 0,8117 | 0,8043 | 0,811 | −0,67 | 5 dari 5 |
| E101 | Inappropriate teks multitask constrained | 0,7797 | 0,7981 | 0,815 | −1,69 | 5 dari 5 |
| E110 | Hate fusi multitask | 0,8277 | 0,8029 | 0,801 | +0,19 | 3 dari 5 |
| E110 | Inappropriate fusi multitask constrained | 0,7922 | 0,8109 | 0,803 | +0,79 | 3 dari 5 |

E059 dan E006 berhenti normal pada epoch keempat setelah tidak ada perbaikan validation selama dua epoch. Ini early stopping, bukan kegagalan training. E101 dan E110 mencapai batas lima epoch. Checkpoint terakhir tidak otomatis menjadi checkpoint terbaik.

## Analisis modalitas dan multitask

Perbandingan teks E041 dan fusi E059 untuk hate tidak menunjukkan kenaikan macro-F1 pada run ini. Fusi mengoreksi 19 meme yang salah pada teks, tetapi merusak 22 prediksi yang sebelumnya benar. Keduanya benar pada 809 meme dan keduanya salah pada 155 meme. Total 41 prediksi berbeda. Fusi memiliki tiga prediksi benar lebih sedikit dan macro-F1 0,1515 pp lebih rendah. Recall hate naik dari 250/347 atau 72,05% menjadi 255/347 atau 73,49%, tetapi false positive naik dari 77 menjadi 85. Ini tradeoff, bukan peningkatan seragam.

Untuk inappropriate, fusi E060 mengoreksi 33 kesalahan teks E042 dan merusak 22 prediksi yang sebelumnya benar. Keduanya benar pada 789 meme dan keduanya salah pada 161. Total 55 prediksi berbeda. Fusi memiliki 11 prediksi benar lebih banyak dan macro-F1 naik 1,4134 pp. Recall inappropriate naik dari 302/433 atau 69,75% menjadi 326/433 atau 75,29%, sementara false positive naik dari 63 menjadi 76.

Multitask tidak otomatis meningkatkan skor. E101 dibanding single-task teks turun 0,1460 pp pada hate dan 0,0348 pp pada inappropriate constrained. E110 dibanding single-task fusi turun 0,1401 pp pada hate dan 0,1662 pp pada inappropriate. Ini pengamatan satu seed. Tidak ada interval kepercayaan atau uji signifikansi yang membuktikan perbedaan tersebut stabil.

## Confusion matrix dan kelompok kesalahan

Urutan matriks adalah baris label sebenarnya 0 lalu 1, dan kolom prediksi 0 lalu 1. Karena itu tabel berikut memakai TN, FP, FN, TP, dengan arti kelas positif sebagaimana dijelaskan di atas.

| ID tugas | TN | FP | FN | TP | Recall kelas positif |
|---|---:|---:|---:|---:|---:|
| E005 hate | 504 | 154 | 169 | 178 | 51,30% |
| E006 inappropriate | 402 | 170 | 191 | 242 | 55,89% |
| E041 hate | 581 | 77 | 97 | 250 | 72,05% |
| E042 inappropriate | 509 | 63 | 131 | 302 | 69,75% |
| E059 hate | 573 | 85 | 92 | 255 | 73,49% |
| E060 inappropriate | 496 | 76 | 107 | 326 | 75,29% |
| E101 hate | 586 | 72 | 102 | 245 | 70,61% |
| E101 inappropriate constrained | 490 | 82 | 115 | 318 | 73,44% |
| E110 hate | 578 | 80 | 97 | 250 | 72,05% |
| E110 inappropriate constrained | 492 | 80 | 105 | 328 | 75,75% |

Kelompok **not hate tetapi inappropriate** hanya berisi 86 meme test dan tetap sulit bagi model appropriateness: E042 benar pada 48/86, E060 51/86, E101 constrained 47/86, dan E110 constrained 50/86. Kelompok hate dan inappropriate lebih mudah terdeteksi sebagai inappropriate: E042 254/347, E060 275/347, E101 271/347, E110 278/347. Perbedaan ini menunjukkan perlunya audit khusus konten inappropriate tanpa hate. Angka tersebut belum menjelaskan penyebab semantiknya.

Aturan constrained memaksakan hate yang diprediksi menjadi inappropriate melalui operasi maksimum. Pada checkpoint E101, aturan ini mengubah satu prediksi test yang semula benar menjadi salah. Macro-F1 inappropriate raw 0,7990196879 turun menjadi constrained 0,7980653604. Pada E110, tidak ada prediksi test berubah sehingga kedua skor sama. Pada validation E110 terdapat satu perubahan yang memperbaiki prediksi. Kebijakan constrained tetap dipertahankan seperti protokol semula, tidak dipilih ulang berdasarkan mana yang menguntungkan test.

`research_facts.json` memuat tiga contoh ID untuk setiap arah kesalahan pada setiap hasil tugas, beserta truth, prediksi, dan probabilitas positif. Contoh dipilih dengan urutan ID, bukan penilaian subjektif kesulitan. Gambar dan teks sensitif tidak disalin. Belum ada anotasi kualitatif yang dapat memastikan kesalahan disebabkan sarkasme, OCR, ambiguitas atau konteks budaya.

## Kandidat berdasarkan validation

Di antara delapan konfigurasi tetap pada checkpoint yang sudah terpilih, kandidat hate dengan validation macro-F1 tertinggi adalah **E059 0,8292910**. Runner-up E110 0,8276533, selisih 0,1638 pp. Kandidat inappropriate adalah **E042 0,7923554**, sangat dekat E110 constrained 0,7922070 dengan selisih 0,01484 pp. Test kandidat ini masing-masing 0,8042856 dan 0,7984131.

Aturan lintas konfigurasi ini dituliskan secara eksplisit saat penutupan laporan sebagai interpretasi bukti validation dari konfigurasi yang sudah ditetapkan, bukan perubahan training atau pencarian dari test. E060 memang memiliki macro-F1 test inappropriate tertinggi, tetapi validation-nya 0,7848898. Karena itu E060 tidak dijadikan kandidat final hanya karena melihat test. Selisih validation yang sangat kecil, satu seed, dan deviasi implementasi tidak mendukung klaim pemenang yang stabil. Model tetap berstatus kandidat riset, belum sistem moderasi yang tervalidasi untuk penggunaan nyata.

## Batas dan pekerjaan berikutnya

Cakupan empiris selesai untuk audit data RQ1 dan konfigurasi terpilih RQ2/RQ4. RQ3/API, semua grid encoder dan input, sweep seed/lambda, anotasi sarkasme, serta metode novelty tidak dilaksanakan. Langkah riset berikutnya yang dapat diusulkan ialah audit kualitatif berpedoman, evaluasi beberapa seed, penanganan near-duplicate dalam protokol baru yang terdokumentasi, dan pengujian tanpa anotasi topik rilis. Semua ini merupakan rencana lanjutan, bukan hasil yang sudah tersedia.

Bukti backup dan uji pembukaan checkpoint berada pada laporan terpisah `lab-dashboard/transit/audit-2026-10-29`. Status backup tidak mengubah nilai metrik ilmiah. Pemeriksaan ulang langsung server dapat tertunda ketika port SSH tidak terjangkau; seluruh analisis penutupan ini memakai artefak lokal yang sudah diperiksa checksum-nya.
