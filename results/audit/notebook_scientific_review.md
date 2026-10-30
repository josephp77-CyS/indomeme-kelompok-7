# Pemeriksaan ilmiah notebook lengkap

Pemeriksaan akhir memakai notebook sumber dan hasil eksekusi aktual di server lab pada 30 Oktober 2026. Notebook terdiri atas 81 sel, termasuk 52 sel kode. Pemeriksa membaca source, hasil eksekusi, HTML, receipt, serta tiga contact sheet grafik. Pemeriksa tidak mengubah notebook dan tidak menjalankan training atau inferensi baru.

## Hasil pemeriksaan aktual

**Pemeriksaan ilmiah dan keluaran server lulus. Tidak ditemukan penghalang ilmiah untuk memakai versi ini sebagai notebook analisis lengkap atas artefak eksperimen tersimpan.**

Receipt mencatat host labkc-comm1, Linux, Python 3.12.3, durasi eksekusi 17,0 detik, dan 52 sel kode selesai. Execution count tersusun tepat dari 1 sampai 52 dan tidak terdapat keluaran error. SHA-256 sumber dan hasil cocok dengan receipt:

- Sumber yang dipertahankan dalam source/Notebook_Lengkap_Eksperimen_INDOMEME.source.ipynb: 493c8c190daab1e9139acf01a6abbdbcb786843075078778009283bf74c5da5f.
- Hasil Notebook_Lengkap_Eksperimen_INDOMEME.executed.ipynb: e644d5da2388e4bc0c1ef0e3afc10b73cf66ebbfb827b67fc30b7a753e11fc3e.
- Notebook utama juga memuat salinan hasil beroutput tersebut, bukan versi kosong sebelum eksekusi.

Hasil notebook memuat **41 keluaran gambar PNG dan 152 keluaran HTML**, termasuk tabel dan tampilan source. Field visual_outputs pada receipt menghitung objek display secara lebih luas, sehingga nilainya tidak dinyatakan sebagai jumlah grafik. HTML memuat seluruh 14 insight baseline dan pesan penyelesaian sel analisis terakhir.

## Cakupan dan angka

- Delapan konfigurasi inti dan sepuluh hasil test dibedakan dari empat belas baseline validation, E001 historis yang memiliki test, serta E002 parsial yang tidak memiliki test.
- Empat belas baseline dihitung ulang pada validation 804, dengan ID dan label cocok terhadap metadata. Setiap B01 sampai B14 menampilkan empat metrik, metrik per kelas, confusion matrix, contoh FP/FN yang tersedia, dan insight berbahasa Indonesia. Majority diberi penjelasan recall kelas positif nol. Varian topik diberi batas penggunaan anotasi rilis.
- Sepuluh hasil tugas inti memiliki konfigurasi, history, epoch pilihan, grafik loss dan validation, perhitungan validation/test, confusion matrix, metrik per kelas, contoh kesalahan, perbandingan paper, dan insight per eksperimen. Fungsi metrik manual dibandingkan terhadap scikit-learn dan nilai referensi yang sudah diaudit.
- Kelas 1 pada hate berarti hate. Kelas 1 pada tugas internal appropriate berarti inappropriate. Confusion matrix memakai baris truth dan kolom prediksi. Penjelasan ini konsisten dengan perhitungan serta gambar yang diperiksa.
- Kriteria checkpoint multitask adalah rerata validation kedua tugas. Pemilihan kandidat menggunakan validation per tugas pada checkpoint yang sudah dipilih. E059 hate dan E042 inappropriate tidak ditentukan melalui peringkat test.
- Delta fusi terhadap teks, pertukaran kesalahan pada pasangan ID yang sama, perubahan raw/constrained, serta kelompok not hate tetapi inappropriate sesuai bukti terhitung. Penjelasan tidak mengklaim signifikansi atau penyebab semantik yang belum dianotasi.
- E001 memuat history, metrik validation/test, confusion matrix dan contoh kesalahan. E002 menampilkan tiga epoch yang tersimpan, validation terbaik pada epoch 2, serta status tidak memiliki hasil test.
- Screening pHash dinyatakan sudah dilakukan. Sebanyak 197 pasangan adalah kandidat kemiripan, bukan seluruhnya duplikat terkonfirmasi. Review manual masih belum selesai dan split resmi tidak diubah.

## Kalibrasi kegagalan dan source

Empat catatan kalibrasi lab dibaca dari artefak SSH kelompok sendiri dan diperiksa pada notebook. Assert memeriksa kelulusan, satu optimizer step, delapan accumulation step, simpan/muat checkpoint, panjang input, waktu dan VRAM positif, serta test tidak dievaluasi. E006 yang hanya menerima gambar dibedakan dari kondisi dengan teks panjang 256.

Kalibrasi lokal/Colab serta cek runtime tetap dipisahkan dari hasil test. Log CUDA OOM historis tidak diberi metrik yang dibuat-buat. Lima percobaan LLaVA/Qwen historis tidak dinyatakan sebagai hasil penelitian RQ3 lengkap. Konfigurasi tanpa response tidak dipakai untuk mengarang skor atau penyebab error.

Implementasi data, model, loss, training dan evaluasi ditampilkan sebagai source melalui AST. Fungsi training/inferensi tersebut tidak dipanggil. Eksekusi yang dibuktikan di server adalah perhitungan ulang analisis dari prediksi dan artefak beku, bukan pelatihan ulang bobot atau eksekusi API.

## Temuan awal yang sudah diselesaikan

Versi awal belum menampilkan insight per baseline. Versi akhir menampilkan seluruh empat belas insight beserta empat metrik, metrik per kelas, matriks dan contoh kesalahan. Pengantar yang semula menyebut “percobaan API” juga sudah diganti menjadi “percobaan inferensi RQ3 historis”, sesuai bukti lokal yang ditampilkan.

## Pemeriksaan visual dan batas QA

Contact sheet 2, 3, dan 5 dibuka untuk memeriksa baseline confusion matrix, loss dan checkpoint validation, confusion matrix core, E001/E002 historis, serta grafik delta dan perbandingan paper. Angka matriks, orientasi truth/prediksi, label, dan tren yang terlihat sesuai data. Pemeriksaan ini mencakup gambar yang sama dengan versi akhir; perubahan versi terakhir menambah tabel dan narasi, bukan mengganti data grafik.

Review ini tidak mengulang model training/inference dan tidak membuktikan reproduksi penuh paper, beberapa seed, atau penyebab semantik kesalahan. Pemeriksaan ini juga belum menilai eksekusi lokal terpisah setelah hasil server. Bila dijalankan lagi di laptop, bukti host dan waktu lokal harus disimpan sebagai receipt tambahan agar tidak menggantikan jejak eksekusi server.

