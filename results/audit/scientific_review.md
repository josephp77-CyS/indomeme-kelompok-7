# Pemeriksaan ilmiah paket pengumpulan

Pemeriksaan penutupan untuk paket pengumpulan 30 Oktober 2026. Artefak yang ditinjau ialah Markdown laporan, teks serta speaker notes pada 20 slide PPTX, data dua grafik native PPTX, README, REPRODUCING, dan runner `src/reproduce_core.py`. Artefak penulis tidak diubah oleh pemeriksa.

## Hasil substantif

Tidak ditemukan kesalahan angka atau kesimpulan ilmiah yang mengubah hasil. Sepuluh baris metrik pada laporan dan lampiran presentasi cocok dengan `results/core_results.csv` pada pembulatan yang ditampilkan. Kedua grafik PowerPoint memuat lima kondisi per tugas dengan nilai reproduksi dan angka paper yang sesuai. Rentang sumbu kedua grafik adalah 0 hingga 1.

Pemeriksaan mencakup hal berikut:

- Cakupan delapan konfigurasi dan sepuluh hasil tugas konsisten. RQ3/API dan seluruh grid tidak diklaim telah direproduksi.
- Kelas 1 pada tugas hate berarti hate. Kelas 1 pada tugas internal appropriate berarti inappropriate. Baris confusion matrix adalah truth dan kolom adalah prediksi, dengan urutan kelas 0 lalu 1.
- Sepuluh macro-P, macro-R, macro-F1, accuracy, confusion matrix, jumlah epoch, dan epoch pilihan sesuai bukti run. Macro-F1 dijelaskan sebagai rerata F1 kelas, bukan harmonic mean dari macro-P dan macro-R.
- E101 inappropriate utama memakai constrained 0,7981 dan raw 0,7990. E110 memakai 0,8109 pada keduanya. Penjelasan aturan maksimum antar-head sesuai implementasi.
- Delta fusi terhadap teks −0,1515 pp untuk hate dan +1,4134 pp untuk inappropriate sesuai perhitungan. Angka disagreement serta perbaikan dan kerusakan prediksi sesuai pasangan ID yang sama.
- Skor seleksi multitask dibedakan dari validation per tugas. Kandidat E059 hate dan E042 inappropriate mengikuti validation dari konfigurasi tetap. E060 tidak dipilih hanya karena test inappropriate tertinggi. Margin E042 terhadap E110 sebesar 0,01484 pp diberi batas interpretasi yang tepat.
- Pembagian 3.214 fit, 804 validation, dan 1.005 test, grouping duplikat train, serta dua grup duplikat lintas split resmi konsisten dengan audit data. Pemisahan ID tidak diklaim membuktikan ketiadaan duplikat gambar lintas train/test.
- Deviasi batch, ViT 224 versus 384, pooling, validation, early stopping, lambda, satu seed, dan anotasi topik rilis dijelaskan. Tidak ada klaim signifikansi atau penyebab semantik kesalahan yang tidak didukung anotasi.
- Pertanyaan pembuka terkait manfaat penambahan gambar secara langsung didukung pasangan eksperimen dan jawaban sementara. Narasi tidak mengubahnya menjadi klaim keunggulan multimodal secara umum.

## Pemeriksaan runner dan petunjuk reproduksi

Fungsi `frozen_config` berhasil diperiksa pada kedelapan ID. Checksum `master_training.py` dan sumber pendukung cocok dengan konfigurasi yang dibekukan. Pengujian ini hanya menginspeksi konfigurasi dan tidak mengimpor runtime training, mengunduh bobot, atau memulai training.

Runner memerlukan `--run` dan workspace baru untuk eksekusi training. Runner memakai konfigurasi beku, revision model yang tersimpan, data dengan checksum tetap, split yang cocok dengan hash konfigurasi, kemudian mengunci checkpoint sebelum evaluasi test. Petunjuk menyatakan kebutuhan environment, GPU, kalibrasi, ruang kosong, dan kemungkinan perbedaan numerik antarlingkungan. README dan REPRODUCING membedakan paket GitHub yang tidak membawa CSV rinci dari paket Drive yang membawa prediksi audit.

## Satu koreksi faktual kecil yang diminta

Bagian protokol laporan semula menyatakan bahwa history, prediksi validation, dan checkpoint resume disimpan pada setiap akhir epoch. Kode `master_training.py` sebenarnya memperbarui `validation_predictions.csv` dan `validation_metrics.json` hanya ketika skor validation membaik. History dan `last.pt` disimpan setiap epoch.

Koreksi telah dikirim kepada penulis: “Setiap akhir epoch, history dan checkpoint resume disimpan. Prediksi dan metrik validation serta checkpoint terbaik diperbarui saat skor validation membaik.” Koreksi ini telah diterapkan pada laporan akhir dan diverifikasi oleh penulis setelah render ulang. Deskripsi artefak kini sesuai kode dan angka hasil tidak berubah.

## Batas pemeriksaan

Pemeriksaan ini membaca struktur dan isi PPTX, bukan mengimpor presentasi ke Canva. Tidak ada klaim hasil impor Canva telah diperiksa. Review ini juga tidak menjalankan ulang training atau inferensi. Pemeriksaan tampilan dokumen dan slide merupakan pekerjaan penulis artefak yang terpisah.

Halaman/PDF PeerJ tidak berhasil diambil ulang karena HTTP 403. Identitas paper serta angka acuan mengikuti teks primer tersimpan dan materi tambahan yang sudah ditelusuri. Batas satu seed, privileged topic, dan duplikat lintas split tetap membatasi klaim penelitian meskipun angka dan jejak artefaknya konsisten.
