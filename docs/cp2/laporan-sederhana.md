**Rancangan CP2 INDOMEME dalam Bahasa Sederhana**

Pendamping diskusi Joseph dan Rafa · 11 November 2026

Kita sedang merancang model yang menilai kebencian dan kelayakan meme Indonesia. Tujuan pengembangan adalah menguji cara memakai gambar bersama tulisan agar keputusan model lebih berguna dan tetap dapat diperiksa. Laporan ini membantu kita memahami rancangan sebelum masuk ke detail kode.

**Posisi pekerjaan sekarang**

Reproduksi inti CP1 sudah selesai. Dokumen rancangan CV01 sudah dibuat dan menunggu review partner. Model inovasi CP2, pengujian implementasi, kalibrasi GPU dan training belum dijalankan. Penjelasan tentang manfaat metode di bawah adalah hipotesis yang akan diuji.

**Mengapa kita melanjutkan baseline**

Pada hasil CP1 yang tersedia, penambahan gambar membantu penilaian ketidakpantasan, sedangkan deteksi kebencian tidak membaik terhadap model teks. Hasil tersebut berasal dari satu seed, yaitu satu pengaturan keacakan latihan. Temuannya belum menunjukkan keunggulan yang stabil.

Pertanyaan kita adalah kapan informasi gambar membantu, kapan justru mengubah jawaban benar menjadi salah, dan bagaimana model mengatur kontribusi gambar itu.

**Dua pertanyaan untuk setiap meme**

| Penilaian | Pilihan keluaran | Pertanyaan yang dijawab |
| --- | --- | --- |
| Kebencian | Hate atau not hate | Apakah meme berlabel kebencian menurut pedoman dataset? |
| Kelayakan | Appropriate atau inappropriate | Apakah meme dinilai pantas atau tidak pantas menurut pedoman dataset? |

Kedua label berkaitan, tetapi berbeda. Dalam rilis INDOMEME, semua hate juga inappropriate. Sebagian not hate tetap inappropriate. Jadi, model yang menjawab not hate belum otomatis menyatakan meme itu pantas.

Rancangan kerja memakai dua head, yaitu dua bagian pengambil keputusan pada model yang berbagi hasil pembacaan input. Satu head menilai kebencian; satu lagi menilai kelayakan. Pilihan ini masih perlu diselaraskan dengan protokol NLP sebelum eksperimen dimulai.

**Informasi yang diberikan kepada model**

Model menerima gambar meme, OCR atau tulisan yang dibaca dari gambar, serta caption atau deskripsi gambar dari rilis dataset. Input teks CP2 tidak memakai topik anotasi manusia. Karena input berubah dari CP1, baseline teks dan gabungan perlu dilatih ulang agar perbandingannya adil. Caption tetap membawa informasi yang berasal dari gambar.

<!-- PAGE BREAK -->

**Enam model dan tujuan pembandingannya**

Huruf B menandai baseline atau kontrol pembanding. Huruf M menandai pengembangan bertahap. Encoder adalah komponen yang mengubah input menjadi kumpulan angka. Semua kondisi dalam rancangan kerja menghasilkan dua penilaian yang sama.

| Model | Cara kerjanya | Hal yang ingin diketahui |
| --- | --- | --- |
| B0 | Membaca caption dan OCR melalui encoder teks. | Seberapa baik informasi teks yang diperkaya? |
| B1 | Menggabungkan ringkasan teks dan ringkasan gambar. | Bagaimana hasil fusi baseline pada input CP2? |
| B2 | Menambahkan kontribusi ringkasan gambar yang diatur gate. | Bagaimana hasil ringkasan gambar yang diatur gate? |
| M1 | Menghubungkan token teks dengan potongan gambar melalui cross-attention. | Bagaimana hasil interaksi yang lebih terarah? |
| M2 | Menggabungkan cross-attention dengan gate. | Apakah gate membantu dibanding M1? |
| M3 | Memakai M2 dan visual dropout saat latihan. | Apakah latihan itu membantu ketika jalur ViT dimatikan? |

**Cross attention menghubungkan tulisan dan bagian gambar**

Teks diolah sebagai token atau unit teks, sedangkan encoder gambar ViT membagi gambar menjadi 49 patch atau potongan. Cross-attention memakai token teks untuk membobotkan informasi dari patch gambar. Bobot ini dipelajari saat latihan; bagian yang diberi bobot tinggi belum tentu merupakan penjelasan kausal keputusan model.

Sebagai ilustrasi, kata “hebat” dapat dibaca bersama gambar untuk memeriksa konteks pujian atau sindiran. Ini contoh penjelasan mekanisme, bukan contoh dataset, hasil prediksi, atau penetapan label hate.

**Gate mengatur besar kontribusi gambar**

Gate dapat dibayangkan seperti pengatur volume pada kontribusi visual. Nilainya dipelajari per meme: nilai kecil mengurangi tambahan visual, sedangkan nilai besar memperkuatnya. Nilai gate bukan probabilitas bahwa gambar benar. Jalur langsung teks tetap tersedia, tetapi tidak menjamin prediksi model B0 selalu dipertahankan.

**Visual dropout melatih kondisi jalur visual hilang**

Pada M3, kontribusi jalur ViT dinolkan pada sekitar 10 persen contoh saat training. File gambar tidak dihapus. Saat evaluasi normal, jalur visual aktif; saat uji tanpa ViT, kontribusinya dinolkan secara sengaja. Caption tetap tersedia sehingga model masih menerima sebagian informasi asal gambar.

Mask menentukan bagian yang dipakai atau diabaikan. Mask teks mencegah padding dan token khusus ikut dirata-ratakan. Mask visual memastikan kontribusi yang dimatikan benar-benar nol, termasuk setelah proyeksi dan bias.

<!-- PAGE BREAK -->

**Pembagian kerja dan urutan pengerjaan**

| Pekerjaan | Joseph atau CV | Rafa atau NLP |
| --- | --- | --- |
| Rancangan | Menjelaskan model, bentuk data, mask, gate dan dropout. | Memeriksa input, label, pembanding dan evaluasi. |
| Implementasi | Mengerjakan model, program pelatihan dan pemeriksaan perilaku. | Menyiapkan input, konfigurasi B0, evaluator dan uji kehilangan input. |
| Eksperimen | Mengoperasikan satu worker GPU, kalibrasi dan backup. | Memeriksa konfigurasi, validation, artefak dan tabel. |
| Analisis dan penulisan | Menjelaskan visual, metode dan biaya. | Menghitung metrik serta menulis hasil dan pembahasan. |

Kita saling mereview pekerjaan. Pembagian disiplin mengatur pemilik tugas; keduanya tetap perlu memahami seluruh alur.

**Urutan kerja yang perlu diikuti**

1. Selesaikan review CV01 dan protokol NLP. Sepakati dua label, format input, model pembanding dan aturan pemilihan model.
2. Bangun komponen bertahap. CV mengerjakan B1/B2, M1/M2, lalu M3; NLP menyiapkan input dan evaluator secara paralel.
3. Jalankan smoke test pada sedikit contoh untuk memeriksa alur. Pastikan mask, perhitungan kesalahan, pembaruan bobot, simpan dan muat model bekerja.
4. Lakukan kalibrasi dan pilot. Ukur memori, waktu, ukuran checkpoint dan penyimpanan sebelum menentukan jumlah run.
5. Kunci protokol lalu training. Fit memperbarui bobot; validation memilih checkpoint atau versi bobot tersimpan. Test dipakai untuk pelaporan setelah keputusan dikunci.
6. Hitung ulang hasil, analisis kesalahan, lalu lengkapi notebook, laporan, slide dan backup. Pengumpulan meme terbaru ditunda sampai eksperimen inti selesai.

**Cara menilai apakah pengembangan membantu**

Macro-F1 menilai keseimbangan performa antar-kelas. Laporkan skor kebencian dan kelayakan secara terpisah. Rancangan mengusulkan bobot loss atau nilai kesalahan latihan yang sama dan pemilihan checkpoint dari rerata dua Macro-F1 validation raw; raw berarti sebelum aturan koreksi antardua label diterapkan. Aturan ini perlu disepakati sebelum training.

Bandingkan M2 dengan M1 untuk gate, M2 dengan B2 untuk interaksi token–patch dalam jalur bergate, serta M3 dengan M2 untuk visual dropout. Kapasitas model berbeda dan harus dilaporkan. Catat pula rescue, yaitu jawaban salah yang diperbaiki, dan hurt, yaitu jawaban benar yang dirusak. Selisih keduanya bukan rumus perubahan Macro-F1.

Run adalah satu pelatihan satu konfigurasi dengan satu seed. Paket lengkap berisi 18 run, paket ringkas 14, dan paket minimum eksploratif enam. Setiap run melatih dua head. Pilihan paket mengikuti kapasitas nyata. Model final dipilih melalui validation; M3 tidak wajib menjadi pemenang.

<!-- PAGE BREAK -->

**Naskah penjelasan sekitar dua menit**

“Pada CP1, kita sudah menyelesaikan reproduksi inti. Hasil yang tersedia menunjukkan bahwa gambar membantu penilaian ketidakpantasan, tetapi belum membantu deteksi kebencian secara konsisten. Karena itu, kita ingin menguji cara menggabungkan informasi gambar dan tulisan dengan lebih terarah.

Model kita akan menjawab dua pertanyaan untuk setiap meme: apakah mengandung kebencian, dan apakah pantas. Dua pertanyaan ini berbeda, karena meme yang tidak berlabel hate masih bisa inappropriate.

Kita menyiapkan enam kondisi. B0 memakai teks yang berisi caption dan OCR. B1 menggabungkan ringkasan teks dan gambar. B2 menambahkan gate, yaitu pengatur kontribusi gambar. M1 memakai cross-attention untuk menghubungkan token teks dengan potongan gambar. M2 menggabungkannya dengan gate. M3 menambah latihan saat kontribusi jalur gambar sesekali dimatikan.

Pembanding ini membantu kita mengetahui komponen mana yang berguna. Kita tidak menganggap model paling lengkap pasti paling baik. Kita memeriksa hasil kedua tugas, prediksi yang diperbaiki, prediksi yang dirusak, serta biaya komputasinya.

Bagian CV mengerjakan model dan operasi eksperimen. Bagian NLP mengerjakan input, label dan evaluasi. Kita saling mereview. Sebelum training penuh, kita memastikan alur teknis benar dan mengukur kapasitas GPU. Pemilihan model menggunakan validation, lalu hasilnya dilaporkan pada test setelah keputusan dikunci.

Saat ini yang sudah tersedia adalah rancangan. Manfaat inovasi masih perlu dibuktikan melalui eksperimen. Pengumpulan meme terbaru menjadi tahap berikutnya setelah eksperimen inti selesai.”

**Pertanyaan yang mungkin muncul saat diskusi**

**Apakah kita membuat enam dataset?** Tidak. Enam kondisi memakai data dan pembagian yang sama. Yang berubah adalah cara model mengolah informasi.

**Mengapa B0 disebut teks jika memakai caption gambar?** Karena input ke encoder B0 berbentuk teks. Caption tetap membawa informasi visual yang sudah diubah menjadi deskripsi.

**Apakah gate pasti mencegah jawaban benar menjadi salah?** Belum tentu. Kita menguji manfaatnya dengan skor dan perubahan prediksi yang benar-benar terjadi.

**Apakah enam run sudah mencakup kedua label?** Ya, jika memakai rancangan dua head ini. Enam run berarti satu seed untuk masing-masing enam kondisi; variasi antar-seed belum diuji.

**Bagaimana jika baseline tetap terbaik?** Laporkan hasil tersebut dan pilih model sesuai aturan validation. Analisis yang adil tetap dapat menjelaskan batas metode pengembangan.

**Apa yang perlu disepakati sekarang?** Dua head, kontrak input dan mask, pembanding, policy validation, serta pemilik dan bukti selesai setiap tugas.

Rujukan teknis: [CP2-CV01 cv-design.md](cv-design.md). Laporan ini menyederhanakan rancangan; status review dan keputusan final mengikuti dokumen teknis yang telah disepakati.
