# Memahami Hasil Eksperimen Meme Indonesia

Kelompok 7 · Joseph Pratama Lase dan Rafa Maritza Hanasaputri · Intelligent Systems Lab FILKOM 2026

30 Oktober 2026

Jika meme memadukan gambar dan teks, apakah tambahan gambar selalu membantu komputer menilainya?

Dalam eksperimen kami, jawabannya bergantung pada pertanyaan yang diajukan. Model gabungan memperoleh hasil lebih baik pada penilaian konten tidak pantas, tetapi hasil deteksi kebenciannya tidak membaik dibandingkan model teks. Temuan ini menarik karena menambah informasi dan membuat model lebih rumit ternyata belum menjamin hasil yang lebih baik.

Ada konteks penting di balik jawaban itu. Model yang kami sebut model teks juga menerima tulisan hasil pembacaan gambar dan deskripsi gambarnya. Jadi, perbandingan ini mengukur manfaat tambahan membaca gambar secara langsung di atas informasi teks yang sudah diperkaya. Hasilnya belum membuktikan bahwa gambar pada meme tidak penting.

## Dua penilaian yang berbeda

Bayangkan sebuah gambar orang tertawa. Gambar itu dapat menjadi candaan biasa, tetapi pesan yang menyertainya dapat mengubah maknanya. Ini hanya ilustrasi tentang hubungan gambar dan teks, bukan contoh yang diambil dari dataset atau prediksi eksperimen.

Penelitian ini mengikuti dua penilaian pada dataset INDOMEME. Pertama, apakah sebuah meme mengandung kebencian menurut label penyusun dataset? Kedua, apakah meme tersebut tidak pantas menurut pedoman anotasinya? Kedua pertanyaan berhubungan, tetapi jawabannya tidak selalu sama. Dalam data yang digunakan, semua meme berlabel mengandung kebencian juga berlabel tidak pantas. Namun, ada meme tidak pantas yang tidak berlabel kebencian.

Perbedaan ini penting. Komputer yang cukup baik menemukan kebencian belum tentu cukup baik mengenali seluruh konten tidak pantas. Kami ingin melihat apakah cara memberikan informasi kepada model, serta meminta model menjawab satu atau dua pertanyaan, mengubah hasilnya. Penilaian di sini mengikuti label dataset, dengan batas konteks dan pedoman penyusunnya. [1]

## Bahan yang dipakai dan cara belajar

Kami menggunakan dataset INDOMEME yang sudah dirilis peneliti, bukan mengumpulkan meme baru. Rilis yang digunakan memuat **5.023 meme** beserta gambar, teks, deskripsi, topik, dan labelnya. Versi data dipatok agar eksperimen dapat ditelusuri. Dataset card mencantumkan lisensi CC BY 4.0. [2]

Pembagian data dapat dipahami seperti latihan, simulasi ujian, dan ujian akhir. Sebanyak **3.214 meme untuk latihan** digunakan agar model menyesuaikan pola yang dipelajari. **804 meme untuk simulasi**, disebut validation, membantu memilih hasil latihan yang akan dibawa ke ujian. Sisanya, **1.005 meme untuk ujian**, disebut test, digunakan setelah versi hasil latihan pada tiap percobaan ditetapkan.

Data latihan dan simulasi berasal dari bagian train resmi yang berjumlah 4.018 meme. Ujian tetap mengikuti 1.005 meme test resmi. Pemisahan peran ini mencegah kami terus mengubah model hanya untuk mengejar nilai pada soal ujian yang sama.

![Alur data dan eksperimen](figures/alur_data_dan_eksperimen.png)

*Gambar 1. Data latihan mengubah model, data simulasi membantu memilih hasil latihan, dan data ujian mengukur hasil pilihan tersebut. Semua bagian berisi meme nyata dari rilis. Semua metode inti menggunakan pembagian data yang sama.*

Sebelum pelatihan, kami memeriksa kecocokan label, gambar yang dapat dibaca, teks yang tersedia, dan gambar duplikat. Tulisan serta deskripsi gambar memakai hasil yang telah disediakan pada rilis. Kami tidak menjalankan ulang pembuatan deskripsi gambar atau anotasi manusia.

## Tiga cara memberikan informasi kepada model

Kami memakai model yang sebelumnya telah belajar dari data lain, lalu menyesuaikan kemampuannya menggunakan data meme Indonesia. Proses penyesuaian ini disebut fine-tuning. Model belajar hubungan antara pola masukan dan label, sehingga keberhasilannya dinilai melalui prediksi yang dapat diperiksa, bukan dengan menganggapnya memahami meme seperti manusia.

Tiga jenis masukan dibandingkan:

- **Gambar saja.** Model ViT menerima gambar meme secara langsung.
- **Teks yang diperkaya.** Model IndoBERTweet menerima OCR, caption, dan topik. OCR adalah tulisan yang dibaca dari gambar. Caption adalah deskripsi gambar yang tersedia pada rilis. Topik adalah anotasi tentang pokok pembicaraan meme.
- **Gabungan gambar dan teks.** Model menggabungkan representasi ViT dan IndoBERTweet. Penggabungan ini disebut fusi.

Karena caption sudah menjelaskan gambar dan OCR sudah memindahkan tulisan dari gambar menjadi teks, jalur teks telah membawa sebagian informasi visual. Fusi menambahkan jalur gambar langsung di atas bekal tersebut. Ini lebih tepat daripada membayangkan perbandingannya sebagai komputer yang sama sekali tidak mendapat informasi gambar melawan komputer yang mendapat gambar.

Paket inti berisi **delapan pengaturan eksperimen**, dengan sepuluh hasil tugas karena dua pengaturan menjawab kedua pertanyaan sekaligus. Empat belas pembanding sederhana juga pernah dijalankan pada data simulasi. Hasil historis lain dan percobaan parsial dicatat terpisah. Semuanya tersedia dalam notebook lengkap, dengan status yang membedakan hasil selesai, hasil simulasi, dan proses yang belum selesai. [3]

## Apa yang berubah pada hasil ujian

Gambar berikut membandingkan model teks dengan model gabungan pada 1.005 meme ujian yang sama. Angka menunjukkan **jumlah prediksi yang cocok dengan label dataset**.

![Jumlah prediksi benar](figures/jumlah_prediksi_benar.png)

*Gambar 2. Pada kebencian, model teks benar pada 831 meme dan model gabungan pada 828 meme. Pada konten tidak pantas, hasilnya berubah dari 811 menjadi 822 prediksi benar. Kedua pertanyaan memakai 1.005 meme ujian yang sama.*

Dalam perbandingan ini, model gabungan menghasilkan tiga jawaban benar lebih sedikit untuk kebencian dan sebelas jawaban benar lebih banyak untuk konten tidak pantas. Hasil ini memperlihatkan perbedaan hasil dua penilaian, dengan manfaat tambahan gambar yang belum dapat digeneralisasi ke semua pengaturan.

Angka keseluruhan juga menyembunyikan pertukaran kesalahan. Pada penilaian konten tidak pantas, model gabungan memperbaiki 33 kesalahan model teks, tetapi 22 jawaban yang semula benar menjadi salah. Jadi, peningkatan sebelas jawaban berasal dari perbaikan dan kemunduran pada meme yang berbeda. Daftar kesalahan tersebut dapat dilacak di notebook untuk pemeriksaan berikutnya.

Jumlah benar dibagi jumlah soal disebut accuracy. Sebagai contoh, 831 jawaban benar dari 1.005 soal setara dengan accuracy sekitar 82,69%. Angka ini mudah dipahami, tetapi belum menjelaskan apakah model sama baiknya pada kedua kelas.

## Mengapa jumlah benar belum cukup

Pembanding paling sederhana pada data simulasi memberi contoh nyata. Model ini selalu menjawab kelas yang paling sering muncul. Ia memperoleh **531 jawaban benar dari 804 meme**, tetapi **tidak menemukan satu pun dari 273 meme berlabel kebencian**. Nilai jumlah benarnya terlihat lumayan karena banyak meme memang tidak berlabel kebencian.

Untuk melihat ketimpangan seperti itu, kami juga menggunakan Macro-F1. Secara sederhana, skor F1 mempertimbangkan kemampuan menemukan anggota suatu kelas dan ketepatan ketika menyebut kelas tersebut. Macro-F1 menghitung skor untuk masing-masing kelas, lalu meratakannya dengan bobot yang sama. Dengan demikian, kelas yang jumlahnya lebih sedikit tetap mendapat perhatian.

**Macro-F1 bukan persentase jawaban benar.** Skor 0,81 tidak berarti 81% meme dijawab dengan benar. Jumlah benar tadi dan Macro-F1 menjawab pertanyaan yang berbeda, sehingga keduanya perlu dibaca bersama.

![Skor seimbang Macro F1](figures/skor_seimbang_macro_f1.png)

*Gambar 3. Macro-F1 memakai skala 0 sampai 1. Untuk kebencian, skor teks 0,8058 menjadi 0,8043 setelah gambar ditambahkan. Untuk konten tidak pantas, skor naik dari 0,7984 menjadi 0,8125. Nilai yang lebih tinggi menunjukkan hasil lebih baik menurut ukuran ini.*

Pada ukuran ini, skor model gabungan sekitar 0,0015 lebih rendah untuk kebencian dan 0,0141 lebih tinggi untuk konten tidak pantas dibandingkan model teks. Ini searah dengan perubahan jumlah benar. Meski demikian, eksperimen ini belum cukup untuk menyatakan bahwa pola tersebut pasti berulang setiap kali model dilatih.

## Dua pertanyaan sekaligus dan pilihan kandidat

Kami juga mencoba satu model yang belajar menjawab dua pertanyaan sekaligus. Pendekatan multitask ini mirip menggunakan bekal pengetahuan bersama untuk dua mata pelajaran yang berkaitan. Model tetap memiliki keluaran terpisah untuk kebencian dan konten tidak pantas.

Pada percobaan yang tersedia, cara tersebut belum meningkatkan hasil secara seragam dibandingkan model yang mempelajari tiap tugas secara terpisah. Hubungan antartugas dapat membantu, tetapi juga membuat model membagi kemampuan yang sama. Penyebab pastinya belum diuji melalui eksperimen khusus.

Pilihan kandidat tetap mengikuti nilai simulasi, bukan mencari nilai ujian tertinggi. Dengan dasar itu, kandidat kebencian adalah **E059, model gabungan**, sedangkan kandidat konten tidak pantas adalah **E042, model teks yang diperkaya**. Model gabungan E060 memang memperoleh nilai ujian konten tidak pantas tertinggi dalam paket inti, tetapi nilai simulasinya lebih rendah daripada E042.

Pada konten tidak pantas, selisih nilai simulasi antara E042 dan kandidat terdekat sangat kecil. Karena itu, pilihan ini berguna untuk menetapkan kandidat dari bukti yang ada, dengan kestabilannya masih perlu diuji. Kode eksperimen tersedia untuk melihat kriteria pemilihan tersebut.

## Batas hasil dan langkah penelitian berikutnya

Seluruh hasil inti memakai satu seed, yaitu satu pengaturan pengacakan untuk proses belajar. Ibarat mencoba satu susunan latihan, hasilnya belum menunjukkan seberapa besar nilai berubah pada susunan lain. Kami belum membuat klaim bahwa selisih kecil antarmodel sudah stabil atau bermakna secara statistik.

Topik anotasi pada rilis juga memberi informasi tambahan kepada model. Jika model hendak dipakai pada meme baru, perlu ditentukan siapa yang menyediakan topik atau bagaimana topik dihasilkan. Hasil saat ini belum mengukur sistem lengkap yang melakukan semua langkah tersebut secara otomatis.

Pada data, dua kelompok gambar identik melintasi bagian train dan test resmi. Pemeriksaan kemiripan gambar juga menghasilkan 197 pasangan kandidat, dengan peninjauan manual yang belum selesai. Kandidat kemiripan tidak otomatis merupakan duplikat yang terkonfirmasi. Temuan ini perlu diperhitungkan ketika menafsirkan nilai ujian.

Beberapa keputusan pelatihan disesuaikan dengan sumber daya dan implementasi yang tersedia. Contohnya, ViT kami menerima gambar berukuran 224 × 224 piksel, sedangkan bagian kode penulis memakai 384 × 384 piksel. Pembentukan data simulasi dan pengaturan pelatihan juga memiliki deviasi yang dicatat. Maka, hasil kami disebut reproduksi eksperimen inti terpilih, dengan batas perbandingan terhadap paper tetap dijelaskan.

Langkah berikutnya adalah meninjau kesalahan secara terstruktur. Peneliti dapat memeriksa meme yang hanya benar pada salah satu model, lalu mencatat apakah tulisan sulit dibaca, deskripsi kurang membantu, atau konteksnya perlu penjelasan. Itu masih pertanyaan pemeriksaan, bukan penyebab kesalahan yang sudah terbukti. Pengulangan dengan beberapa seed dan pengujian tanpa topik anotasi kemudian dapat dirancang sebelum mengambil keputusan penggunaan.

## Kesimpulan

Eksperimen ini menunjukkan bahwa menambahkan jalur gambar langsung belum otomatis memperbaiki semua penilaian meme. Pada data dan pengaturan yang digunakan, manfaatnya terlihat pada konten tidak pantas, sedangkan deteksi kebencian tidak membaik. Jalur teks yang sudah membawa OCR, deskripsi gambar, dan topik menjadi konteks penting bagi temuan tersebut.

Paket inti delapan konfigurasi telah selesai dan hasilnya dapat ditelusuri melalui notebook beroutput. Kesimpulan saat ini adalah temuan penelitian awal yang memiliki batas, bukan bukti sistem siap digunakan untuk mengambil keputusan moderasi nyata. RQ3, yang membandingkan model multimodal tanpa pelatihan khusus pada tugas ini, serta pengembangan metode baru belum diselesaikan dalam paket aktif.

## Istilah singkat

| Istilah | Arti dalam tulisan ini |
| --- | --- |
| Dataset | Kumpulan meme dan informasi pendamping yang digunakan dalam penelitian. |
| Model | Sistem yang mempelajari pola masukan untuk menghasilkan prediksi label. |
| Fine-tuning | Menyesuaikan model yang telah belajar sebelumnya menggunakan data tugas ini. |
| Validation | Data simulasi untuk memilih hasil latihan sebelum ujian akhir. |
| Accuracy | Proporsi prediksi yang benar dari seluruh contoh yang diuji. |
| Macro-F1 | Rata-rata skor F1 tiap kelas dengan bobot setara, bukan persentase jawaban benar. |

## Bukti dan bacaan

[1] Pamungkas dkk. (2026), *Decoding hate in memes: multimodal and multitask approaches for low-resource Indonesian social media*. PeerJ Computer Science. https://peerj.com/articles/cs-3736/ DOI 10.7717/peerj-cs.3736.

[2] Dataset INDOMEME dan keterangan lisensi. https://huggingface.co/datasets/aiatums/INDOMEME

[3] Notebook lengkap dengan keluaran analisis dan data pendamping. https://drive.google.com/drive/folders/1c7cuGs_ij6ghZxLF21D6GZ1Hk3a8nplH

Kode dan progres proyek tersedia pada repository privat GitHub Kelompok 7. https://github.com/josephp77-CyS/indomeme-kelompok-7 Akses mengikuti izin pemilik repository.
