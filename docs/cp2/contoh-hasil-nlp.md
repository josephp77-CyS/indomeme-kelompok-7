**Contoh hasil pekerjaan Rafa pada jalur NLP**

Saya menyiapkan contoh ini supaya hasil pekerjaan CV dan NLP bisa saling dipakai. Rafa dapat menggunakannya saat menulis protokol dan menyerahkan hasil ke bagian CV. Ini contoh format; pekerjaan, commit dan review Rafa tetap dicatat dari hasil yang benar-benar dibuat.

Rujukan kontrak terbaru adalah [cv-design.md](cv-design.md), yang memakai dua head sebagai rancangan kerja untuk review. Jika partner mengubah keputusan, kedua dokumen diselaraskan sebelum freeze.

**1. Hasil yang diharapkan untuk tugas sekarang CP2-NLP01**

Hasil utamanya adalah `docs/cp2/nlp-protocol.md`. Dokumen ini dapat dibaca CV untuk mengetahui apa yang masuk ke model, apa arti keluarannya, dan bagaimana hasil akan dinilai. Ia bukan laporan training selesai.

Contoh isi protokol yang sudah terisi secara substantif:

| Bagian | Contoh keputusan yang dituliskan Rafa |
| --- | --- |
| Tujuan | Menilai hatefulness dan appropriateness pada gambar meme dengan caption+OCR tanpa topik |
| Dua tugas | `hate`: 0 not hate, 1 hate; `appropriate`: 0 appropriate, 1 inappropriate |
| Format label batch | `labels` berbentuk `[B,2]`: kolom pertama hate, kolom kedua appropriateness |
| Sumber data | INDOMEME pada revisi yang sama dengan CV01; sumber dan checksum manifest disebutkan |
| Split | Rujukan historis 3.214 fit, 804 validation, 1.005 test; audit ulang implementasi dilakukan pada NLP02 |
| Teks | Caption lalu separator tokenizer lalu OCR; maksimum 256 token; topik tidak masuk fitur |
| Nilai kosong | Field kosong ditangani sengaja; missing value tidak diubah menjadi teks `nan` |
| Token khusus | `content_mask` memilih token aktif yang bukan anggota `tokenizer.all_special_ids`, termasuk separator literal |
| Truncation | Rencana mengukur input yang terpotong dan bagian OCR yang bertahan; hasil audit aktual tidak diisi sebelum diukur |
| B0 | Memakai input teks yang sama dengan kondisi multimodal, tanpa encoder ViT; dua head sesuai bentuk baseline multitask CV01 |
| Loss | Usulan awal 0,5 loss hate dan 0,5 loss appropriateness; ditinjau bersama sebelum training |
| Pemilihan checkpoint | Usulan rerata dua Macro-F1 validation raw; raw/constrained dilaporkan terpisah sesuai CV01 |
| Evaluasi | Metrik per task, confusion matrix, recall kelas positif, rescue/hurt; ID dan seed dipasangkan |
| Robustness | Normal, tanpa caption, tanpa OCR, tanpa jalur ViT; arti sumber yang masih tersedia ditulis |
| Test | Digunakan setelah keputusan dikunci; tidak memilih arsitektur/seed/prompt dari skor test |
| Bukti eksperimen | Run ID, commit sumber, config, checksum split, versi, status dan lokasi artefak |

Protokol yang selesai tidak hanya menulis “gunakan preprocessing” atau “evaluasi dengan F1”. Ia menetapkan pilihan sehingga CV dapat mengimplementasikannya tanpa menebak.

**2. Contoh kontrak input yang bisa langsung dipahami CV**

Ini bentuk yang direncanakan untuk batch berisi dua contoh, bukan keluaran tokenizer yang sudah dijalankan:

```text
input_ids       : [2, 256]
attention_mask  : [2, 256]
content_mask    : [2, 256]
labels          : [2, 2]
pixel_values    : [2, 3, 224, 224] untuk kondisi multimodal
```

B0 tidak memerlukan `pixel_values`. Field ID dipakai untuk pencocokan evaluasi, bukan fitur klasifikasi. Label tidak diteruskan sebagai fitur ke encoder atau gate.

Contoh label buatan untuk memeriksa pemetaan:

```text
labels = [[0, 0],
          [0, 1]]

Baris 1: not hate dan appropriate
Baris 2: not hate dan inappropriate
```

Kedua baris tersebut menunjukkan mengapa not hate belum tentu appropriate. Ini fixture buatan, bukan anotasi meme nyata atau hasil model.

Contoh mask simbolik dengan urutan caption lalu OCR:

| Posisi | BOS | token caption | SEP | token OCR | EOS | PAD |
| --- | --- | --- | --- | --- | --- | --- |
| attention_mask | 1 | 1 | 1 | 1 | 1 | 0 |
| content_mask | 0 | 1 | 0 | 1 | 0 | 0 |

Nama token di tabel bersifat simbolik; ID/format sebenarnya mengikuti tokenizer pinned. Mask pertama dipakai encoder teks. Mask kedua dipakai CV saat merata-ratakan hasil cross-attention. Jika semua token isi tidak ada, kontrak CV01 menetapkan kontribusi attention nol dan tetap finite.

**3. Bukti bahwa NLP01 siap direview dan benar-benar selesai**

| Status | Bukti yang tersedia | Langkah berikutnya |
| --- | --- | --- |
| Draf | Protokol ditulis dan keputusan utama sudah terisi | Rafa membaca ulang dan memeriksa konsistensi dengan CV01 |
| Siap review | Commit oleh Rafa, branch dipush, PR dari akun Rafa, diff serta pemeriksaan spesifikasi dijelaskan | Joseph memeriksa kontrak dan memberi temuan/approval |
| Perlu revisi | Review menemukan label, mask, scope atau policy yang belum cocok | Rafa memperbaiki pada branch/PR yang sama |
| Selesai | Temuan review beres, persetujuan aktual tercatat dan perubahan digabung ke main | Tutup issue sesuai kriteria dan mulai implementasi NLP02 |

Contoh branch dan pesan commit yang dapat digunakan saat pekerjaan benar-benar sudah tersedia:

```text
Branch: cp2/nlp-protocol
Commit: docs(nlp): define multitask inputs and evaluation [CP2-NLP01]
PR: [CP2-NLP01] Tetapkan kontrak input dan evaluasi dua task
Reviewer: Joseph/CV
```

Rafa membuat commit/push/PR dari clone dan akun miliknya. Nomor issue, SHA commit, tautan PR dan nama akun reviewer diisi dari hasil nyata. Tidak ada contoh nomor PR atau SHA buatan yang dianggap bukti pekerjaan.

Contoh isi serah terima ketika protokol siap review, bukan pesan aktual Rafa:

> Joseph, protokol NLP sudah siap direview pada branch `cp2/nlp-protocol`. Inputnya caption+OCR tanpa topik, panjang 256, dengan dua kolom label dan content mask sesuai CV01. Saya sudah menjelaskan B0, pembagian data, loss, pemilihan checkpoint dan kondisi evaluasi. Tolong periksa kompatibilitas mask serta output model. Commit dan tautan PR akan saya sertakan dari hasil push. Kode input, audit runtime dan evaluator belum dinyatakan selesai pada tugas protokol ini.

Untuk penutupan tugas setelah merge, serah terima menyebut commit yang sudah masuk main, tautan PR/review sebenarnya dan bagian yang bisa dipakai CV. Bila belum direview, gunakan status siap review.

**4. Yang Joseph periksa sebagai reviewer**

1. Kedua label memiliki arti yang sama dengan CV01; kelas 1 pada head `appropriate` berarti inappropriate.
2. Input tanpa topik, urutan caption/OCR, bentuk tensor dan label cocok dengan model.
3. Content mask mengabaikan padding dan token khusus, termasuk separator di tengah input.
4. B0 dan model multimodal mendapat sumber teks serta split yang sama.
5. Dua loss, pemilihan checkpoint raw dan pelaporan constrained tidak bertentangan dengan rancangan CV.
6. Robustness mematikan sumber yang dinyatakan; tanpa ViT masih memiliki caption.
7. Rencana pemeriksaan, hasil yang sudah ada dan hal yang belum dijalankan dibedakan secara jujur.

Jika ketujuh bagian ini jelas, CV dapat melanjutkan model tanpa menunggu seluruh training NLP selesai. Persetujuan protokol tidak berarti implementasi data/evaluator sudah teruji.

**5. Contoh hasil tugas NLP setelah protokol**

| Tugas | Bentuk hasil yang diserahkan | Bukti selesai yang diperlukan |
| --- | --- | --- |
| NLP02 input dan B0 | Penggunaan helper input yang ada, konfigurasi B0, ringkasan audit data/token dan kontrak batch | Log pemeriksaan aktual: jumlah contoh, ID/split/label, input tanpa topik, nilai kosong, truncation dan mask. Berkas baru dibuat hanya jika helper lama belum cukup |
| NLP03 evaluator | Evaluator dua task dan self-check kecil | Log fixture benar, ID ganda/hilang ditolak, Macro-F1 dan confusion matrix sesuai hitungan; jangan memakai akurasi saja |
| NLP04 robustness | Evaluasi normal/tanpa OCR/tanpa caption/tanpa ViT dan rescue/hurt per task | Bukti transformasi input, pairing ID/seed, jumlah contoh dan kebenaran kondisi visual-off |
| NLP05 hasil dan analisis | Tabel dua task, perubahan dari baseline, robustness, kasus gagal dan diskusi | Angka dapat dihitung ulang dari prediksi tersimpan; tabel merujuk run/config/source commit yang benar |

Contoh fixture untuk pemeriksa evaluator, bukan skor penelitian:

```text
truth      = [0, 0, 1, 1]
prediction = [0, 1, 1, 1]
confusion matrix = [[1, 1], [0, 2]]
accuracy = 0,75
Macro-F1 = 11/15 atau sekitar 0,733333
```

Angka tersebut dapat dipakai pada self-check satu task. Evaluator dua task juga perlu menguji urutan kolom label serta policy raw/constrained. Untuk laporan data/model nyata, cantumkan hasil yang benar-benar dijalankan beserta perintah, source commit dan artefaknya. Data, ID dan prediksi individual tetap privat sesuai kebijakan proyek; repo menerima kode, konfigurasi yang aman dan ringkasan agregat.

Langkah pertama Rafa adalah menulis `nlp-protocol.md`, memeriksa kecocokannya dengan CV01, lalu mengajukan PR dari akun sendiri. Contoh di atas belum menjadi bukti penyelesaian NLP01–NLP05. Rancangan dua head disepakati bersama sebelum konfigurasi dikunci.
