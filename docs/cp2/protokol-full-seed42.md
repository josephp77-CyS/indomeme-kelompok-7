# Protokol full training CP2 seed 42

**Dikunci 19 November 2026 WIB untuk paket eksploratif CP2.** Joseph menugaskan penetapan protokol setelah telaah teknis dengan tenggat 20 November pukul 00.00 WIB. Catatan ini menggunakan desain CV, kontrak NLP, source integrasi `0a9edf0`, dan hasil audit sebelumnya. Persetujuan Rafa pada versi ini masih belum tercatat. Keputusan untuk mulai training mengikuti otorisasi Joseph dan ketersediaan slot GPU. [Config frozen](../../configs/cp2_nlp_protocol.json) · [PR integrasi](https://github.com/josephp77-CyS/indomeme-kelompok-7/pull/5)

## Dasar pemilihan

- INDOMEME menunjukkan bahwa IndoBERTweet dan ViT layak sebagai pasangan backbone dan bahwa dua tugas ini dapat dipelajari bersama. Pertahankan pretrained revision, fit/validation historis, dan representasi caption/OCR yang sudah diaudit. Gunakan B1 sebagai kontrol concat internal dengan aturan CP2 yang identik. Hasil paper dan CP1 menjadi konteks karena konfigurasi input, split yang dilaporkan, atau seleksi checkpoint dapat berbeda. [Pamungkas dkk., INDOMEME](https://doi.org/10.7717/peerj-cs.3736) · [Dataset card versi yang dipakai](https://huggingface.co/datasets/aiatums/INDOMEME/blob/3e0cb8cae9cd726582f69afdb122163d0eab108b/README.md)
- Gating multimodal, cross-attention, dan dropout modalitas memiliki preseden. Kontribusi CP2 yang diuji adalah perbandingan terkontrol ketiganya pada INDOMEME dengan dua tugas, termasuk penurunan skor ketika kontribusi ViT ditiadakan. Angka 0,1 dipertahankan dari rancangan CV01 agar M3 menguji satu perubahan terhadap M2 tanpa mencari nilai dari hasil validation. [Arevalo dkk., Gated Multimodal Units](https://arxiv.org/abs/1702.01992) · [ViBERT-X](https://www.sciencedirect.com/science/article/pii/S0262885626003471) · [Neverova dkk., ModDrop](https://arxiv.org/abs/1501.00102) · [Rancangan CV01](cv-design.md)
- Bekukan metrik dan kandidat sebelum official test untuk mengurangi bias seleksi. Satu seed cukup untuk paket eksploratif jika tiap perbandingan memakai seed yang sama. Penilaian stabilitas membutuhkan pengulangan pada seed tambahan. [Cawley dan Talbot, seleksi model](https://jmlr.csail.mit.edu/papers/v11/cawley10a.html) · [Reimers dan Gurevych, variasi antar-seed](https://aclanthology.org/D17-1035/)

## Aturan eksperimen

| Aspek | Keputusan |
| --- | --- |
| Data | Revisi dataset dan SHA manifest di config. Gunakan **3.214 fit / 804 validation / 1.005 official test**. Audit hash piksel historis menemukan nol grup identik lintas fit/validation dan dua sampel official test yang cocok dengan fit atau validation. |
| Input | Caption lalu OCR hingga 256 token. Keluarkan `topic`. B1–M3 memakai gambar ViT 224 × 224 dan melatih ulang kedua backbone dari bobot pretrained revisi yang sama. |
| Training | Seed 42. AdamW dengan LR 2e-5, weight decay 0,01, jadwal linear, FP32, batch fisik 1 dan akumulasi 8. Maksimal lima epoch dengan patience 2. Kedua cross-entropy berbobot 0,5. |
| Seleksi | Pilih checkpoint dengan mean **raw validation Macro-F1** dua task terbesar. Saat seri persis dalam satu run, pilih epoch paling awal. Untuk memilih satu model antarvarian, gunakan skor yang sama dan jumlah parameter total lebih kecil pada seri persis. Laporkan skor per task dan constrained sebagai hasil sekunder. |
| Baseline | Gunakan kembali B0 full seed 42 yang selesai sebelum freeze. Audit 804 ID menunjukkan skor normal **0,782072**. Config run historis cocok pada manifest, input, revisi, loss, seed, dan aturan seleksi. Catat asal run dan hash historisnya. |
| Urutan | Jalankan smoke GPU M1–M3 dan pilot M3 satu epoch. Setelah lolos, jalankan **B1 → M1 → M2 → M3 → B2**. B1 dan M1 memberi perbandingan fusi awal, M2 dan M3 menutup rangkaian ablation, lalu B2 melengkapi pembanding gated. |

## Pembandingan dan evaluasi

- **B1 vs B0** menguji penambahan jalur visual dan perubahan fusi. **M1 vs B1** menguji desain token–patch dibanding concat. **M2 vs M1** menguji gate. **M3 vs M2** menguji dropout visual. **M2 vs B2** membandingkan dua desain gated dengan kapasitas fusi berbeda.
- Untuk M3 vs M2, ukur **selisih Macro-F1 normal dan visual-off** pada ID yang sama. Laporkan skor absolut kedua kondisi per tugas agar penurunan kecil karena skor normal yang rendah tetap terlihat. Evaluasi no-caption dan no-OCR dilakukan tanpa retraining. Saat visual-off, caption dan OCR tetap bisa membawa informasi yang berasal dari gambar.
- Simpan checkpoint, hash config, manifest, log durasi, peak VRAM dan prediksi validation per ID di penyimpanan privat. Laporkan mean raw Macro-F1, Macro-F1 per task, confusion matrix, serta rescue/hurt berpasangan. Nyatakan cakupan aktual bila ada run yang belum selesai. Smoke dan pilot hanya menguji alur dan kapasitas.
- Setelah full run dan seleksi validation selesai, kunci daftar checkpoint serta kondisi evaluasi sebelum inferensi official test. Laporkan test resmi lengkap (1.005) dan subset sensitivitas (1.003) yang mengecualikan dua gambar identik lintas split berdasarkan audit historis. Evaluator saat ini mendukung validation. Implementasi dan verifikasi evaluator test adalah langkah terpisah dari training.
