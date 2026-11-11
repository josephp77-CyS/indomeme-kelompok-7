# CP2-NLP01 — Kontrak input, label, baseline, dan evaluasi INDOMEME

**Status:** DRAF REVIEW — bukan hasil training, bukan protokol yang telah disetujui  
**Tanggal draf:** 11 November 2026  
**Peran:** NLP (Rafa); reviewer: CV (Joseph)  
**Rujukan:** [`docs/cp2/cv-design.md` pada branch `cp2/cv-design`](https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/cp2/cv-design/docs/cp2/cv-design.md); `src/master_training.py`, `src/master_workflow.py`, `results/runs/E101/config.json`, `results/runs/E110/config.json`.

> Dokumen ini dibuat sebagai draf serah-terima untuk CP2-NLP01. Data, model dan evaluator CP2 belum dijalankan atau diverifikasi secara runtime. Semua hal bertanda **USULAN / PERLU REVIEW** harus disepakati sebelum freeze eksperimen.

## 1. Tujuan dan cakupan

Melatih dan membandingkan enam kondisi (B0, B1, B2, M1, M2, M3) dengan **input, label, pembagian data, loss, seed, dan kebijakan pemilihan checkpoint yang sama**. Setiap run memakai dua classification head:

- `hate`: 0 = `not hate`, 1 = `hate`.
- `appropriate`: 0 = `appropriate`, 1 = `inappropriate`.

Nama head `appropriate` tetap dipertahankan agar kompatibel dengan kode CP1; **kelas positif adalah inappropriate**, bukan appropriate. Dua output tidak boleh disatukan menjadi satu klasifikasi tiga kelas tanpa perubahan protokol formal. Hasil hate dan appropriateness dilaporkan terpisah.

## 2. Dataset, pembagian, dan provenance

| Bagian | Kontrak / keputusan |
| --- | --- |
| Dataset | `aiatums/INDOMEME` pada revisi `3e0cb8cae9cd726582f69afdb122163d0eab108b` |
| Jumlah resmi | train 4.018; test 1.005; total 5.023 meme |
| Partisi training CP1 yang dipertahankan | fit 3.214; validation 804; test 1.005 |
| Metode fit/validation | Manifest historis: `StratifiedGroupKFold`, 5 fold, fold pertama; seed split 42; gunakan manifest identik, **jangan melakukan split ulang** |
| SHA-256 manifest acuan | `4368e5767d73dfe988f9734dde59a146e125a0cda82c2783ee8eff46acab1a39` (cek byte dan newline sebelum memakai) |
| ID sampel | `image_path`; wajib unik pada unit evaluasi dan digunakan untuk pairing prediksi antarmodel |
| Pemakaian test | Hanya setelah pemilihan model dan kebijakan analisis dibekukan dari validation |
| Penyimpanan | Dataset gambar, prediksi individual, ID dan checkpoint di workspace privat, tidak di-commit ke GitHub |

**Audit wajib sebelum training (NLP02):** verifikasi ukuran tiap partisi, distribusi joint label, urutan ID, hash manifest, kelas positif, overlap fit–validation, missing values, duplikasi tepat dan kandidat kemiripan. Sumber CP1 mencatat dua grup gambar identik lintas split resmi; persoalan ini harus dilaporkan sebagai keterbatasan, bukan disembunyikan atau dihapus diam-diam dari test resmi. Jika dilakukan evaluasi terdeduplikasi, sajikan sebagai analisis sensitivitas tambahan dengan aturan yang ditentukan terlebih dahulu.

## 3. Aturan preprocessing teks

1. Fitur teks **hanya** `caption` dan `ocr`, dalam urutan **caption → separator tokenizer → OCR**. Tidak memakai `topic` (anotasi manusia), gold label, hasil prediksi model lain, maupun label ketidakpantasan sebagai input.
2. Gunakan tokenizer pinned `indolem/indobertweet-base-uncased`, revisi `32e28c05b47e33b6675d2670a1745c50a65e987a`; `max_length=256`, `padding='max_length'`, `truncation=True`. Pemisah berasal dari `tokenizer.sep_token` seperti helper CP1.
3. Bersihkan nilai kosong secara eksplisit sebelum menggabungkan field. `None`, `NaN` dan nilai kosong tidak boleh menjadi token literal `nan` atau `None`. Jika kedua field kosong, dokumentasikan kondisi, hasil tokenisasi dan kebijakan penerimaan/penolakan sampel; jangan diam-diam mengisi label atau teks buatan.
4. Separator hanya muncul **di antara dua field yang berisi**; jangan menambah pemisah ketika satu field kosong.
5. Audit tokenisasi tanpa truncation terlebih dahulu: hitung jumlah dan proporsi urutan dengan panjang >256, berapa token OCR yang terpotong dan berapa yang tersisa. Laporkan angka **hasil eksekusi**, bukan estimasi. Hitung pada fit/validation/test sebagai audit, tetapi jangan melakukan tuning berdasarkan label/prediksi test.
6. Pada pengujian *no-caption* atau *no-OCR*, bangun ulang teks dari field yang tersisa kemudian tokenisasi ulang dengan aturan yang sama. Jangan hanya men-zero token di tengah encoding tanpa kebijakan yang dinyatakan.

**Catatan reproduksi:** `text_values` CP1 memanggil `str(row[c])`, sehingga bila DataFrame baru memuat nilai null, dapat menimbulkan literal `nan`. Pemeriksaan input eksplisit diperlukan meskipun loader CSV CP1 menggunakan `keep_default_na=False`.

## 4. Kontrak batch NLP → CV

| Field | Tipe/shape | Kewajiban |
| --- | --- | --- |
| `input_ids` | `torch.int64`, `[B,L]` (L = 256) | Semua model |
| `attention_mask` | binary `[B,L]`, 0/1 | Semua model; posisi token valid = 1 |
| `content_mask` | `torch.bool`, `[B,L]` | Semua model; True hanya token isi nonspesial |
| `pixel_values` | `torch.float32`, `[B,3,224,224]` | B1/B2/M1/M2/M3; **tidak dibutuhkan B0** |
| `labels` | `torch.int64`, `[B,2]` | Kolom 0 hate, kolom 1 appropriate (1=inappropriate) |
| `image_path` | list ID panjang B | Metadata audit/evaluasi, tidak dimasukkan ke model |

`content_mask = attention_mask.bool() & ~isin(input_ids, tokenizer.all_special_ids)`.

Kontrak ini mengecualikan CLS/BOS, SEP/EOS, PAD dan separator literal yang dimasukkan di antara caption–OCR. `attention_mask` tetap dipakai encoder teks. **Jangan** mengirim `content_mask` sebagai `key_padding_mask` MHA: key/value pada cross-attention adalah 49 patch ViT, bukan token teks. Pada seluruh `content_mask=False`, pooled attention harus tepat nol dan finite. Pemrosesan gambar hanya sekali lewat processor ViT pinned; gambar RGB yang rusak ditolak eksplisit, tidak diganti gambar hitam.

**Contoh fixture buatan, bukan data riil:**

```python
labels = [[0, 0], [0, 1]]
# sample 0: not hate, appropriate
# sample 1: not hate, inappropriate
```

## 5. Model dan kesetaraan pembanding

| ID | Input/fusi | Bentuk output | Catatan |
| --- | --- | --- | --- |
| B0 | IndoBERTweet teks; `t → Dropout(0.3) → Linear(768,384) → Dropout(0.3)` | dua head `Linear(384,2)` | Tidak menginisialisasi ViT |
| B1 | `concat(t,v) → Linear(1536,768) → Dropout(0.3)` | dua head `Linear(768,2)` | Replikasi struktur fusi multitask E110, tetapi input CP2 baru |
| B2 | `u=Linear(768,768)(v)`; `g=sigmoid(Linear([t;u]))`; `h=t+g*u` | dua head `[B,2]` | Pooled gate, bukan cross-attention |
| M1 | `u=masked_mean(MHA(H,P,P))`; `h=t+u` | dua head `[B,2]` | Query teks, key/value 49 patch, 8 head |
| M2 | Sama dengan M1, ditambah `g`; `h=t+g*u` | dua head `[B,2]` | Uji M2 vs M1 dan M2 vs B2 |
| M3 | Sama dengan M2, `h=t+m*g*u`; saat training `m∼Bernoulli(0.9)` **per meme** | dua head `[B,2]` | Saat eval normal `m=1`; saat visual-off `m=0` |

Arsitektur CV tetap mengikuti CV01. Perbedaan parameter B0/B1 vs model lain dilaporkan; klaim bahwa B1→M1 hanya mengubah attention tidak valid, sebab bentuk fusi juga berubah. M2–M1 mengisolasi *tambahan gate* secara lebih dekat; M3–M2 mengisolasi strategi visual dropout dengan parameter identik.

## 6. Loss dan kebijakan pemilihan checkpoint (**USULAN CP2, PERLU PERSETUJUAN**)

```text
L_hate  = CE(logits['hate'], labels[:, 0])
L_app   = CE(logits['appropriate'], labels[:, 1])
L_total = 0.5 * L_hate + 0.5 * L_app
```

- Gunakan logits mentah pada CE, bukan probabilitas hasil softmax. Tidak menambah class weights saat protokol awal.
- Hitung `Macro-F1_hate_raw` dan `Macro-F1_appropriate_raw` pada validation; checkpoint per seed/konfigurasi dipilih menurut `0.5*(Macro-F1_hate_raw + Macro-F1_appropriate_raw)`. Jika nilai sama persis, pilih epoch yang lebih awal.
- **Ini berbeda dari E101/E110 CP1**, yang memakai constrained appropriateness pada selection. Karena itu, **B0 dan B1 harus dilatih ulang** memakai `Caption+OCR` dan aturan selection CP2 yang sama. Skor CP1 tidak menjadi kontrol langsung bagi B2–M3.
- Pilih arsitektur akhir dengan rerata validation pada seed yang sepadan. Ambang keputusan pragmatis selisih ≤0,005 untuk memilih metode lebih sederhana perlu disepakati sebelum training; tidak boleh diubah setelah melihat test.
- Simpan metrik **raw** dan **constrained** terpisah: `pred_inappropriate_constrained = max(pred_hate_raw, pred_inappropriate_raw)` untuk prediksi 0/1. Koreksi hanya tahap pelaporan sekunder; tidak diubah pada logits/training.
- Simpan hitungan prediksi yang melanggar relasi `hate => inappropriate`, baik sebelum maupun sesudah constraint.

## 7. Evaluasi dan artefak prediksi (NLP03/NLP05)

**Untuk setiap task dan seed:** confusion matrix `[0,1]` × `[0,1]`, Macro-F1, precision/recall/F1 per kelas, recall positif, support, accuracy sebagai metrik tambahan, validation loss, run ID, config, dan source commit. Jangan hanya melaporkan rerata dua task tanpa angka masing-masing.

Format privat prediksi minimum (satu baris per `image_path` dan task):

```text
run_id,seed,split,image_path,task,truth,prediction_raw,probability_positive,prediction_constrained
```

Kondisi wajib:

- Satu ID yang sama tidak boleh hilang, terduplikasi atau tidak terpasangkan antara dua task/run saat perbandingan.
- Urutan DataFrame harus di-*join by image_path*, bukan mengandalkan urutan baris.
- Probabilitas kelas positif dihitung sebagai `softmax(logits, dim=-1)[:,1]`, dicek finite dan berada pada `[0,1]`.
- `prediction_constrained` untuk task hate = raw; untuk appropriate = maksimum dengan prediksi hate yang dipasangkan.
- Rescue/hurt dihitung **terpisah per task dan dibandingkan pada ID yang sama**: rescue = baseline salah, kandidat benar; hurt = baseline benar, kandidat salah. Keduanya bukan rumus langsung perubahan Macro-F1.

**Self-check evaluator yang harus lulus:**

```text
truth      = [0, 0, 1, 1]
prediction = [0, 1, 1, 1]
confusion  = [[1, 1], [0, 2]]
accuracy   = 0.75
Macro-F1   = 11/15 ≈ 0.733333
```

Buat fixture kedua dengan dua task untuk membuktikan label map, relasi constrained dan pairing ID, serta fixture yang sengaja memiliki ID duplikat/hilang untuk memastikan evaluator menolak input rusak.

## 8. Robustness dan ablation (NLP04)

| Kondisi | Transformasi | Interpretasi yang benar |
| --- | --- | --- |
| Normal | Caption+OCR, ViT aktif untuk multimodal | Input utama |
| No-caption | OCR saja, retokenisasi | Menguji ketergantungan pada caption |
| No-OCR | Caption saja, retokenisasi | Masih mengandung informasi visual tersusun dalam caption |
| Visual-off | Menolkan kontribusi akhir ViT pada model multimodal; B1 menolkan `v` **sebelum concat** | Menghilangkan **jalur encoder visual langsung**, bukan semua informasi gambar |

Model tetap dilatih sesuai kondisi normal; no-caption/no-OCR/visual-off dipakai sebagai **uji evaluasi tanpa retraining** kecuali eksperimen lain dinyatakan. M3 menerima mask visual `[B,1]` per meme saat training, tidak memakai `nn.Dropout(0.1)` yang menskalakan aktivasi, dan normal eval harus deterministik. Tidak menilai visual-off sebagai pure text-only atau menyamakan logits visual-off dengan B0.

## 9. Urutan pengerjaan dan acceptance criteria

**NLP01 — dokumen ini.**

- [ ] Joseph dan Rafa mengonfirmasi dua head, label map, urutan kolom dan input tanpa topic.
- [ ] Memastikan `content_mask` mengecualikan token spesial sekaligus separator literal.
- [ ] Memastikan loss 0,5/0,5 dan **raw validation** sebagai selection disetujui.
- [ ] Menentukan cara menangani missing data, token truncation dan kebijakan evaluasi sampel ganda.
- [ ] Commit pada branch kerja NLP; ajukan PR dan catat review nyata sebelum menandai NLP01 selesai.

**NLP02 — data dan B0:** Implementasikan/ubah helper input secara terpisah dari source CP1 frozen; tes batch data dan tokenisasi, audit nilai kosong/split/truncation; latih B0 hanya setelah review dan smoke.  
**NLP03 — evaluator:** Implementasikan skor dua task raw+constrained, pairing ID, self-check confusion matrix dan source provenance.  
**NLP04 — robustness:** Jalankan empat kondisi pada checkpoint terpilih dengan ID/seed sama; simpan rescue/hurt dan kondisi input yang sebenarnya.  
**NLP05 — analisis:** Bandingkan B0/B1/B2/M1/M2/M3, waktu/parameter, per task, per seed, label constraint, dan batasan generalisasi; tidak mengklaim hasil yang belum diuji.

## 10. Prasyarat eksperimen bersama

- Konfigurasi utama direncanakan: AdamW, LR `2e-5`, weight decay `0.01`, FP32, batch fisik `1`, grad accumulation `8`, epoch maksimum `5`, patience `2`, linear scheduler tanpa warmup, gradient clipping `1.0`; **semua masih subject to pilot hardware**.
- Uji unit CV-T01–CV-T16 dan integrasi model/evaluator sebelum training penuh. Terutama cegah runner CP1 memanggil class `MemeClassifier` lama untuk B2–M3.
- Gunakan calibration GPU + pilot yang benar; model/seed hanya dijalankan berdasarkan anggaran aktual. Paket 6 run satu seed bersifat eksploratif, 14 run parsial antar-seed, 18 run penuh untuk enam konfigurasi × tiga seed.
- Log minimal: seed, run ID, source SHA, revisi dataset/model, manifest SHA, konfigurasi, versi environment, loss dan validation tiap epoch, parameter, durasi, puncak memori, ukuran checkpoint, resume/save/load, lokasi backup privat.

## 11. Keputusan terbuka untuk review (wajib dijawab sebelum freeze)

1. Apakah kedua head dan loss berbobot sama diterima untuk semua kondisi B0–M3?
2. Apakah selection **raw** validation menjadi kebijakan final CP2 (berbeda dari constrained CP1), dan constrained tetap sekunder?
3. Apakah kebijakan missing field mengikuti skip-field dan audit reject jika keduanya kosong, tanpa data imputation buatan?
4. Apakah CSV evaluasi disimpan privat; apakah ID lintas split akan diuji termasuk gambar identik dari split resmi?
5. Apakah M3 default visual dropout 0.1 ditetapkan tanpa tuning berdasarkan test?
6. Apakah benchmark hanya pada label INDOMEME dan tidak diklaim otomatis berlaku sebagai moderasi Facebook atau konten luar distribusi?

## 12. Hasil review / approval (diisi setelah ada bukti nyata)

- Link PR NLP: **belum ada**
- Commit NLP: **belum ada**
- Reviewer CV: **belum memberikan persetujuan**
- Status: **DRAF**
- Temuan dan revisi: **menunggu review**

**Sumber utama untuk pengecekan:**
- https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/cp2/cv-design/docs/cp2/cv-design.md
- https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/cp2/cv-design/src/master_training.py
- https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/cp2/cv-design/src/master_workflow.py
- https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/cp2/cv-design/results/runs/E101/config.json
- https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/cp2/cv-design/results/runs/E110/config.json
