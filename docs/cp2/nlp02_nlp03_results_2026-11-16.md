# CP2 INDOMEME — hasil NLP02 (input/B0) dan NLP03 (evaluator)
 
**Tanggal pelaporan:** 16 November 2026  
**Branch:** `cp2/nlp-implementation` (PR #4, stacked di atas PR #3)  
**Dataset:** 3.214 fit, 804 validation, 1.005 official test (test **belum dievaluasi** dalam CP2)  
**Manifest SHA-256:** `4368e5767d73dfe988f9734dde59a146e125a0cda82c2783ee8eff46acab1a39`  
**Label:** hate (1=hateful); appropriate (1=inappropriate).  
**Input B0:** Caption+OCR, **tanpa topic**, panjang maksimal 256. Dua head di atas IndoBERTweet.

> **Asal bukti dan verifikasi:** Untuk AI Data Center A100 (seed 42), angka ditranskripsi dari output Terminal JupyterLab yang dikirim operator pada percakapan; file privat server belum diakses langsung. Untuk Colab T4 (seed 123), notebook `seed123.ipynb` dan file `config.json`, `history.json`, `result.json`, `validation_metrics.json` dibaca langsung melalui Google Drive yang dibagikan pengguna. Status ini **bukan** review independen atas bobot model atau hasil official test. Jangan mengunggah `.pt`, data mentah, atau CSV prediksi per-ID ke repo ini.

## A. Status kontribusi

| Paket | Status | Bukti | Batas penyelesaian |
| --- | --- | --- | --- |
| **NLP02 — Input & baseline B0** | **Implementasi dan pelaksanaan B0 selesai**, siap review | `src/cp2/nlp_data.py`, `nlp_models.py`, `nlp_runner.py`; audit input, smoke, pilot dan full seed 42/123 | Freeze protokol dan review tim tetap diperlukan; B0 tidak sama dengan integrasi B1–M3 |
| **NLP03 — evaluator CP2** | **Implementasi dan pengujian B0 selesai; handoff/integrasi multimodal belum selesai** | `src/cp2/nlp_evaluation.py`, pairing `image_path`, raw/constrained dua tugas, confusion matrix, 27 unit tests; file validation metrics privat untuk seed 123 | Baca metrik per tugas seed 42; pastikan runner B1–M3 menggunakan evaluator yang sama dan uji end-to-end setelah kode CV tersedia |
| **NLP01 — protokol** | Draft / review belum dinyatakan freeze | `configs/cp2_nlp_protocol.json`, PR #3 | Persetujuan bersama sebelum klaim eksperimen final |
| **NLP04 / NLP05** | Belum selesai | Robustness, rescue/hurt lintas model, error analysis, test final | Memerlukan model multimodal, ID-paired predictions privat dan rencana test terkunci |

## B. Pemeriksaan data dan komponen

- Unit test CP2: **27 passed** pada AI Data Center menurut operator; **27 passed** pada Colab menurut notebook yang dibaca. `verify_submission.py` juga memeriksa agregat CP1, **bukan** inferensi CP2; prediksi individual CP1 tidak diaudit dari checkout ini.
- `init-data` dan `audit`: **VALID SPLIT** (3.214/804/1.005) dan **AUDIT OK**, manifest sesuai di dua lingkungan. Statistik detail audit token perlu tetap disimpan sebagai JSON privat.
- Kalibrasi satu optimizer step, **component-only** di A100, batch 1, fp32, hasil operator:
  
  | Komponen | Waktu step (detik) | Peak allocated (byte) | Jumlah parameter |
  | --- | ---: | ---: | ---: |
  | Cross-attention sintetis | 0.756965 | 67,471,872 | 2,362,368 |
  | B0 IndoBERTweet | 1.190418 | 2,247,137,280 | 110,855,044 |
  | ViT | 1.135192 | 1,780,019,712 | 88,045,824 |

  Kalibrasi ini **bukan** profiling B1/M3 lengkap, bukan akurasi validasi dan tidak dapat dijumlahkan untuk mengklaim VRAM multimodal.

## C. Smoke dan pilot (bukan skor final)

| Run | GPU | Train fit / val | Epoch | Loss | Mean raw val Macro-F1 | Waktu epoch (detik) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| B0 smoke seed 42 | A100 | 32 / 16 | 1 | 0.7116 | 0.1980 | 7.1 |
| B0 pilot full-data seed 42 | A100 | 3,214 / 804 | 1 | 0.5443 | 0.7491 | 575.4 |
| B0 smoke seed 123 | Tesla T4 | 32 / 16 | 1 | 0.7075 | 0.3333 | 2.8 |
| B0 pilot full-data seed 123 | Tesla T4 | 3,214 / 804 | 1 | 0.5515 | 0.7507 | 209.5 |

Smoke adalah cek pipeline, bukan hasil evaluasi riset.

## D. B0 full — dua seed selesai

**Aturan:** maksimum 5 epoch, `patience=2`, AdamW LR `2e-5`, weight decay `0.01`, fp32, batch fisik 1, accumulation 8, loss 0.5 hate + 0.5 appropriate. Pemilihan checkpoint dari `mean(raw validation Macro-F1 hate, appropriate)`; constrained appropriateness sekunder saja; tidak menggunakan official test.

| Seed | Tempat | E1 | E2 | E3 | E4 | Epoch terbaik | Best selection Macro-F1 | Berhenti |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **42** | AI Data Center / A100 | 0.7669 | **0.7821** | 0.7746 | 0.7806 | **2** | **0.7821** | Epoch 4, early stopping |
| **123** | Colab / Tesla T4 | **0.7712136** | 0.7512181 | 0.7599154 | — | **1** | **0.7712135742** | Epoch 3, early stopping |

Rata-rata best validation selection dua seed secara deskriptif = **0.7766568**; berbeda seed **dan** environment, tidak menyiratkan signifikan secara statistik atau perbandingan arsitektur yang terkendali. Kedua run full dilaporkan selesai. `test_evaluated=false` diverifikasi langsung pada `result.json` seed 123; untuk seed 42 official test tidak terlihat dijalankan dalam log operator.

### Detail evaluasi **seed 123** — dibaca dari Drive

File `validation_metrics.json` mencerminkan **best checkpoint epoch 1**, bukan metrik epoch terakhir.

| Metrik validation (804 meme) | Hate | Appropriate raw | Appropriate constrained |
| --- | ---: | ---: | ---: |
| Macro-F1 | **0.7941173** | **0.7483099** | **0.7483099** |
| Accuracy | 0.8171642 | 0.7537313 | 0.7537313 |
| Confusion matrix `[[TN,FP],[FN,TP]]` | `[[463,68],[79,194]]` | `[[362,92],[106,244]]` | `[[362,92],[106,244]]` |

- Mean raw selection: `(0.7941172884 + 0.7483098600) / 2 = 0.7712135742`.
- Pelanggaran aturan raw `hate=>inappropriate` pada best checkpoint: **0**, maka skor raw dan constrained appropriate sama pada seed 123 best epoch.
- Full seed 123: total 687.34 detik menurut `result.json`, 3 epoch; model/provenance dari `config.json`. GPU Tesla T4, Python 3.13.16, torch 2.14.1+cu130, Transformers 4.46.3; Git commit eksekusi `f4563b8b14b6c6f58be0722db6f77aa4d34e1d0f`.
- Log TensorFlow di notebook melaporkan `Could not find cuda drivers`, namun PyTorch pada notebook melaporkan `torch.cuda.is_available() == True`, GPU Tesla T4; runner memakai `--require-gpu` dan berakhir berhasil. Log TensorFlow tidak cukup untuk menyimpulkan training PyTorch memakai CPU.
- Output `pip check` pada notebook menunjukkan **konflik dependency** dengan komponen bawaan Colab (antara lain google-colab/ipykernel/pandas; numba/numpy; gradio/diffusers/huggingface-hub). Walau training B0 berhasil, konflik ini perlu didokumentasikan dan diselesaikan untuk lingkungan yang lebih bersih/reproduksibel.

**Seed 42**: per-task `validation_metrics.json`, confusion matrix, `result.json`, dan provenance masih **menunggu ekspor operator** dari folder privat AI Data Center. Tidak mengarang nilai per-task seed 42 dari skor agregat. Lingkungan kalibrasi A100 yang dilaporkan: Python 3.12.16, torch 2.6.0+cu124, Transformers 4.46.3; Git SHA operator pada saat kalibrasi `e5e1d2538a0a080cb7069bd8affc4e38d4ebcc90`.

## E. Akses data dan lokasi artefak

- Notebook Colab yang dibaca: [`seed123.ipynb`](https://colab.research.google.com/drive/1Hk-4T9uaWmkHg_C_PMXW6U1zu9EXzAYS).
- Hasil privat Colab full: [folder Drive B0 full seed 123](https://drive.google.com/drive/folders/1yDBydyQexr9p2_SLhi4YamjujKrbaygN). Tidak memindahkan, memperluas izin berbagi, atau mengunggah prediksi/checkpoint dari folder tersebut ke repo.
- AI Data Center privat: `~/kelompok-7-private/cp2_runs/B0_full_seed42/` (tidak tersedia langsung bagi asisten).
- Berkas ringkasan yang aman untuk ditinjau: `config.json`, `history.json`, `result.json`, `validation_metrics.json`. `best.pt`, `last.pt`, `validation_predictions_PRIVATE.csv` tetap di storage privat.

## F. Langkah berikutnya

1. Operator mengekspor **empat JSON seed 42** dari AI Data Center (tidak perlu memindahkan bobot `.pt` ke GitHub).
2. Lengkapi tabel metrik per tugas untuk seed 42, cocokkan checksum manifest/provenance, ulangi smoke unit test setelah perubahan source dan simpan catatan reproducibility dua lingkungan.
3. Reviewer NLP/CV menyetujui kontrak evaluator dua tugas (`raw` untuk seleksi, `constrained` sekunder), dan memastikan runner B1/B2/M1/M2/M3 benar-benar memanggil evaluator bersama serta menghormati `image_path`, `content_mask`, split yang sama.
4. Partner CV mengimplementasikan dan menguji B1/B2/M1/M2/M3; profiling end-to-end B1/M3 harus menggantikan sekadar kalibrasi komponen. Mulai dari B1 sebagai kontrol multimodal; gunakan seed 42 dan 123 secara konsisten.
5. Setelah validation/protokol terkunci, lanjut evaluasi robustness NLP04, rescue/hurt NLP05, kemudian official test final **sekali sesuai protokol**, bukan untuk pemilihan varian.

**Interpretasi CP1 vs CP2:** skor CP1 dengan `Caption+OCR+Topic` dan/atau selection constrained serta beberapa skor official test **tidak sebanding langsung** dengan B0 CP2 (`Caption+OCR`, mean raw validation Macro-F1). Jangan menyimpulkan B0 wajib mengalahkan CP1; B0 adalah baseline untuk perbandingan B1–M3 dalam protokol CP2 yang sama.
