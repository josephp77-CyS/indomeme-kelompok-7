# CP2 B1/B2 pooled vision — Implementasi awal, belum training

> **UPDATE STATUS — 16 Nov 2026:** Bagian di bawah merupakan *rencana awal saat kode belum dijalankan*, bukan status terkini. **Smoke B1 dan B2 telah berhasil**; **pilot 1 epoch seluruh data B1 dan B2 juga telah berhasil** di AI Data Center berdasarkan stdout operator. B1 pilot loss **0.5354**, raw validation Macro-F1 **0.7672**, durasi **1084.8 dtk**; B2 pilot loss **0.5344**, raw validation Macro-F1 **0.7542**, durasi **1044.7 dtk**. **Full training belum dilakukan**, hasil pilot bukan hasil utama. Lihat [handoff bukti terbaru](cp2_b1_b2_pilot_nlp04_handoff_2026-11-16.md); CV review dan audit resource dari JSON privat masih diperlukan.

**Tanggal:** 16 November 2026  
**Branch:** `cp2/nlp-implementation`, PR #4 (draft; **review CV Joseph diperlukan**).  
**Status:** Kode dan tes offline ditulis; **tes tersebut belum dijalankan di server**; belum ada smoke/pilot/full GPU B1/B2, belum ada Macro-F1 B1/B2.

## A. Hubungan B0 ke B1 dan B2

| Model | Backbone | Formula fusi (CV01) | Head/output |
| --- | --- | --- | --- |
| B0 | IndoBERTweet | `t -> Drop(0.3) -> Linear(768,384) -> Drop(0.3)` | Dua `Linear(384,2)` |
| B1 | IndoBERTweet + ViT | `[t;v] -> Linear(1536,768) -> Drop(0.3)` | Dua `Linear(768,2)` |
| B2 | IndoBERTweet + ViT | `u=Linear(768,768)(v)`; `g=sigmoid(Linear(1536,1)([t;u]))`; `h=t+g*u`; `Drop(0.3)(h)` | Dua `Linear(768,2)` |

- `t` dan `v` adalah `pooler_output` kedua encoder pretrained, dimensi 768.
- B1 memakai teks dulu lalu gambar pada concat; tanpa aktivasi/LayerNorm tambahan setelah projection. Pada `visual_off`, nolkan `v` sebelum projection, **bias B1 tetap aktif**.
- B2 memakai satu shared scalar gate per meme; tanpa task-specific gate. Pada `visual_off`, kontribusi `g*u` ditetapkan persis nol **setelah bias projection**.
- Backbone B1/B2 dimulai dari revisi pretrained pinned yang sama; **tidak** memuat bobot hasil fine-tune B0 sebagai initial checkpoint.
- Parameter **non-encoder** menurut CV01: B1 `1,183,492`, B2 `595,205`. Runner memeriksa secara runtime.
- Keduanya menggunakan **Caption+OCR tanpa Topic** dan citra asli train dengan image processor pinned `google/vit-base-patch32-224-in21k`; `[B,3,224,224]` FP32; split `fit=3214,validation=804` dari hash manifest B0. Gambar official test **tidak diunduh** dalam runner.
- Loss `0.5*CE_hate+0.5*CE_appropriate`, selection mean **raw** validation Macro-F1, constrained metric sekunder, `patience=2`, maks 5 epoch, AdamW LR `2e-5`, FP32, batch fisik 1, accumulation 8.
- Prediksi per-ID, CSV, `best.pt`, dan `last.pt` tetap privat. `result.json` tidak mengklaim test evaluation.

**Implementasi dan pemeriksaan yang ditambahkan:**

| Lokasi | Tugas |
| --- | --- |
| `src/cp2/mm_models.py` | B1 concat dan B2 pooled gate, visual_off, dua heads, pemeriksaan parameter |
| `src/cp2/mm_data.py` | Adaptor `MemeTextDataset` + pinned TRAIN parquet + RGB ViT processor |
| `src/cp2/mm_runner.py` | smoke/pilot/full/resume, evaluator NLP03, checkpoint atomik & provenance |
| `tests/test_cp2_mm.py` | offline mock backbones, bentuk tensor, visual_off, parameter, backprop, image decode, CLI guards |
| `scripts/cp2_mm_run.sh` | Perintah terminal yang bertahap dan mengharuskan konfirmasi full |

**Penting:** `tests/test_cp2_mm.py` **belum dieksekusi** oleh asisten pada environment proyek yang lengkap. Operator harus menguji dan melaporkan output. Artefak ini merupakan *candidate implementation* untuk review CV, **bukan** bukti bahwa eksperimen multimodal lulus.

## B. AI Data Center — jalur utama yang direkomendasikan

**Prasyarat:** mendapatkan alokasi GPU resmi, memastikan pengguna lain tidak terganggu, izin mengunduh dataset/model pinned; ruang storage cukup. Kode belum membutuhkan instalasi global; gunakan environment `(cp2-nlp-py312)` yang sudah ada dan jangan `sudo`.

Di **Terminal JupyterLab**:

```bash
cd "$HOME/Kelompok 7-INDOMEME/indomeme-kelompok-7"
git status --short
# Jika ada perubahan lokal, simpan/review dulu; jangan menimpa.
git pull --ff-only origin cp2/nlp-implementation
export PYTHONPATH="$PWD/src"
export PYTHONNOUSERSITE=1
bash scripts/cp2_mm_run.sh test
bash scripts/cp2_mm_run.sh preflight
```

Cek GPU/alokasi dan kebijakan admin sebelum tes CUDA berat. **Pertama jalankan hanya B1 smoke**:

```bash
bash scripts/cp2_mm_run.sh smoke-b1
```

Script akan **mengunduh pinned TRAIN Parquet berisi gambar**, sehingga unduhan awal bisa besar dan lambat. Ini berbeda dari B0 text-only. Image bytes tidak di-commit.

Jika B1 smoke **lulus**, dan reviewer CV mengonfirmasi bentuk fusi, jalankan bertahap:

```bash
bash scripts/cp2_mm_run.sh smoke-b2
bash scripts/cp2_mm_run.sh pilot-b1
bash scripts/cp2_mm_run.sh pilot-b2
```

**Tinjau**: puncak CUDA `peak_gpu_allocated_bytes` / `peak_gpu_reserved_bytes` pada `history.json`, waktu tiap epoch, ruang disk, dua Macro-F1 raw, parameter dan `validation_metrics.json` per-task. Pertahankan baseline B0 seed 42 sebagai pembanding; jangan menafsirkan smoke/pilot sebagai skor final.

Full training hanya **setelah approval CV/NLP, smoke/pilot berhasil, GPU dialokasikan, dan waktu/storage memadai**:

```bash
CP2_CONFIRM_MM_FULL=YES bash scripts/cp2_mm_run.sh full-b1
CP2_CONFIRM_MM_FULL=YES bash scripts/cp2_mm_run.sh full-b2
```

Dua perintah full **tidak harus dijalankan sekaligus**. Bila run full terputus setelah menyimpan `last.pt`, cek file dan lanjutkan pada **environment/config/source identik** menggunakan `resume-b1` atau `resume-b2` (dengan konfirmasi yang sama). Jangan memodifikasi source saat ada proses berlangsung.

Direktori output privat secara default:
- `~/kelompok-7-private/cp2_runs/B1_smoke_seed42/`
- `~/kelompok-7-private/cp2_runs/B2_smoke_seed42/`
- `~/kelompok-7-private/cp2_runs/B1_pilot_full_data_seed42/`, `B2_pilot_full_data_seed42/`
- `~/kelompok-7-private/cp2_runs/B1_full_seed42/`, `B2_full_seed42/`

**Jangan menjalankan ulang** aksi dengan nama sama jika `result.json` sudah ada. Untuk run seed berbeda gunakan `CP2_SEED=123`; output dibuat terpisah.

## C. Alternatif Colab gratis

Colab T4 **mungkin** cukup, tetapi B1/B2 memerlukan dua encoder dengan optimizer state. Kebutuhan VRAM untuk model utuh belum diukur; kalibrasi ViT+B0 terpisah **bukan** bukti. Colab gratis juga dapat memutus runtime dan paket default dapat konflik. Pilih **AI Data Center** sebagai tempat smoke/pilot pertama jika alokasi tersedia; Colab setelahnya untuk seed 123 pada workflow yang sama.

Jika menggunakan Colab, berikan perintah manual berikut **setelah menyalin kode terbaru** dan mengatur `CP2_WORKSPACE`, `CP2_OUT`, `PYTHONPATH` seperti notebook B0, lalu pastikan CUDA True, data berhasil diaudit dan Drive terpasang:

```python
!bash scripts/cp2_mm_run.sh test
!bash scripts/cp2_mm_run.sh preflight
!bash scripts/cp2_mm_run.sh smoke-b1
# Sesudah smoke B1 benar-benar berhasil dan review CV:
!bash scripts/cp2_mm_run.sh smoke-b2
```

Colab output `CP2_OUT` harus folder Drive privat yang persisten; jangan menulis model besar ke GitHub atau folder publik. Untuk sumber code baru, jalankan `git pull --ff-only` dalam direktori repository Colab (atau clone baru) dan `%cd` ke root repo; jangan membuat run pada source lama.

## D. Acceptance / bukti sebelum mengklaim "B1/B2 selesai"

- [ ] Review bersama CV terhadap arsitektur, label dan parameter.
- [ ] Tes offline lulus, `verify_submission.py` CP1 tetap lulus.
- [ ] Image loader hanya dari TRAIN parquet pinned, ID cocok manifest, tanpa corrupt bytes.
- [ ] Kedua encoder benar-benar pretrained dan CUDA; loss/backward finite, dua head `[B,2]`.
- [ ] B1/B2 smoke lulus, termasuk parameter, save/load/validation dan peak VRAM terukur.
- [ ] B1/B2 pilot 1 epoch full fit/validation lulus dan ruang/waktu cukup.
- [ ] Full seed 42 B1/B2 lulus, kemudian seed 123 bila sumber daya cukup.
- [ ] Per task Macro-F1 + raw/constrained metrics, confusion matrix, best epoch dilaporkan.
- [ ] Simpan `config/history/result/validation_metrics/parameter_summary` agregat; bobot/CSV prediksi privat.
- [ ] Test resmi belum digunakan untuk pemilihan desain.

## E. Integrasi dan kepemilikan

Rancangan CV01 berada di `cp2/cv-design` (PR #2). Implementasi B1/B2 ini ditambahkan sebagai **kandidat integrasi** di `cp2/nlp-implementation` (PR #4) karena permintaan melanjutkan branch yang sama, **tidak** secara otomatis menimpa branch CV atau dianggap mendapat persetujuan Joseph. Reviewer CV perlu memeriksa dua model dan memastikan apakah ingin memindahkan modul final ke jalur CV sendiri sebelum merge. B2 bukan M2 cross-attention; M1/M2/M3 belum dikerjakan.
