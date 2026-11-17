# Paket kontribusi CP2 NLP — INDOMEME Kelompok 7

**Status integrasi terbaru ada di [mulai-cp2.md](docs/cp2/mulai-cp2.md).** Dokumen di bawah merekam pekerjaan NLP yang sudah diserahkan hingga 16 November, termasuk status sementara sebelum audit dan sebelum integrasi M1–M3. Gunakan panduan baru untuk menjalankan eksperimen dari branch integrasi.

> **STATUS TERBARU — 16 November 2026 (berdasarkan bukti operator dan notebook yang dijalankan):** **B0 full training dua seed (42/123) telah selesai**, dan NLP04 validation-only pada B0 dua seed juga telah selesai. **B1 dan B2 telah lulus smoke dan pilot seluruh data satu epoch seed 42** di AI Data Center (B1 pilot train loss **0.5354**, mean raw validation Macro-F1 **0.7672**, **1084.8 dtk**; B2 pilot train loss **0.5344**, mean raw validation Macro-F1 **0.7542**, **1044.7 dtk**). **Belum full training B1/B2**; M1–M3 belum memiliki implementasi resmi end-to-end. [Laporan bukti dan handoff untuk CV/Joseph](docs/cp2/cp2_b1_b2_pilot_nlp04_handoff_2026-11-16.md). [Laporan NLP04 dua seed](docs/cp2/nlp04_b0_two_seed_robustness_2026-11-16.md). Paragraf status historis setelah ini adalah snapshot saat penulisan awal, **bukan status terkini**. Angka pilot tidak dipakai sebagai skor final; metrik/perangkat/VRAM rinci menunggu audit JSON privat.


> **Integrasi lintas model (pembaruan berikutnya, 16 November):** Kontribusi NLP **juga berlaku pada B1, B2, M1, M2, M3**, bukan sekadar B0. Kode `src/cp2/nlp_cv_bridge.py` menyediakan kontrak batch/dua logits/evaluator bersama untuk keenam model; `src/cp2/nlp_cv_eval.py` menyediakan robustness validation-only; `mm_runner.py` B1/B2 sudah tersambung ke bridge. Lihat [audit, ownership, instruksi dan status pengujian](docs/cp2/nlp_cv_integration_2026-11-16.md). **M1–M3 masih memerlukan factory & full model dari CV.** NLP04 B0 belum dijalankan di server; jangan menyebutnya selesai.

> **Update 16 November 2026:** Status awal README di bawah adalah catatan implementasi awal NLP. **B0 full seed 42 dan 123 sudah selesai**; laporan agregat + metrik dua tugas ada di [NLP02/NLP03 results](docs/cp2/nlp02_nlp03_results_2026-11-16.md). **Kandidat implementasi B1 concatenation dan B2 pooled gate** kini ada di `src/cp2/mm_models.py`, `src/cp2/mm_data.py`, `src/cp2/mm_runner.py` dengan skrip `scripts/cp2_mm_run.sh`, tes `tests/test_cp2_mm.py` dan [panduan B1/B2](docs/cp2/b1_b2_implementation_2026-11-16.md). **Tes dan training B1/B2 belum dijalankan dan memerlukan review CV**, jangan melaporkan skor B1/B2 sebelum hasil aktual tersedia. M1–M3 belum diintegrasikan.

**Status:** implementasi NLP untuk review. Source dan tes disiapkan lokal; kalibrasi/training pada dataset asli dan server AI Data Center belum dibuktikan. Branch GitHub dan bukti eksekusi perlu diperiksa sebelum dianggap selesai.

## 0. Mengapa paket ini diperlukan?

Audit branch publik per 15 November 2026:

- `main`: source CP1 di `src/master_training.py` dan `src/master_workflow.py`, hasil CP1.
- `cp2/nlp-protocol`: PR #3 terbuka; dua dokumen NLP. Tidak ada perubahan implementasi Python.
- `cp2/cv-design`: PR #2 terbuka; rancangan B0/B1/B2/M1/M2/M3 dan laporan CV. Tidak ada model CP2 baru di `src`.
- CV01 telah menetapkan M1 = cross-attention + **residual**, M2 = M1 + **shared scalar gate**, M3 = M2 + visual dropout 0.1. **Belum ada** arm cross-attention + concat atau **task-specific gate** seperti dalam brainstorming. Perlu PR perubahan protokol dan persetujuan sebelum diperlakukan sebagai rancangan resmi.

File baru di paket ini **tidak mengubah** `src/master_training.py`, `src/master_workflow.py`, `verify_submission.py`, hasil CP1, atau README CP1.

## 1. Isi paket dan kontribusi

| File | Fungsi | Status |
|---|---|---|
| `src/cp2/nlp_data.py` | input Caption+OCR, null sanitization, `content_mask`, label, split/hash, audit token | tes sintetis |
| `src/cp2/nlp_evaluation.py` | raw/constrained dua task, Macro-F1, confusion matrix, pairing ID, rescue/hurt | tes sintetis |
| `src/cp2/nlp_models.py` | B0 IndoBERTweet dua head sesuai CV01 | belum diunduh/ditrain |
| `src/cp2/nlp_runner.py` | CLI init-data, audit, pilot B0, full B0, profiling B0/ViT/MHA | antarmuka teruji tanpa data; GPU belum diuji |
| `src/cp2/fusion_reference.py` | reference **tensor sintetis** cross-attention + concat/residual, shared/task gates, visual dropout | bukan pipeline CV resmi; tes sintetis |
| `tests/test_cp2_nlp.py` | regression checks label/mask/evaluator/gate | lokal lulus |
| `configs/cp2_nlp_protocol.json` | config rancangan sebelum freeze | perlu approval |
| `docs/cp2/handoff_nlp_cv.md` | handoff NLP/CV, alignment arsitektur dan acceptance | draft |

Data privat, prediksi per-ID, cache Hugging Face, checkpoint, dan token **tidak** termasuk paket.

## 2. Siapkan branch kerja dan tempat file

Di laptop/server tempat repo boleh diakses:

```bash
git clone https://github.com/josephp77-CyS/indomeme-kelompok-7.git
cd indomeme-kelompok-7
git fetch origin
# Buat implementation branch dari protokol NLP yang sudah ada.
git switch -c cp2/nlp-implementation origin/cp2/nlp-protocol
```

Salin **isi** paket ini ke root repository (struktur `src/cp2`, `tests`, `configs`, `docs/cp2`, dan berkas requirements khusus), tidak sebagai folder `cp2_nlp_contribution/` yang membungkus semuanya. Jangan menimpa dokumentasi milik partner.

Untuk melakukan review lintas branch tanpa merge sumber eksperimen yang belum disetujui:

```bash
git diff origin/main...origin/cp2/nlp-protocol
git diff origin/main...origin/cp2/cv-design
git status --short
```

**Penting:** `cp2/nlp-protocol` sudah menjadi head PR #3 yang masih terbuka. Bila ingin PR implementasi terpisah, buat PR baru dari branch `cp2/nlp-implementation` dengan **base `cp2/nlp-protocol`** terlebih dahulu (stacked PR), atau tunggu PR #3 merged lalu retarget PR implementasi ke `main`. Jangan membuat PR implementasi langsung ke `main` tanpa menyadari diff dokumentasi akan ikut masuk.

## 3. AI Data Center: environment aman dan verifikasi GPU

Gunakan folder privat/tim yang diizinkan di server; contoh `~/kelompok-7-private` hanya jika kebijakan server mengizinkan. **Jangan** memakai `sudo`, mengubah versi CUDA global, menginstal agent, atau mengunggah secrets ke GitHub.

```bash
python3 --version
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu --format=csv,noheader
free -h
df -h "$HOME"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Di bawah ini hanya CONTOH bila driver mendukung CUDA 12.4; sesuaikan dengan driver server.
# Gunakan wheel sesuai panduan resmi PyTorch/admin AI Data Center.
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -r requirements-cp2-nlp.txt
python -c 'import torch,transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
```

> Jangan memaksakan URL cu124 jika driver/server memakai versi lain atau memakai environment PyTorch yang sudah disediakan. CP1 mencatat Torch 2.6.0/cu124, Transformers 4.46.3; reproduksi pada environment lain harus dicatat. `requirements.txt` di repo asal hanya mencakup analisis dan tidak memasang PyTorch/Transformers.

Pastikan `PYTHONPATH` memasukkan `src` agar `master_workflow` dan package CP2 dapat diimport:

```bash
export PYTHONPATH="$PWD/src"
python -m cp2.nlp_runner --help
python -m pytest -q tests/test_cp2_nlp.py
python verify_submission.py
```

Unit test tidak mengunduh model atau dataset. `verify_submission.py` memeriksa bukti historis CP1 dan **bukan** bukti bahwa CP2 sudah training.

## 4. Inisialisasi dataset dan manifest (sekali per workspace)

Sediakan direktori privat, misalnya:

```bash
mkdir -p "$HOME/kelompok-7-private/cp2_data" "$HOME/kelompok-7-private/cp2_runs"
export CP2_WORKSPACE="$HOME/kelompok-7-private/cp2_data"
export CP2_OUT="$HOME/kelompok-7-private/cp2_runs"

# Gunakan kode CP1 untuk mengambil suplement CSV pinned dan membuat split historis.
python -m cp2.nlp_runner init-data --workspace "$CP2_WORKSPACE"
# Verifikasi metadata, split SHA, distribusi label, token truncation dan audit field.
python -m cp2.nlp_runner audit --workspace "$CP2_WORKSPACE" --output "$CP2_OUT/audits"
```

Script `init-data` bergantung pada akses jaringan ke sumber data yang sudah ditetapkan di `master_workflow.py`. Jika data sudah disalin dari workspace CP1, boleh melewati `init-data` **asalkan** ada:

```
$CP2_WORKSPACE/
  data/processed/metadata.csv
  splits/train_validation.csv
```

`audit` **menolak** manifest dengan hash selain `4368e5767d73dfe988f9734dde59a146e125a0cda82c2783ee8eff46acab1a39`. Jumlah fit/validation/test wajib 3214/804/1005. **Tidak ada split ulang menggunakan seed baru.** Jika gagal, hentikan pekerjaan dan bandingkan checksum dengan CP1; jangan mengganti konstanta hash sekadar agar lulus.

Buka `$CP2_OUT/audits/token_audit.json` dan `$CP2_OUT/audits/nlp_input_audit.json`. Audit mengevaluasi *panjang input* test saja, **bukan** metrik label test.

## 5. Arti konfigurasi dan kalibrasi

**Konfigurasi** = menyepakati parameter metodologi sebelum run: revisi dataset/model/tokenizer, Caption+OCR, tanpa topic, label positif, 256 token, 49 patch ViT, dua head, loss 0.5+0.5, selection `mean(raw validation Macro-F1 dua task)`, batch1/accum8, seed, protokol visual-off, lokasi checkpoint privat.

**Kalibrasi komputasi** = memastikan hardware benar-benar mampu menjalankan model dan menghitung biaya step/epoch, *peak VRAM*, RAM, disk, ukuran checkpoint, dan resume. Ini **berbeda** dengan kalibrasi probabilitas seperti temperature scaling; jangan mengubah ambang keputusan atau memilih checkpoint berdasarkan test.

Uji komponen satu optimizer step (hanya baca komponen; bukan model gabungan B1/M3):

```bash
python -m cp2.nlp_runner calibrate --component attention --batch 1 --output "$CP2_OUT/calibration" --require-gpu
python -m cp2.nlp_runner calibrate --component b0        --batch 1 --output "$CP2_OUT/calibration" --require-gpu
python -m cp2.nlp_runner calibrate --component vision    --batch 1 --output "$CP2_OUT/calibration" --require-gpu
```

`attention` memakai representasi tensor buatan `[B,256,768]` dan 49 patch; ia menguji optimizer MHA saja. `b0` memuat IndoBERTweet asli dan dua head. `vision` memuat ViT asli tetapi tanpa IndoBERTweet/cross-attention. Karena memori dapat naik saat model digabungkan, **angka tiga komponen tidak boleh dijumlahkan lalu dinyatakan sebagai hasil kalibrasi M3**. Setelah partner CV mengimplementasikan model multimodal penuh, ulangi satu training step terintegrasi B1 dan M3 dengan config identik untuk tahap C3.

Jika `CUDA out of memory`, cek GPU lain bila resmi tersedia, pengaturan batch, checkpointing atau strategi precision yang **disepakati bersama**, bukan diam-diam mengurangi token, patch, model, atau mengubah distribusi input. Simpan log gagal.

**Minimum handoff kalibrasi:** nama GPU, tanggal, versi torch/transformers/CUDA, VRAM free awal, max allocated/reserved, waktu step, banyak parameter, disk free, versi source/commit, sumber HF pinned, dan klasifikasi hasil `component-only` atau `end-to-end`.

## 6. Jalankan baseline B0 terlebih dahulu

### A. Smoke pilot 32 fit/16 validation

```bash
python -m cp2.nlp_runner train-b0 --workspace "$CP2_WORKSPACE" \
  --output "$CP2_OUT" --seed 42 --epochs 1 --max-train 32 --max-validation 16 --require-gpu
```

Tujuan: cek forward/backward, labels, validasi raw/constrained, output CSV privat, best/last checkpoints. **Jangan masukkan skor pilot kecil ke tabel hasil CP2.**

### B. Pilot hardware 1 epoch menggunakan seluruh fit/validation

```bash
python -m cp2.nlp_runner train-b0 --workspace "$CP2_WORKSPACE" \
  --output "$CP2_OUT" --seed 42 --pilot-full-data --epochs 1 --require-gpu
```

Ini mengikuti konsep C4 di dokumen CV: data fit/validation penuh untuk mengukur durasi. Checkpoint pilot bukan model final. Contoh pilot B1/M3 baru dapat dilakukan setelah implementasi milik CV selesai.

### C. Training B0 penuh setelah freeze protokol

```bash
python -m cp2.nlp_runner train-b0 --workspace "$CP2_WORKSPACE" \
  --output "$CP2_OUT" --seed 42 --full --epochs 5 --require-gpu
```

Run di atas **melatih B0 saja**. Fitur: AdamW, FP32, grad accum8, max epoch5, patience2, linear scheduler, gradient clipping, pemilihan best epoch dari dua raw validation Macro-F1. Hasil test resmi **tidak** dijalankan. Epoch yang selesai disimpan dalam `last.pt`, terbaik sebagai `best.pt`; lanjutkan dengan perintah sama plus `--resume` jika run terputus pada batas epoch.

Setelah validasi 42 berjalan, jika anggaran memungkinkan dan protokol sudah tetap, buat B0 seed 123 dan 2026 dengan mengganti `--seed` (run ID berbeda). Jangan menyalin `best.pt` B0 sebagai checkpoint final untuk varian CV tanpa penjelasan desain.

### Menggunakan `tmux`

```bash
tmux new -s cp2-nlp
# di dalam tmux jalankan satu perintah training saja;
# Ctrl+B lalu D untuk detach
# kembali: tmux attach -t cp2-nlp
```

Hindari dua worker yang berebut satu GPU. Perhatikan kebijakan server untuk jam komputasi, kuota dan proses latar.

## 7. Berkas output dan kebijakan Git

Contoh output privat:

```
cp2_runs/
  audits/nlp_input_audit.json
  audits/token_audit.json
  calibration/calibration_b0_batch1.json
  B0_smoke_seed42/config.json
  B0_smoke_seed42/history.json
  B0_smoke_seed42/validation_metrics.json
  B0_smoke_seed42/validation_predictions_PRIVATE.csv
  B0_smoke_seed42/best.pt
  B0_smoke_seed42/last.pt
  B0_smoke_seed42/result.json
```

**Yang boleh di-commit:** source `.py`, tests, JSON protokol tanpa rahasia, README/handoff, dan ringkasan agregat metrik setelah diverifikasi tanpa sample ID. **Jangan commit:** `metadata.csv`, `image_path` individual, prediksi per-meme, bobot `.pt`, `.bin`, `.safetensors`, dataset zip/parquet/gambar, HF cache, akses SSH, API tokens, log host sensitif.

Sebelum `git add`, gunakan `git status --short` dan pengecekan ukuran file; gunakan `git add` path yang eksplisit, bukan `git add .` jika repo berada di server bersama.

```bash
git add src/cp2/ tests/test_cp2_nlp.py configs/cp2_nlp_protocol.json \
  docs/cp2/handoff_nlp_cv.md README_CP2_NLP.md requirements-cp2-nlp.txt
git diff --cached --stat
git commit -m "feat(cp2-nlp): add input protocol, raw evaluator, B0 runner and component calibration"
git push -u origin cp2/nlp-implementation
```

Buat pull request di GitHub dengan checklist acceptance, tautkan PR #3, minta review partner CV, dan **jangan merge sepihak** sebelum freeze keputusan bersama.

## 8. Urutan kerja NLP vs CV

1. **NLP01:** finalisasi protokol label/input/evaluasi; koreksi dokumen NLP yang masih bertuliskan "PR belum ada" padahal PR #3 sudah terbuka.
2. **NLP02:** implementasi `nlp_data.py` dan B0; audit/cek manifest dan 256 token; jalankan smoke B0.
3. **NLP03:** evaluator raw/constrained, pairing ID, unit test, ringkasan per task; pastikan runner multimodal nanti memanggil evaluator yang sama.
4. **Joint CV/NLP:** partner CV membuat B1/B2/M1/M2/M3; NLP memasok batch, dua label, metrics, dan menjalankan integrasi C2/C3 bersama. `fusion_reference.py` hanyalah demonstrasi aljabar tensor usulan tambahan.
5. **NLP04:** no-caption/no-OCR retokenisasi dan visual-off pada checkpoint terbaik. Metode `visual_off` harus disepakati pada API model CV; visual-off **bukan** pure text karena caption dapat mengandung informasi visual.
6. **NLP05:** rescue/hurt per task dan per ID, per-seed mean dan dispersi, macro-F1 raw, constrained sekunder, durasi, parameter, ablation/robustness. Test resmi digunakan setelah selection/protokol terkunci.

## 9. Batasan dan masalah yang belum selesai

- `PADASOURCE` atau percakapan ChatGPT sebelumnya tidak tersedia dalam paket lokal; tidak ada klaim bahwa telaah literatur menyeluruh sudah selesai.
- Pemisahan eksak token OCR yang terpotong menggunakan offset dari *fast tokenizer*, bila tersedia. `ocr_tokens_trimmed_offset_based=null` bila tokenizer tidak mendukung offset; **jangan** melaporkan nol sebagai fakta dalam kondisi itu.
- CPU unit tests dan reference fusion sintetik tidak memastikan bobot IndoBERTweet/ViT terunduh, integrasi trainer CV, atau ketahanan GPU.
- Pinned HF revisions wajib tersedia/diizinkan di data center dan perlu akses internet atau cache internal.
- Baseline CP1 E101/E110 memakai `Caption+OCR+Topic` dan checkpoint selection yang berbeda. Bandingkan dengan **baseline B0/B1 dilatih ulang dalam CP2**, bukan memakai skor lama sebagai kontrol langsung.
- Pernyataan novelty harus dibatasi pada hipotesis kontribusi: pengujian adaptasi penggunaan visual yang berbeda pada dua task pada dataset bahasa Indonesia, **bukan** penemuan pertama cross-attention/gate.

## 10. Sumber GitHub acuan

- [Repo main](https://github.com/josephp77-CyS/indomeme-kelompok-7)
- [PR NLP #3](https://github.com/josephp77-CyS/indomeme-kelompok-7/pull/3)
- [PR CV #2](https://github.com/josephp77-CyS/indomeme-kelompok-7/pull/2)
- [CV design](https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/cp2/cv-design/docs/cp2/cv-design.md)
- [NLP protocol](https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/cp2/nlp-protocol/docs/cp2/nlp-protocol.md)


## 11. Jalur pengawasan JupyterLab (baru)

Gunakan `notebooks/00_cp2_nlp_operator.ipynb` untuk mengawasi preflight, tes, audit, dan kalibrasi. Setiap sel berat memakai flag `False` secara default, sehingga membuka notebook tidak otomatis melatih model. Alternatif terminal, dari **root repository** gunakan:

```bash
bash scripts/cp2_ai_center_run.sh preflight
bash scripts/cp2_ai_center_run.sh test
bash scripts/cp2_ai_center_run.sh init-data
bash scripts/cp2_ai_center_run.sh audit
bash scripts/cp2_ai_center_run.sh calibrate-attention
bash scripts/cp2_ai_center_run.sh calibrate-b0
bash scripts/cp2_ai_center_run.sh calibrate-vision
bash scripts/cp2_ai_center_run.sh smoke-b0
bash scripts/cp2_ai_center_run.sh pilot-b0
# Setelah review dan approval full run:
export CP2_CONFIRM_FULL=YES
bash scripts/cp2_ai_center_run.sh full-b0
```

Buat virtual environment dan pasang paket CUDA kompatibel **sebelum** langkah ini. Jika repo berbeda dari main/branch yang diuji, simpan bukti `git rev-parse HEAD`. Kode tidak menyentuh official test. Ini bukan permintaan agar proses dijalankan tanpa pengawasan.
