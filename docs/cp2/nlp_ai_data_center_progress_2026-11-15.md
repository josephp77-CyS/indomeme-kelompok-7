# CP2 NLP — log progres eksekusi AI Data Center

> **PEMBARUAN 16 November 2026:** Status historis di bawah merekam keadaan **15 November**, bukan status terbaru. Pada 16 November operator telah membagikan bukti bahwa kalibrasi attention/B0/ViT, B0 smoke, pilot dan full training seed 42 **selesai** di A100. Best raw validation mean Macro-F1 seed 42 = **0.7821** (epoch 2 dari 4; early stopping). Colab Tesla T4 juga menyelesaikan B0 seed 123 dengan best **0.7712135742** (epoch 1 dari 3; early stopping), yang dikonfirmasi dari file JSON di Drive. Detail lengkap, batas verifikasi, dan langkah NLP02/NLP03 ada di [laporan hasil 16 November](nlp02_nlp03_results_2026-11-16.md). **Jangan mengutip daftar "belum dijalankan" di bawah sebagai progres terbaru.**

**Tanggal laporan:** 15 November 2026  
**Cabang:** `cp2/nlp-implementation`  
**Commit yang dijalankan operator:** `e5e1d2538a0a080cb7069bd8affc4e38d4ebcc90`  
**Status:** environment, unit test, inisialisasi dataset, dan audit input **lulus**; kalibrasi GPU dan training CP2 **belum dijalankan**.

> **Asal bukti dan batas verifikasi.** Angka dan status pada dokumen ini ditranskripsikan dari output Terminal JupyterLab yang **dibagikan operator di percakapan**. Asisten dapat memverifikasi keberadaan source/branch melalui GitHub, tetapi **tidak mengakses langsung terminal AI Data Center**, file audit privat, atau log proses GPU. Karena itu dokumen ini adalah *operator-reported execution log*, bukan audit independen maupun hasil eksperimen ilmiah terverifikasi pihak ketiga.

## 1. Identitas source dan environment

| Komponen | Bukti yang tersedia |
| --- | --- |
| Git branch | `cp2/nlp-implementation` |
| Git commit | `e5e1d25` (output `git log -1 --oneline`) |
| Python | Environment Conda `cp2-nlp-py312` (Python 3.12) |
| PyTorch | `torch-2.6.0+cu124` berhasil dipasang menurut output pip |
| Transformers | `4.46.3`, sesuai requirement CP2 (terlihat pada output dependency) |
| Pillow | `python -I` berhasil mengimpor `PIL._imaging` dari environment Conda |
| GPU node | NVIDIA A100-SXM4-80GB terdeteksi oleh `nvidia-smi` |

**Catatan:** Keberhasilan impor dan semua unit test tidak membuktikan forward/backward model IndoBERTweet pretrained dengan CUDA, kecepatan training, maupun kompatibilitas kombinasi model penuh. Perintah `pip check` terakhir yang bersih dan detail `torch.cuda.is_available()` **setelah** perpindahan ke environment Python 3.12 tidak tersedia dalam transkrip bukti; jangan diasumsikan.

## 2. Pengujian source (selesai)

Perintah:

```bash
bash scripts/cp2_ai_center_run.sh test
```

Output operator:

```text
...........................                                             [100%]
27 passed in 7.96s
PASS: 8 configs, 10 outcomes, 0 prediction audits, 52 executed notebook cells, 25 PPTX slides.
Prediction CSVs omitted in GitHub checkout; aggregate confusion matrices were checked. Full audit ZIP is on Drive.
```

**Interpretasi:** 27 unit test CP2 lulus di server; `verify_submission.py` berhasil memeriksa artefak agregat CP1 yang tersedia di checkout. `0 prediction audits` berarti **CSV prediksi per-sampel tidak diaudit** pada checkout tersebut. Ini bukan hasil Macro-F1 CP2 dan bukan bukti training pretrained.

## 3. Data dan input NLP (selesai)

Perintah:

```bash
bash scripts/cp2_ai_center_run.sh init-data
bash scripts/cp2_ai_center_run.sh audit
```

Output operator `init-data`:

```text
Mengunduh sumber: INDOMEME_Dataset.zip
VALID SPLIT 4368e5767d73dfe988f9734dde59a146e125a0cda82c2783ee8eff46acab1a39: fit=3214, validation=804, test=1005
```

Output operator `audit`:

```text
AUDIT OK. Output privat: ~/kelompok-7-private/cp2_runs/audits/nlp_input_audit.json
```

| Partisi | Jumlah meme | Kegunaan |
| --- | ---: | --- |
| Fit | 3.214 | Training |
| Validation | 804 | Pemilihan checkpoint |
| Test resmi | 1.005 | **Belum dievaluasi untuk CP2** |

**SHA-256 manifest split:** `4368e5767d73dfe988f9734dde59a146e125a0cda82c2783ee8eff46acab1a39`.  
Tokenizer IndoBERTweet dilaporkan berhasil mengambil berkas konfigurasi/vocab. File JSON lengkap di atas adalah **output privat di server**, bukan berkas yang telah diunggah atau dibaca asisten. Karena itu statistik detail audit token tidak diklaim di sini.

## 4. Kondisi GPU dan antrean

Hasil observasi terakhir dari operator:

```text
0, NVIDIA A100-SXM4-80GB, 46404 MiB, 81920 MiB, 100 %
```

Utilisasi GPU 100% saat pemeriksaan, meskipun VRAM bebas masih sekitar 45,3 GiB. Daftar aplikasi menunjukkan beberapa proses GPU aktif. **Nama, PID, dan lokasi proyek milik pengguna lain tidak dicantumkan** demi menghindari penyebaran informasi yang tidak diperlukan.

Perintah `sbatch`, `squeue`, `sinfo`, `srun`, `qsub`, dan `qstat` **tidak tersedia** dalam terminal JupyterLab ini, dan `SLURM_JOB_ID` tidak disetel. Hal ini **tidak membuktikan** platform tidak memiliki mekanisme penjadwalan melalui portal/admin.

**Keputusan operasional:** tidak meluncurkan kalibrasi/training pada GPU yang sedang penuh; cek alokasi dan aturan admin AI Data Center sebelum menjalankan tugas berat. Tidak menghentikan pekerjaan pengguna lain.

## 5. Tahap yang belum selesai

| Pekerjaan | Status |
| --- | --- |
| Commit source NLP & PR #4 | Selesai |
| Setup environment, 27 unit test, verifikasi CP1 | Selesai berdasarkan output operator |
| Inisialisasi split dan audit input NLP | Selesai berdasarkan output operator |
| Kalibrasi attention sintetis (GPU) | **Belum dijalankan** |
| Kalibrasi IndoBERTweet B0 (GPU) | **Belum dijalankan** |
| Kalibrasi ViT (GPU) | **Belum dijalankan** |
| B0 smoke 32 fit/16 validation | **Belum dijalankan** |
| B0 pilot 1 epoch full fit/validation | **Belum dijalankan** |
| B0 full training, validasi Macro-F1 | **Belum dijalankan** |
| Integrasi multimodal B1–M3, robustness, review CV | **Belum selesai** |
| Evaluasi final CP2 pada test resmi | **Belum dijalankan** |

## 6. Langkah saat GPU dialokasikan/tersedia

Dari **root repository** dan environment `cp2-nlp-py312`, setelah memastikan izin penggunaan GPU dan tidak berebut dengan pekerjaan aktif:

```bash
git log -1 --oneline
nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu --format=csv,noheader
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
bash scripts/cp2_ai_center_run.sh calibrate-attention
# Tinjau hasilnya dahulu sebelum langkah berikutnya:
bash scripts/cp2_ai_center_run.sh calibrate-b0
bash scripts/cp2_ai_center_run.sh calibrate-vision
bash scripts/cp2_ai_center_run.sh smoke-b0
bash scripts/cp2_ai_center_run.sh pilot-b0
```

Pelaksanaan harus bertahap, **bukan** menjalankan semua perintah secara otomatis. Training penuh B0 hanya setelah pilot dan persetujuan protokol. Hasil metrik/checkpoint/dataset privat tetap di storage server; hanya ringkasan agregat terverifikasi yang boleh masuk GitHub.

## 7. Referensi

- [PR #4 — Implementasi NLP](https://github.com/josephp77-CyS/indomeme-kelompok-7/pull/4)
- [PR #3 — Protokol NLP](https://github.com/josephp77-CyS/indomeme-kelompok-7/pull/3)
- [PR #2 — Desain CV](https://github.com/josephp77-CyS/indomeme-kelompok-7/pull/2)
