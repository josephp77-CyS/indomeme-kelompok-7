# CP2 — Kontribusi NLP pada B1/B2/M1/M2/M3 (integrasi bersama CV)

**Tanggal:** 16 November 2026. **Branch:** `cp2/nlp-implementation` (PR #4, draft).  
**Status:** Modul antarmuka/evaluasi dan pengujian sintetis ditambahkan. Tes lokal pada salinan modul sepadan: **46 passed** (25 tes NLP lama + 21 tes integrasi). **Belum ada uji model pretrained/GPU end-to-end untuk M1–M3 dan belum ada training B1/B2/M1–M3.** Penilaian implementasi CV dan persetujuan protokol tetap diperlukan.

## 1. Audit commit sebelum integrasi

- `src/cp2/nlp_data.py` / `nlp_runner.py`: dataset teks, split, B0 seed 42/123 selesai, fungsi loss dan selection sama untuk semua model.
- `src/cp2/nlp_evaluation.py`: evaluasi **raw** dua task, `appropriate_constrained` sebagai sekunder, confusion matrix, `rescue_hurt` per-ID.
- `src/cp2/mm_models.py`, `mm_data.py`, `mm_runner.py`: **kandidat B1/B2** fusion & train sudah di-commit, tetapi belum GPU-tested/di-review CV; bukan implementasi M1–M3.
- `src/cp2/fusion_reference.py`: *synthetic-tensor demonstrator*, **bukan** M1–M3 resmi yang memakai pretrained IndoBERTweet/ViT.
- M1/M2/M3 **belum punya model full pretrained maupun training runner CV yang disetujui**. Jangan melaporkan sudah selesai.

## 2. Kode baru NLP untuk seluruh enam varian

| Source | Kepemilikan/fungsi |
| --- | --- |
| `src/cp2/nlp_cv_bridge.py` | **NLP**: pemeriksaan batch `input_ids/attention_mask/content_mask/labels/image_path` + `pixel_values`, anti-label-leakage, dua logits `[B,2]`, evaluator NLP03 **yang sama untuk B0–M3**, visual-off eksplisit |
| `src/cp2/nlp_cv_eval.py` | **NLP**: CLI untuk evaluasi robustness *validation only*, load `best.pt` full yang sah, verifikasi config/manifest; `normal/no_caption/no_ocr`, tambah `visual_off` pada model multimodal |
| `src/cp2/mm_data.py` | **shared**: parameter `mode` ditambahkan sehingga Caption/OCR ditokenisasi **ulang**, mempertahankan citra dan ID yang sama |
| `src/cp2/mm_runner.py` | **shared**: runner B1/B2 kini memeriksa batch dan logits via `NLPCheckedModel/CheckedLoader`, memilih checkpoint berdasarkan evaluator NLP03 yang sama |
| `tests/test_cp2_nlp_cv_bridge.py` | Kontrak dan robustness diuji dengan fake model untuk **B0,B1,B2,M1,M2,M3**, tanpa mengunduh bobot |
| `scripts/cp2_mm_run.sh` | Aksi `test` juga menjalankan tes bridge |

**Pembagian tugas:** CV tetap bertanggung jawab atas implementasi arsitektur B1/B2/M1–M3 (ViT, concat, gated fusion, token–patch cross-attention, visual dropout). NLP bertanggung jawab atas representasi teks, mask, dua label, input consistency, evaluasi, robustness, dan error analysis pada **setiap** model. Integrasi, smoke/pilot, validation/selection, hardware profiling, dan acceptance dilakukan **bersama**.

## 3. Antarmuka yang CV harus serahkan

```python
from torch import nn

class CVFusionModel(nn.Module):
    def forward(self, batch, visual_off: bool = False):
        # input: input_ids [B,L], attention_mask [B,L],
        #        content_mask bool [B,L], pixel_values [B,3,224,224]
        # model CV berhak memakai/mengabaikan content_mask sesuai varian
        # visual_off harus menolkan kontribusi visual pada posisi yang tepat
        return {"hate": logits_hate, "appropriate": logits_appropriate}
```

- **M1:** gunakan `content_mask` ketika pooling cross-attention token–patch; jangan gunakan mask teks sebagai `key_padding_mask` patch.
- **M2:** shared scalar gate sesuai CV01; input NLP tidak boleh memuat outcome label sebagai fitur.
- **M3:** hanya tambah visual dropout training pada M2; eval normal deterministik; mode `visual_off=True` harus nolkan kontribusi akhir ViT, bukan mengosongkan caption.
- **Semua:** output logits mentah *persis* dua head, masing-masing `[B,2]`; cross entropy 0.5/0.5; `selection_macro_f1_raw` pada validation; constrained sekunder.
- CV M1/M2/M3 harus menyediakan **factory yang telah direview** `module:factory(variant=...)`, serta checkpoint `best.pt` dengan skema `{"model": state_dict, "config_hash": sha256(config.json)}` jika ingin menggunakan CLI NLP di bawah. Bila CV punya format berbeda, **review adapter/format dahulu**; jangan paksa load secara diam-diam.

## 4. Instruksi pengujian operator AI Data Center

Dari **root repo** pada branch `cp2/nlp-implementation` (pastikan `git status --short` tidak menunjukkan perubahan penting):

```bash
git pull --ff-only origin cp2/nlp-implementation
export PYTHONPATH="$PWD/src"
bash scripts/cp2_mm_run.sh test
bash scripts/cp2_mm_run.sh preflight
```

`test` tidak butuh data model pretrained dan dapat dijalankan CPU. Jika gagal, **jangan langsung full-training**. Jika semua sukses, reviewer CV memvalidasi B1/B2 dan memulai `smoke-b1` / `smoke-b2` pada GPU yang berizin. Tahap ini **tidak** otomatis menjalankan training di chat.

## 5. Evaluasi NLP04 dengan checkpoint yang sudah selesai

B0 seed 42 **sudah memiliki** `best.pt` privat pada AI Data Center. Evaluasi robustness B0, tanpa retraining:

```bash
python -m cp2.nlp_cv_eval \
  --variant B0 --seed 42 \
  --checkpoint "$HOME/kelompok-7-private/cp2_runs/B0_full_seed42/best.pt" \
  --workspace "$HOME/kelompok-7-private/cp2_data" \
  --output "$HOME/kelompok-7-private/cp2_runs/B0_robustness_seed42" \
  --require-gpu
```

Untuk **B1/B2**, tunggu selesai *full run* masing-masing; ubah `--variant`, `--checkpoint`, dan `--output` secara konsisten. Untuk M1–M3 tambahkan `--cv-factory nama_module:build_model` **setelah** model CV resmi tersedia.

**Output privat:** `robustness_metrics.json`, `validation_normal_predictions_PRIVATE.csv`, `validation_no_caption_predictions_PRIVATE.csv`, `validation_no_ocr_predictions_PRIVATE.csv` dan, untuk multimodal, `validation_visual_off_predictions_PRIVATE.csv`. Per kondisi dihitung Macro-F1, confusion matrix, constrained secondary dan rescue/hurt relatif kondisi normal dengan pairing ID. **Official test tidak dipakai.**

**Jangan unggah** `.pt`, data gambar mentah, atau `*_PRIVATE.csv` ke GitHub. Arsip empat/lima JSON agregat aman setelah diperiksa tidak mengandung nama/path privasi.

## 6. Acceptance & batas klaim

- [x] Kontrak batch/evaluasi dua task B0–M3 tersedia secara kode, dengan tes sintetis lokal.
- [x] Kode B1/B2 menautkan evaluator bersama dan pemeriksaan input pada runner.
- [x] NLP04 mampu memeriksa no-caption/no-OCR dan visual-off lewat interface yang sama.
- [ ] Jalankan tes repo pada AI Data Center lalu catat jumlah tes dan versi environment.
- [ ] Jalankan NLP04 B0 seed 42/123; simpan skor dari checkpoint asli.
- [ ] Uji integrasi B1/B2 dengan pretrained IndoBERTweet + ViT, loss/backward/save-load/VRAM.
- [ ] Reviewer CV menyepakati M1/M2/M3 factory dan output interface; implementasi resmi belum ada.
- [ ] Integrasi M1–M3 dengan runner/checkpoint CV yang sesungguhnya.
- [ ] Full training B1–M3 dan pengukuran Macro-F1 nyata.
- [ ] Analisis NLP05 rescue/hurt antar-varian setelah prediksi validasi ber-ID tersedia.

**Status penelitian:** B0 full dua seed selesai, B1/B2 candidate + NLP bridge tertulis namun belum GPU-validated, M1–M3 CV model belum tersedia. Pekerjaan ini bukan N1/N2 tambahan; ini implementasi **tanggung jawab NLP pada enam varian inti**.
