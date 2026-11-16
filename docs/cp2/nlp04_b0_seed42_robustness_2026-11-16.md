# CP2 NLP04 — Robustness Caption/OCR pada B0 seed 42

**Tanggal pencatatan:** 16 November 2026  
**Status:** Evaluasi validation **selesai** untuk B0 seed 42; **ringkasan output terminal operator**, bukan inspeksi langsung file JSON. Metrik lengkap per-tugas dan prediksi belum diaudit; jangan menandai seluruh NLP04 (semua seed dan varian) sebagai selesai.

## 1. Asal hasil dan kondisi

Operator menjalankan CLI `python -m cp2.nlp_cv_eval` pada AI Data Center (environment `cp2-nlp-py312`), menggunakan `best.pt` dari `B0_full_seed42`, `--variant B0 --seed 42 --require-gpu`, dan workspace split CP2 yang sebelumnya diaudit.

Hasil dikirim melalui output terminal operator sebagai berikut:

```text
normal: mean raw validation Macro-F1=0.782072
no_caption: mean raw validation Macro-F1=0.692749
no_ocr: mean raw validation Macro-F1=0.756750
PRIVATE validation-only results: .../cp2_runs/B0_robustness_seed42
```

Mode `normal` adalah Caption+OCR. `no_caption` hanya OCR; `no_ocr` hanya Caption. Kondisi teks dibangun ulang dan ditokenisasi ulang, **tanpa retraining dan tanpa mengubah gambar/label/ID**. B0 tidak mempunyai jalur ViT sehingga tidak ada kondisi `visual_off`. CLI menjalankan evaluasi pada split **validation**, bukan official test.

| Kondisi | Sumber teks aktif | Mean raw validation Macro-F1 | Delta vs normal |
| --- | --- | ---: | ---: |
| Normal | Caption + OCR | **0.782072** | — |
| No-caption | OCR saja | **0.692749** | **-0.089323** |
| No-OCR | Caption saja | **0.756750** | **-0.025322** |

Selisih tersebut merupakan **poin Macro-F1**, bukan perubahan persentase relatif. Angka normal konsisten dengan selection B0 seed 42 best checkpoint `0.7820723139803762` yang telah diverifikasi sebelumnya.

## 2. Interpretasi awal dan batasan

Checkpoint B0 seed 42 lebih sensitif terhadap penghapusan Caption daripada penghapusan OCR pada validation. Penghapusan Caption mengurangi Macro-F1 lebih besar (0.089323 poin) dibanding penghapusan OCR (0.025322 poin). **Belum boleh menyimpulkan signifikansi statistik, kepentingan kausal token, perilaku seed 123, atau pola pada model multimodal**, karena output agregat tunggal belum membuktikannya.

B0 memiliki teks caption yang merupakan deskripsi gambar; menghapus OCR tidak serta-merta menghilangkan semua informasi visual dari input teks. Tidak ada hyperparameter ataupun checkpoint dipilih ulang berdasarkan robustness.

## 3. Bukti tersimpan secara privat

Output default pada AI Data Center:
`~/kelompok-7-private/cp2_runs/B0_robustness_seed42/`.

Program menulis `robustness_metrics.json` dan file `validation_*_predictions_PRIVATE.csv`. **Jangan commit CSV prediksi per-ID atau bobot `.pt`** ke GitHub. Jika membagikan JSON agregat untuk audit, periksa terlebih dahulu metadata/path sensitif.

## 4. Tindak lanjut

1. **Baca** `robustness_metrics.json` agar Macro-F1 **hate** dan **appropriate** per kondisi, confusion matrix, raw/constrained, dan rescue/hurt dapat divalidasi dari artefak aktual.
2. Jalankan evaluasi robustness seed 123 pada lingkungan yang memiliki `B0_full_seed123/best.pt` (misalnya Colab/Drive privat); catat perbedaan environment.
3. Setelah CV menyiapkan checkpoint multimodal valid, jalankan no-caption/no-OCR serta visual-off pada B1/B2/M1/M2/M3 dengan evaluator yang sama, lalu lakukan NLP05.
4. Jangan menggunakan official test untuk memilih arsitektur atau ablation.

**Status yang tepat:** `NLP04 B0 seed42 validation robustness: executed (operator stdout)`; `NLP04 keseluruhan: in progress`.
