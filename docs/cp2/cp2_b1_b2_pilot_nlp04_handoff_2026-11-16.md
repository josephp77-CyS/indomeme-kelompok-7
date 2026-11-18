# INDOMEME CP2 — Handoff NLP/CV: B1–B2 smoke/pilot AI Data Center + NLP04 B0 dua seed

**Tanggal laporan:** 16 November 2026 (Asia/Jakarta)  
**Penanggung jawab dokumentasi:** jalur NLP, untuk review bersama CV/Joseph  
**Repository / working branch:** `josephp77-CyS/indomeme-kelompok-7`, `cp2/nlp-implementation` ([PR NLP #4](https://github.com/josephp77-CyS/indomeme-kelompok-7/pull/4))  
**Dokumen desain CV:** [CV01 pada branch `cp2/cv-design`](https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/cp2/cv-design/docs/cp2/cv-design.md) ([PR CV #2](https://github.com/josephp77-CyS/indomeme-kelompok-7/pull/2))  
**Status:** **Smoke dan pilot B1/B2 telah dieksekusi**, tetapi **belum full training**. **NLP04 B0 seed 42 dan seed 123 sudah dievaluasi pada validation**, tetapi NLP04 lintas model belum selesai. Semua angka di sini adalah *validation*, **bukan official test**.

> **Tingkat verifikasi:** Hasil smoke/pilot B1/B2 diambil dari **stdout Terminal AI Data Center yang dilaporkan operator dalam percakapan**, belum diperiksa langsung terhadap `result.json`, `history.json`, `validation_metrics.json`, atau log resource. Hasil NLP04 seed 123 dibaca dari **file notebook Colab yang sudah dieksekusi dan dibagikan pengguna**, bukan membaca langsung `robustness_metrics.json` di Google Drive. Jangan menafsirkan laporan ini sebagai audit sumber daya atau skor test resmi.

## 1. Eksperimen AI Data Center: B1 dan B2

Operator menggunakan environment `(cp2-nlp-py312)`, repository pada `~/Kelompok 7-INDOMEME/indomeme-kelompok-7`, dan skrip `bash scripts/cp2_mm_run.sh`. Berdasarkan implementasi skrip pada branch ini, `smoke` mengambil hingga 32 fit + 16 validation dan `pilot` menggunakan **seluruh fit 3.214 + validation 804** pada **1 epoch**. Model mengikuti rancangan CV01: **B1 concatenation (IndoBERTweet + ViT)** dan **B2 pooled gate (IndoBERTweet + ViT)** dengan dua head (`hate`, `appropriate`). Runner memilih checkpoint memakai rata-rata raw Macro-F1 validation kedua tugas.

### 1A. Smoke test — sudah berhasil

| Kondisi | Perintah | Epoch | Train loss | Mean raw validation Macro-F1 | Waktu epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| B1 | `bash scripts/cp2_mm_run.sh smoke-b1` | 1 | 0.6879 | **0.4051** | **13.3 dtk** |
| B2 | `bash scripts/cp2_mm_run.sh smoke-b2` | 1 | 0.7163 | **0.3374** | **21.5 dtk** |

**Bukti stdout:** `B1 smoke finished` dan `B2 smoke finished`; output privat `~/kelompok-7-private/cp2_runs/B1_smoke_seed42/` dan `B2_smoke_seed42/`. Pada eksekusi pertama B1 diunduh Parquet resmi `train-00000-of-00001.parquet` untuk gambar. Pesan *slow image processor* dari Transformers bersifat informatif, **bukan error**.

### 1B. Pilot seluruh data — keduanya berhasil

| Kondisi | Perintah | Data fit / validation | Epoch | Train loss | Mean raw validation Macro-F1 | Waktu epoch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| **B1** | `bash scripts/cp2_mm_run.sh pilot-b1` | 3.214 / 804 | 1 | **0.5354** | **0.7672** | **1.084,8 dtk** (18 m 04,8 dtk) |
| **B2** | `bash scripts/cp2_mm_run.sh pilot-b2` | 3.214 / 804 | 1 | **0.5344** | **0.7542** | **1.044,7 dtk** (17 m 24,7 dtk) |

**Potongan bukti operator (stdout):**

```text
B1 epoch 1: loss=0.5354, raw-val-macro-F1=0.7672, seconds=1084.8
B1 pilot_full_data finished; output PRIVAT: .../cp2_runs/B1_pilot_full_data_seed42

B2 epoch 1: loss=0.5344, raw-val-macro-F1=0.7542, seconds=1044.7
B2 pilot_full_data finished; output PRIVAT: .../cp2_runs/B2_pilot_full_data_seed42
```

**Interpretasi yang benar:** pilot adalah *training* sungguhan yang melakukan pembaruan bobot, tetapi hanya 1 epoch sebagai uji integrasi penuh dan estimasi biaya. **Skor 0.7672 / 0.7542 bukan skor final B1/B2; tidak boleh ditampilkan sebagai hasil full CP2 atau dibandingkan langsung dengan B0 full 5-epoch sebagai pembuktian keunggulan.** Nilai kedua pilot berbeda 0.0130 poin Macro-F1, tetapi perbedaan ini belum memiliki pembuktian multi-seed atau hasil full training.

Checkpoint dan prediksi individual **tidak** diunggah ke GitHub. Direktori privat yang dilaporkan:
- `~/kelompok-7-private/cp2_runs/B1_pilot_full_data_seed42/`
- `~/kelompok-7-private/cp2_runs/B2_pilot_full_data_seed42/`

**Belum diverifikasi dari file aktual:** epoch history lengkap, Macro-F1 hate dan inappropriate masing-masing, confusion matrix, jumlah parameter aktual, peak CUDA allocated/reserved, disk tersedia, checkpoint `best.pt/last.pt`, durasi save/load, GPU UUID/utilisasi serta SHA commit persis saat run. Operator/reviewer perlu memeriksa `config.json`, `history.json`, `result.json`, `validation_metrics.json`, dan `parameter_summary.json` di folder privat.

## 2. NLP04 — B0 robustness dua seed (AI Data Center dan Google Drive/Colab)

Ini **uji evaluasi**, bukan pelatihan ulang. Kedua model B0 sebelumnya sudah selesai full training, dan robustness dievaluasi dengan checkpoint `best.pt` pada split **validation**: `normal` = Caption+OCR, `no_caption` = OCR saja, `no_ocr` = Caption saja. Teks untuk kondisi ablation ditokenisasi ulang; label dan split tetap sama.

**Bukti seed 42 (AI Data Center):** [laporan output operator](nlp04_b0_seed42_robustness_2026-11-16.md).

**Bukti seed 123 (Google Drive):** [Notebook NLP04 B0 seed 123](https://colab.research.google.com/drive/1QkGNlQAcW2g9LsrbY-dPjOpske3A9nkn?usp=sharing) — notebook berisi sel eksekusi yang menemukan `B0_full_seed123/best.pt` pada folder privat Drive, menjalankan pengambilan kode, `init-data` dengan manifest historis, memanggil `cp2.nlp_cv_eval --variant B0 --seed 123` tanpa exception, kemudian mencetak tabel dan `rescue/hurt`. Hasil privat notebook: `MyDrive/INDOMEME_CP2_colab_runs/B0_robustness_seed123/robustness_metrics.json` serta `validation_*_predictions_PRIVATE.csv`. Data tersebut **tetap di Drive** dan tidak di-commit.

### 2A. Mean raw validation Macro-F1

| Kondisi | Seed 42 | Seed 123 | Rerata dua seed | Delta rerata vs normal |
| --- | ---: | ---: | ---: | ---: |
| **Normal** | 0.782072 | 0.771214 | **0.776643** | — |
| **No-caption (OCR saja)** | 0.692749 | 0.678822 | **0.685786** | **-0.090858** |
| **No-OCR (Caption saja)** | 0.756750 | 0.754375 | **0.755563** | **-0.021081** |

Angka ditampilkan sesuai pembulatan stdout / tabel notebook; perhitungan menggunakan enam angka desimal sehingga presisi akhir terbatas.

**Interpretasi awal:** pada **dua seed B0** ini, menghapus Caption menurunkan skor lebih besar daripada menghapus OCR. Hal tersebut menunjukkan sensitivitas prediksi terhadap perubahan sumber teks pada validation, **bukan bukti kepentingan kausal, efek universal, atau signifikansi statistik**.

### 2B. Hasil per tugas dan rescue/hurt — Colab seed 123

| Mode | Hate raw Macro-F1 | Inappropriate raw Macro-F1 | Raw constraint violations |
| --- | ---: | ---: | ---: |
| Normal | 0.794117 | 0.748310 | 0 |
| No-caption | 0.689829 | 0.667814 | 2 |
| No-OCR | 0.777689 | 0.731061 | 0 |

Rescue/hurt vs *normal* dari output notebook (per-ID, per-task):

| Mode | Hate rescue / hurt | Inappropriate rescue / hurt |
| --- | ---: | ---: |
| No-caption | 53 / 97 | 65 / 107 |
| No-OCR | 26 / 34 | 23 / 36 |

Rincian seed 42 per tugas dan nilai JSON full seed 123 masih perlu dicocokkan langsung ke `robustness_metrics.json` privat oleh operator. Referensi pendukung: [analisis NLP04 dua seed](nlp04_b0_two_seed_robustness_2026-11-16.md).

## 3. Peta progres penelitian dan kontribusi bersama

| Pekerjaan | Status per 16 Nov 2026 | Bukti / ketergantungan |
| --- | --- | --- |
| B0 full training seed 42 dan 123 | **Selesai** | [hasil NLP02–NLP03](nlp02_nlp03_results_2026-11-16.md) |
| B0 NLP04 robustness seed 42 | **Selesai pada validation** | AI Data Center operator |
| B0 NLP04 robustness seed 123 | **Selesai pada validation** | Notebook Colab Google Drive |
| B1 smoke + pilot seed 42 | **Selesai (dilaporkan operator)** | Stdout + artefak privat |
| B2 smoke + pilot seed 42 | **Selesai (dilaporkan operator)** | Stdout + artefak privat |
| B1/B2 full training | **BELUM** | Menunggu review CV/NLP, resource guard dan persetujuan GPU |
| M1–M3 end-to-end/training | **BELUM** | Implementasi + integrasi CV diperlukan |
| NLP04 B1–M3 | **BELUM** | Membutuhkan full checkpoint model |
| NLP05 lintas-model | **BELUM lengkap** | Perlu full prediction per-ID dari model yang comparable |

**Kepemilikan bersama:** NLP sudah menyiapkan `src/cp2/nlp_cv_bridge.py`, `nlp_cv_eval.py`, evaluator dua head dan validasi token/mask; candidate B1/B2 berada di `mm_models.py`/`mm_runner.py`. CV/Joseph tetap perlu me-review fusion B1/B2 dan menyelesaikan M1/M2/M3. **Tidak ada perubahan pada branch `cp2/cv-design` dari pencatatan ini.**

## 4. Handoff ke Joseph — tindak lanjut yang perlu dilakukan

1. **Review CV01 vs kode B1/B2 kandidat** (`mm_models.py`, `mm_data.py`, `mm_runner.py`): verifikasi rumus concat/pooled gate, source teks Caption+OCR tanpa Topic, pretrained IndoBERTweet/ViT, output dua head dan visual-off.
2. **Audit output pilot privat** bersama operator: dua tugas secara terpisah, parameter, `history.json`, `validation_metrics.json`, `result.json`, file checkpoint, `peak_gpu_allocated_bytes` dan `peak_gpu_reserved_bytes`, disk dan GPU yang dialokasikan; simpan *hanya agregat aman* pada dokumentasi berikutnya.
3. **Preflight dan freeze keputusan eksperimen**: konfirmasi review PR #2/#3/#4, estimasi runtime/storage berdasarkan pilot 1 epoch + overhead, izin penggunaan satu worker GPU, dan protokol final sebelum full training. Jangan menjalankan B1 dan B2 full bersamaan.
4. **Setelah syarat terpenuhi**, jalankan full seed 42 **satu per satu**, catat raw validation metrics kedua task, best epoch, penggunaan memori dan per-ID privat; seed berikutnya hanya sesuai anggaran. Pilot bukan checkpoint final dan **full run dimulai dari bobot pretrained awal**, tidak melanjutkan checkpoint pilot.
5. **Lanjut NLP04/NLP05** ketika full checkpoint B1/B2 tersedia, dan M1–M3 setelah integrasi resmi CV.

### Panduan audit output (hanya membaca file; tidak memulai training)

Dari terminal AI Data Center:

```bash
cd "$HOME/Kelompok 7-INDOMEME/indomeme-kelompok-7"
python - <<'PY'
import json
from pathlib import Path
root = Path.home() / "kelompok-7-private/cp2_runs"
for model in ("B1", "B2"):
    d = root / f"{model}_pilot_full_data_seed42"
    print(f"\n=== {model} pilot ===")
    for name in ("result.json", "history.json", "validation_metrics.json", "parameter_summary.json"):
        p = d / name
        if not p.is_file():
            print(name, "MISSING")
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        if name == "history.json":
            print(name, json.dumps(obj, indent=2))
        elif name == "validation_metrics.json":
            print(name, {
                "hate_macro_f1": obj["hate"]["macro_f1"],
                "appropriate_macro_f1": obj["appropriate"]["macro_f1"],
                "selection_macro_f1_raw": obj["selection_macro_f1_raw"],
                "constraint_violations": obj["raw_constraint_violations"],
            })
        else:
            print(name, json.dumps(obj, indent=2))
    print("best.pt exists:", (d / "best.pt").is_file())
    print("last.pt exists:", (d / "last.pt").is_file())
PY
```

**Privasi dan reproducibility:** Jangan unggah `.pt`, gambar, file `*_PRIVATE.csv`, credential atau path lokal rahasia ke GitHub. Pastikan link notebook Drive tidak mempunyai akses edit publik yang tidak diperlukan; ubah sharing menjadi *Restricted* dan bagikan hanya ke reviewer yang relevan. Catat `git rev-parse HEAD` dan versi environment aktual pada audit operator nanti.
