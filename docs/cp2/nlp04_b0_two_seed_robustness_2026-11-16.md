# CP2 NLP04 — Evaluasi robustness B0 dua seed (42 dan 123)

**Diperbarui:** 16 November 2026. **Status:** kedua checkpoint full B0 telah menjalani evaluasi *validation-only* untuk tiga kondisi teks. **Belum seluruh NLP04**: B1/B2 dan M1–M3 belum mendapat robustness full-checkpoint. Official test **tidak digunakan**.

## Bukti dan batas audit

- **Seed 42:** hasil terminal AI Data Center yang dicatat pada [laporan seed 42](nlp04_b0_seed42_robustness_2026-11-16.md), menggunakan B0 `best.pt` full.
- **Seed 123:** [notebook Google Colab yang dibagikan operator](https://colab.research.google.com/drive/1QkGNlQAcW2g9LsrbY-dPjOpske3A9nkn?usp=sharing), `NLP04_B0_seed123_INDOMEME_Colab.ipynb`. Notebook tersimpan dengan enam sel code berstatus dieksekusi, menunjukkan checkpoint `B0_full_seed123/best.pt` ditemukan, `init-data` dan pemeriksaan split selesai tanpa error, CLI `cp2.nlp_cv_eval --variant B0 --seed 123` sukses, dan tabel skor/analisis dicetak. Ini merupakan **output yang disimpan dalam notebook**, bukan audit independen atas isi `robustness_metrics.json` privat ataupun prediksi per-ID.
- Repo: branch `cp2/nlp-implementation`. Evaluator `src/cp2/nlp_cv_eval.py` memuat checkpoint full dan membandingkan input dengan **tokenisasi ulang**, bukan training ulang.

## 1. Mean raw validation Macro-F1

| Mode | Input teks | Seed 42 | Seed 123 | Mean dua seed |
| --- | --- | ---: | ---: | ---: |
| `normal` | Caption + OCR | 0.782072 | 0.771214 | **0.776643** |
| `no_caption` | OCR saja | 0.692749 | 0.678822 | **0.685786** |
| `no_ocr` | Caption saja | 0.756750 | 0.754375 | **0.755563** |

Selisih Macro-F1 dibanding normal:
- Seed 42: no-caption **-0.089323**; no-OCR **-0.025322**.
- Seed 123: no-caption **-0.092392**; no-OCR **-0.016839** (berdasarkan angka enam desimal tabel; notebook menyajikan selisih presisi penuh -0.016838).
- Rerata selisih dua seed: no-caption sekitar **-0.090858**; no-OCR sekitar **-0.021080** poin Macro-F1.

Hasil kedua seed konsisten secara *arah*: menghapus Caption menyebabkan penurunan lebih besar daripada menghapus OCR. Ini bukan bukti kepentingan kausal Caption, efek universal, atau signifikansi statistik; hanya dua seed B0 pada split validation.

## 2. Detail seed 123 (dari output Colab)

| Mode | Macro-F1 hate | Macro-F1 inappropriate | Raw constraint violations |
| --- | ---: | ---: | ---: |
| normal | 0.794117 | 0.748310 | 0 |
| no-caption | 0.689829 | 0.667814 | 2 |
| no-OCR | 0.777689 | 0.731061 | 0 |

**Rescue/hurt terhadap kondisi normal, berdasarkan pairing ID:**  
- no-caption: hate rescue **53**, hurt **97**; inappropriate rescue **65**, hurt **107**.
- no-OCR: hate rescue **26**, hurt **34**; inappropriate rescue **23**, hurt **36**.

Metrik per-task seed 42 untuk ketiga kondisi serta prediksi per-ID tidak disalin ke dokumen ini karena output yang tersedia hanya skor agregat; audit isi JSON privat seed 42 dapat melengkapinya.

## 3. Pengujian berikutnya

1. Periksa JSON metrik lengkap seed 42 dan seed 123 untuk menyelesaikan tabel per-task dua seed, varians/dispersi, dan confusion matrix; simpan CSV prediksi individual dan bobot tetap privat.
2. **B1/B2:** smoke GPU berhasil untuk keduanya. Menurut laporan operator, pilot B1 satu epoch telah selesai dan pilot B2 masih berjalan pada 16 November; skor/resource pilot perlu diperiksa setelah `result.json` keduanya tersedia. Pilot bukan hasil akhir.
3. Review CV/NLP, tes dan preflight, VRAM/disk/waktu pilot, dan izin GPU sebelum `full-b1`/`full-b2`. Jangan menjalankan dua training GPU sekaligus.
4. Setelah masing-masing full checkpoint tervalidasi, jalankan NLP04 untuk B1/B2 dengan kondisi teks yang sama dan `visual_off` sebagai uji tambahan; M1–M3 menunggu implementasi resmi CV.
5. NLP05: bandingkan prediksi per-ID, rescue/hurt, per task dan antar seed **setelah** full runs tersedia. Jangan memilih model berdasarkan official test.

**Catatan keamanan:** link notebook Colab tampak memiliki berbagi publik; batasi akses Drive file ke anggota riset yang diperlukan dan jangan tampilkan file checkpoint atau prediksi privat.
