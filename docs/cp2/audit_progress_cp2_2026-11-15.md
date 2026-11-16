# Audit status CP2 INDOMEME — 15 November 2026

**Basis bukti:** inspeksi repository GitHub publik `josephp77-CyS/indomeme-kelompok-7`, membandingkan `main`, `cp2/nlp-protocol`, `cp2/cv-design`, sumber Python, protokol dan PR. Bukan eksekusi model pada AI Data Center; percakapan sebelumnya/PADASOURCE tidak tersedia dalam audit.

## Snapshot GitHub

| Ref | Head SHA | Isi perubahan terhadap main | Status |
|---|---|---|---|
| main | `e573a930539c156925b06222893c0309532b3a4d` | kode CP1 dan bukti hasil | baseline histori |
| cp2/nlp-protocol | `08ff298a07cc50c349468e0be8b152e733dcfb01` | 1 commit, 2 file baru (`docs/cp2/nlp-protocol.md`, `docs/cp2/CV01-review-for-NLP.md`) | [PR #3](https://github.com/josephp77-CyS/indomeme-kelompok-7/pull/3) open |
| cp2/cv-design | `c12f6dfc2ae9fcefedcb196976f31ecdaff03aea` | 3 commit, 5 file README/docs termasuk desain CV | [PR #2](https://github.com/josephp77-CyS/indomeme-kelompok-7/pull/2) open |

`main/src/master_training.py` memiliki `MemeDataset`, `MemeClassifier`, `evaluate`, `run_training`. Secara konkret: Dataset mengeluarkan input_ids/attention_mask tanpa content_mask; model memakai pooled features dan concatenation tanpa token–patch attention; training memakai langsung `MemeClassifier(config)`; checkpoint selection CP1 memakai hate raw + appropriate constrained. `main/src/master_workflow.py` mendukung varian `Caption+OCR` dan pembuatan split historis, yang dapat digunakan lagi untuk CP2.

Desain CV01 menegaskan 6 arm: **B0** text-only, **B1** concat, **B2** pooled gate, **M1** cross-attention residual, **M2** M1+shared scalar gate, **M3** M2+visual dropout. Kode baru untuk keenam model belum ada di `src/` pada dua branch CP2 yang diperiksa.

## Peta progres NLP

| Item | Bukti GitHub | Kategori progres | Pekerjaan berikutnya |
|---|---|---|---|
| NLP01 protokol | `docs/cp2/nlp-protocol.md`, PR #3 | **Dokumen sudah ditulis / PR terbuka / approval belum terbukti** | review/freeze aturan input, label, loss dan selection |
| NLP02 preprocessing & B0 | hanya source CP1 | **Belum menjadi implementasi CP2** | konten mask, tanpa topic, split/hash, B0 re-train |
| NLP03 evaluasi CP2 | evaluator CP1 ada, namun selection berbeda | **Perlu adapter baru + tes** | raw dual-task, constrained sekunder, pairing ID |
| NLP04 robustness | prosedur di dokumen | **Belum ada run** | no-caption/no-OCR, visual-off, dropout evaluation |
| NLP05 analisis hasil | tabel CP1 historis, bukan CP2 | **Belum ada hasil CP2** | seed, rescue/hurt, error analysis, test setelah lock |
| CV implementasi | desain 34 KB, tes konseptual | **Belum ada model CP2** | PyTorch fusion+gate, CV-T01..16, integration |
| AI Data Center | rencana C0..C5 di CV01 | **Kalibrasi GPU CP2 belum terbukti** | profile B0, B1, M3, pilot; logs terverifikasi |

## Isu ilmiah P0

1. `Caption+OCR+Topic` CP1 dibandingkan `Caption+OCR` CP2 adalah **perubahan input**, sehingga tidak fair menggunakan angka CP1 E101/E110 langsung sebagai kontrol.
2. Pemilihan checkpoint CP1 memasukkan constrained appropriateness, CP2 mengusulkan mean **raw** Macro-F1 dua task. Harus disetujui lalu dipakai oleh semua arm.
3. Cross-attention + concat **tidak ada** sebagai arm pada desain CV01; RQ1 usulan tidak sepenuhnya terjawab jika hanya mengikuti enam arm.
4. Shared vs task-specific gate **tidak ada** perbandingan pada CV01; RQ2 belum sepenuhnya terjawab jika hanya M1/M2.
5. `visual-off` tidak menghilangkan informasi visual dari caption buatan/deskripsi; hindari klaim text-only murni.
6. Dataset audit CP1 menunjukkan **dua grup gambar identik lintas train/test resmi**. Ini keterbatasan yang perlu dinyatakan; jangan memodifikasi test resmi secara diam-diam.

## Matriks ablation yang disarankan

| Arm | Tujuan | Prioritas |
|---|---|---|
| B0 text-only CP2 | kontrol teks tanpa topic | wajib |
| B1 concat CP2 | kontrol fusi pooled | wajib |
| B2 pooled gate | gate tanpa token–patch | wajib CV01 |
| XCONCAT token–patch + concat | kontrol langsung RQ1 usulan | **tambahan; belum disetujui** |
| M1 token–patch + residual | RQ1 vs XCONCAT | wajib CV01 |
| M2 shared scalar gate | RQ2 baseline gate | wajib CV01 |
| M2_TASK task-specific scalar gate | RQ2 shared vs task-specific | **tambahan; belum disetujui** |
| M3 visual dropout p=0.1 | RQ3 dibanding M2 yang sama | wajib CV01 |
| Shared/task-specific vector gate | perluasan eksploratif | opsional setelah hasil validation scalar |

Satu seed seluruh arm baru memberi hasil eksploratif, bukan signifikansi. Delapan arm inti baru × 3 seed = **24 run**, berbeda dari rencana 6 × 3 = 18 run CV01; anggaran GPU harus dikalibrasi ulang. Hindari memperbanyak arm sebelum kontrak dan hardware jelas.

## Garis kontribusi antaranggota

**NLP:** audit data, split, komposisi caption+OCR, tokenizer/`content_mask`, label dua head, text baseline B0, raw evaluator, laporan per-task, ablation retokenisasi, robustness pairing, source provenance, kalibrasi B0/attention component. **CV:** ViT pixels/patches, B1/B2/M1/M2/M3, gate, visual mask, parameter tests, patch alignment. **Bersama:** review CV01/NLP01, integrasi end-to-end, C3 GPU B1/M3, lock experimental plan, pilihan model dari validation, test final.

## Tautan source

- [NLP protocol](https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/cp2/nlp-protocol/docs/cp2/nlp-protocol.md)
- [CV design](https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/cp2/cv-design/docs/cp2/cv-design.md)
- [CP1 training](https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/main/src/master_training.py)
- [CP1 workflow](https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/main/src/master_workflow.py)
- [CP1 result config E101](https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/main/results/runs/E101/config.json)
- [CP1 result config E110](https://github.com/josephp77-CyS/indomeme-kelompok-7/blob/main/results/runs/E110/config.json)
