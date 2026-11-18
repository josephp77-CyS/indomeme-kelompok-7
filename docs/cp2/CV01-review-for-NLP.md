# CP2 — Review CV01 untuk sinkronisasi NLP/CV

**Status:** Review awal berbasis dokumen dan source publik, **belum merupakan persetujuan final**.  
**Tanggal:** 11 November 2026

## Keputusan yang dapat diterima sebagai dasar kerja

1. Dua label tetap biner, dengan `appropriate=1` berarti inappropriate.
2. Setiap model B0–M3 memiliki dua head, sumber teks `Caption+OCR` dan tidak menggunakan topik anotasi manusia.
3. M2 vs M1 menguji penambahan scalar gate; M3 vs M2 menguji visual dropout; M2 vs B2 menguji mekanisme token–patch tetapi beda kapasitas.
4. `content_mask` untuk pooling cross-attention berbeda dari `attention_mask` encoder.
5. Evaluasi visual-off mematikan jalur ViT langsung, **tidak** menghilangkan informasi gambar pada caption.
6. CP1 frozen harus dipertahankan; perubahan CP2 berupa source/config/runner baru.

## Temuan prioritas sebelum implementasi

| Prioritas | Temuan | Konsekuensi / tindakan |
| --- | --- | --- |
| P0 | Baseline E101/E110 memakai `Caption+OCR+Topic`, sementara CV01 mengubah fitur menjadi `Caption+OCR` | B0/B1 harus dikonfigurasi dan dilatih ulang; jangan bandingkan angka lama dengan CP2 sebagai kontrol setara |
| P0 | Selection E101/E110 memakai constrained Macro-F1 appropriate; CV01 mengusulkan raw dua task | Implementasikan selection baru di runner/evaluator CP2, dan kunci kebijakan sebelum eksperimen |
| P0 | `master_training.MemeDataset` belum menghasilkan `content_mask` | Implementasi pipeline CP2 harus menambah field dengan tokenizer special IDs; tes separator literal dan padding |
| P0 | `master_training.run_training` membuat `MemeClassifier` secara langsung | Runner baru mesti menginstansiasi class B0–M3 yang tepat; cek train/validation/test sama-sama memakai model baru |
| P1 | `text_values` CP1 menggunakan `str(row[c])` | Berisiko menjadikan nilai null sebagai literal `nan` pada input baru; tambah sanitasi dan audit |
| P1 | Dataset memiliki dua grup gambar identik lintas split resmi menurut audit CP1 | Laporkan keterbatasan evaluasi; sensitivitas terdeduplikasi dipisahkan dari test resmi |
| P1 | CV01 menetapkan MHA pada 49 patch serta `content_mask` | Unit test perlu memastikan 49 patch, pooling semua mask nol, dimensi, dan mask visual setelah bias |
| P1 | B0, B1 dan M1/M2/M3 berbeda jumlah parameter non-encoder | Jangan mengeklaim seluruh pembanding memiliki kapasitas identik |
| P2 | Pelatihan multi-seed dan perangkat GPU belum dikalibrasi | Putuskan 6/14/18 run setelah smoke, kalibrasi dan pilot; tidak mengklaim training selesai |

## Pengecekan source yang dilakukan

- `src/master_training.py`: `MemeDataset` saat ini hanya mengeluarkan `input_ids` dan `attention_mask`; evaluator sudah membedakan raw dan constrained; `run_training` memilih model historis langsung.
- `src/master_workflow.py`: `VARIANTS` sudah memiliki `Caption+OCR`; metode split dan audit duplikasi tersedia.
- `results/runs/E101/config.json` dan `E110/config.json`: versi backbone, seed, loss, split SHA dan policy selection historis cocok dengan kebutuhan mencatat perubahan CP2.
- `docs/cp2/cv-design.md`: rancangan sudah membahas enam model dan smoke check CV-T01–16, tetapi belum mengklaim implementasi tersedia.

## Agenda review bersama Joseph

1. Konfirmasi label dan data source tanpa topic.
2. Setujui raw validation selection dan constrained reporting sekunder.
3. Setujui konstruksi `content_mask` dan respons terhadap data kosong.
4. Tegaskan perubahan source minimal, runner baru dan larangan mengubah CP1 frozen.
5. Sepakati acuan testing, kapasitas hardware dan format bukti eksperimen.

## Pekerjaan berikutnya

1. Review draf `docs/cp2/nlp-protocol.md` bersama Joseph.
2. Setelah disetujui, selesaikan dan ajukan PR `cp2/nlp-protocol` sebagai CP2-NLP01.
3. Mulai NLP02 (input dan B0) dan NLP03 (evaluator) secara paralel dengan implementasi CV. Jangan memakai test sebelum freeze.

**Batas verifikasi:** Pemeriksaan ini dilakukan dari file publik raw pada branch CV dan bukan eksekusi tes pada checkout utuh/GPU; tidak ada commit/PR atau perubahan GitHub yang dilakukan.
