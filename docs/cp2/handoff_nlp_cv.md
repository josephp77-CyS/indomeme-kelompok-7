# CP2 NLP → CV handoff dan eksperimen tambahan (DRAF / butuh approval)

## I. Batch yang diserahkan NLP

- `input_ids`: int64 `[B,256]`
- `attention_mask`: int64 `[B,256]` (untuk encoder IndoBERTweet)
- `content_mask`: bool `[B,256]` = `attention_mask.bool() & ~isin(input_ids, tokenizer.all_special_ids)`; CLS/SEP/PAD dan separator Caption–OCR tidak masuk pooling
- `labels`: int64 `[B,2]`, indeks 0 hate (`1=hate`), indeks 1 appropriate (`1=inappropriate`)
- `image_path`: list ID untuk audit dan pairing evaluator, tidak masuk neural network
- `pixel_values`: FP32 `[B,3,224,224]`, harus ditambahkan bagian CV pada B1/B2/M1/M2/M3 menggunakan image processor pinned. B0 tidak memerlukan ViT

`src/cp2/nlp_data.py` saat ini menyediakan batch teks + label. Pemasangan `pixel_values` ke dataset multimodal harus dilakukan dengan image byte dari dataset release pinned yang sama, tanpa mengubah urutan ID. Jangan mengubah CP1 `MemeDataset` frozen.

## II. Kontrak evaluator

`model(inputs)` mengembalikan `{"hate": logits[B,2], "appropriate": logits[B,2]}`. Gunakan `cp2.nlp_evaluation.evaluate_model` untuk memperoleh metrik/prediksi validation. Model CV yang memiliki argumen `visual_off` perlu adapter dalam runner robustness tersendiri; fungsi evaluasi dasar menguji normal saja. Loss = 0.5*CE hate + 0.5*CE appropriate. Best checkpoint = maksimal rata-rata dua Macro-F1 **raw** validation; tie -> epoch awal. Constrained `inappropriate_pred=max(hate_pred, inappropriate_raw_pred)` untuk laporan sekunder, bukan selection dan bukan train loss.

Jangan menggunakan model `MemeClassifier` dari CP1 dalam path training CP2: runner lama selalu membuat class tersebut dan belum mengenali varian CP2.

## III. Perbedaan desain CV01 saat ini vs brainstorming

| Klaim eksperimen | Yang sudah tertulis pada CV01 | Perlu dibicarakan |
|---|---|---|
| B0 text-only, B1 concat | Ya, kedua baseline perlu train ulang tanpa topic | Jadikan kontrol minimal |
| Cross-attention + concat | Tidak ada sebagai arm tersendiri | Tambahkan `XCONCAT` sebelum RQ1 dapat dibandingkan langsung |
| Cross-attention + residual | Ya, M1 = `t + masked_mean(MHA(H,P,P))` | Pertahankan M1 |
| Shared scalar gate | Ya, M2 | Tetapkan apakah `g` satu per meme dan dipakai kedua head |
| Task-specific scalar gate | Belum ada | Tambahkan `M2_TASK` dan dua gate terpisah dengan head spesifik |
| Shared/task-specific vector gate | Belum ada | Tahap opsional jika scalar menjanjikan |
| Visual dropout | Ya, M3 = M2 + Bernoulli 0.1 per meme saat training | Uji tanpa dan dengan dropout pada arsitektur/gate sama |
| Visual-off eval | Dirancang, belum dieksekusi | Pastikan meniadakan kontribusi visual **setelah bias**, bukan mengacak teks |

`src/cp2/fusion_reference.py` menyediakan ilustrasi executable pada **tensor sintetis** dengan pilihan concat/residual, gate scalar/vector shared/task, dan visual dropout training-only. Ini bukan model CV pra-latih, dan bukan bukti hasil RQ.

## IV. Rekomendasi urutan ablation untuk menjawab RQ

**RQ1** B0, B1, XCONCAT, M1 (cross-attention residual). Bedakan `B1→XCONCAT` vs `B1→M1`; semuanya pada data, seed, preprocessing, head, LR dan rule selection sama. Jumlah parameter fusion beda; laporkan.

**RQ2** Setelah pilihan RQ1 dibekukan, bandingkan shared scalar vs task-specific scalar pada attention yang sama, dan **simpan dua skor task**. Tambahkan vector gate sebagai hipotesis lanjutan, bukan uji tak terbatas. Kriteria pemilihan untuk eksperimen berikutnya harus dinyatakan sebelum test.

**RQ3** Latih tanpa visual dropout dan dengan dropout pada fusion/gating identik, `p=0.1` per meme. Evaluasi normal dan visual-off pada kedua model; laporkan raw Macro-F1 per task dan selisih penurunan relatif/absolut. No-caption/no-OCR adalah analisis terpisah; retokenisasi diperlukan.

Jangan membandingkan model yang berbeda input `topic` dengan model yang tidak memakai topic sebagai ablation arsitektur murni. Jangan memakai `test` untuk memilih mana eksperimen gate diteruskan.

## V. Approval check sebelum freeze dan kalibrasi AI Data Center

- [ ] Dua PR dokumentasi direview bersama; protokol tidak lagi bertanda *DRAFT* jika sudah disepakati.
- [ ] Tambahan XCONCAT/M2_TASK disetujui atau dijadikan eksplorasi sekunder di luar 6-run CV01.
- [ ] Split SHA, dataset/model revisions, label map, text source dan all-special-token mask konsisten.
- [ ] B0/B1 baseline diaudit supaya bukan CP1 dengan topic.
- [ ] Seluruh varian memakai checkpoint selection raw yang sama, validation per seed.
- [ ] Tes parameter, 49 ViT patches, zero-mask, freeze, save/load dan resume lulus.
- [ ] Profil GPU terintegrasi **B1 dan M3** selesai, bukan hanya jumlah biaya komponen.
- [ ] Data, checkpoint dan individual predictions tersimpan privat; repository hanya source, docs, agregat yang layak dipublikasi.
- [ ] RQ1–RQ3 dan scope claim novelty terdokumentasi dalam analisis.
