# Catatan telaah kebaruan CP2 — pencarian literatur terbatas, 15 Nov 2026

Status: **bukan** systematic literature review atau bukti pertama di dunia. Perlu memeriksa paper lengkap, daftar pustaka, dan perbedaan dataset/tugas/kapasitas sebelum menetapkan klaim final.

## Publikasi relevan

1. **Pamungkas dkk. (2026), _Decoding hate in memes: multimodal and multitask approaches for low-resource Indonesian social media_**, PeerJ CS, DOI 10.7717/peerj-cs.3736. Dataset/acuan CP1. https://peerj.com/articles/cs-3736/
2. **Huang dkk. (2025), _Low-Rank Adapter Layers and Bidirectional Gated Feature Fusion for Multimodal Hateful Memes Classification_**, Computers, Materials & Continua, DOI 10.32604/cmc.2025.064734. Low-rank adapter terkait parameter-efficient adaptation, sedangkan gated bidirectional fusion adalah mekanisme fusion tersendiri. https://www.techscience.com/cmc/v84n1/61770
3. **CHARM (2026), _Cross-modal hierarchical attention for recognizing malicious memes_**, Neurocomputing, DOI 10.1016/j.neucom.2025.131938. Menggunakan cross-modal guidance dan adaptive gating pada meme. https://www.sciencedirect.com/science/article/pii/S0925231225026104
4. **DARC-CLIP (2026), _Dynamic Adaptive Refinement with Cross-Attention for Meme Understanding_**, ICASSP, DOI 10.1109/ICASSP55912.2026.11462868. Memperlihatkan attention refinement plus task-sensitive adapters pada berbagai task meme. https://doi.org/10.1109/ICASSP55912.2026.11462868
5. **Ren dkk. (2026), _A survey of multimodal hate meme detection_**, Expert Systems with Applications, DOI 10.1016/j.eswa.2026.132507. Survey konteks kebaruan untuk cross-modal interaction dan gate. https://www.sciencedirect.com/science/article/pii/S095741742601420X

## Klaim yang layak dan yang harus dihindari

**Layak sebagai hipotesis kontribusi yang perlu dibuktikan:** perbandingan terkontrol token-to-patch cross-attention + concat/residual, efek shared versus task-specific scalar gating pada pasangan tugas `hate` dan `inappropriate` di INDOMEME bahasa Indonesia, serta pengujian robust visual-off dan visual dropout per meme. Semua memakai split/protokol konsisten, seed berulang bila budget tersedia.

**Hindari klaim:** penemuan cross-attention pertama, gating pertama, task-aware multimodal fusion pertama, peningkatan SOTA sebelum test, atau dropout memberi robustness sebelum uji. DARC-CLIP dan CHARM membuat klaim 'task-sensitive cross-attention/gate belum pernah ada' terlalu luas.

**Penting:** judul awal yang menyebut **Task-Adaptive Gated Fusion** belum didukung oleh arm CV01 enam konfigurasi karena tidak ada task-specific gating. Judul tersebut baru cocok setelah arm task gate benar-benar diimplementasikan dan dibandingkan; jika tidak, judul sebaiknya menyebut `Shared Scalar Gating` dan `Visual Dropout` secara lebih akurat.
