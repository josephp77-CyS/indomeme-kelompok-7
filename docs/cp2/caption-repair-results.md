# Hasil regenerasi caption INDOMEME

Eksperimen mengukur dua hal. Pertama, pemilihan item untuk regenerasi saat kapasitas hanya 20%. Kedua, efek melatih classifier sejak awal dengan campuran caption regenerasi dan caption asli. Metrik klasifikasi adalah rata-rata Macro-F1 raw untuk hate dan inappropriate. Biaya tertimbang per 100 meme menghitung dua untuk hate yang terlewat dan satu untuk kesalahan lain. [Desain eksperimen](caption-repair-study.md)

## Validation dengan caption donor sintetis

Ringkasan ini mencakup 804 meme, 161 slot regenerasi per replay, 20 replay, dan tiga seed classifier. Caption yang memenuhi pemeriksaan keluaran tersedia untuk 754 meme. Router dan pemeriksaan penerimaan caption memakai prediksi out-of-fold lima lipatan yang dipisahkan berdasarkan ID meme. [Hasil validation agregat](https://node580-ai-hub.ub.ac.id/user/josephplase77/lab/tree/kelompok-7-private/cp2-caption-repair-study/validation_only_ROUTER_PRELIMINARY.json)

| Model dan kebijakan | Macro-F1 | Biaya tertimbang per 100 meme |
| --- | ---: | ---: |
| B2 mempertahankan caption | 76,33% | 55,23 |
| B2 regenerasi berdasarkan ketidakpastian | 75,05% | 58,08 |
| B2 router terlatih | 75,88% | 55,65 |
| B2 regenerasi seluruhnya | 69,52% | 70,74 |
| S1 mempertahankan caption | 74,88% | 58,39 |
| S1 regenerasi berdasarkan ketidakpastian | 74,32% | 59,32 |
| S1 router terlatih | 74,95% | 57,97 |
| S1 router dan pemeriksaan setelah regenerasi | 74,99% | 57,85 |
| S1 regenerasi seluruhnya | 70,69% | 68,21 |

Pada S1, router mengurangi estimasi biaya 1,35 per 100 meme dibanding ketidakpastian, dengan IK 95% −3,40 hingga +0,61 untuk selisih router dikurangi ketidakpastian. Efek pemeriksaan setelah regenerasi terhadap router saja adalah −0,12 dengan IK −0,61 hingga +0,33. Pada B2, router mengalahkan ketidakpastian pada validation dalam biaya tertimbang (−2,44 dengan IK −4,69 hingga −0,20) dan biaya terbaik tetap diperoleh dengan mempertahankan caption. Semua interval eksploratif dan mengondisikan pada seed dan replay yang ditentukan.
