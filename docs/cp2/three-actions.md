# Router tiga aksi pada kuota regenerasi caption

Eksperimen sekunder ini memakai tiga checkpoint S1, caption Qwen tersimpan, 804 meme validation, dan 1.005 meme test retrospektif. Dua regresi Ridge menggunakan fitur probabilitas model dengan dan tanpa caption. Satu memprediksi manfaat mengabaikan caption, satu lagi manfaat membuat ulang caption. Targetnya adalah selisih biaya terhadap mempertahankan caption. Biaya menghitung dua untuk hate yang terlewat dan satu untuk setiap kesalahan lain. Training memakai fold berkelompok per meme dengan bobot 80% asli dan 20% donor. Model refit pada seluruh validation digunakan untuk test.

Untuk tiap meme, router memilih aksi murah dari mempertahankan atau mengabaikan caption. Ia mengurutkan kandidat regenerasi berdasarkan manfaat tambahan terhadap aksi murah, lalu memakai manfaat positif hingga mencapai batas jumlah permintaan. Bila regenerasi gagal, aksi murah tetap dipakai. Kebijakan dapat menggunakan kurang dari batas yang diberikan.

Uji utama adalah campuran 20% caption donor dengan kuota maksimum 20%. Pembanding mencakup mempertahankan caption, mengabaikan caption secara selektif, ketidakpastian, router regenerasi yang lama, dan regenerasi dengan manfaat prediksi positif. Uji sekunder memakai caption asli dan kuota 5% serta 10%. Semua kebijakan memakai penempatan donor dan tie-break yang sama dalam 20 replay. Interval 95% memakai 5.000 bootstrap meme berpasangan dengan replay dan seed tetap. Pembanding 20% diverifikasi terhadap hasil tersimpan sebelum interpretasi. [Studi caption sebelumnya](caption-repair-study.md)

Jalankan dengan `PYTHONPATH=src python -m cp2.repair_three_actions --study /path/to/source-study --job /path/to/caption-repair-study`. Hasil agregat ditulis ke `three_actions_results.json`. Kredensial, gambar, caption mentah, dan prediksi per ID berada di workspace privat.

## Hasil test retrospektif

| Kuota maksimum | Permintaan router awal | Biaya router awal per 100 meme | Permintaan router tiga aksi | Biaya tiga aksi per 100 meme |
| ---: | ---: | ---: | ---: | ---: |
| 5% | 51 | 57,12 | 51 | 57,04 |
| 10% | 101 | 56,66 | 101 | 56,81 |
| 20% | 201 | 56,65 | 184,93 | 57,05 |

Tabel memakai skenario 20% caption donor. Pada kuota 20%, tiga aksi mengurangi permintaan rata-rata dari 201 ke 184,93. Biayanya naik 0,40 per 100 meme dibanding router awal (IK 95% −0,54 hingga +1,31). Pada input asli, biaya untuk tiga aksi dan router awal masing-masing 54,63 dan 53,60. Temuan ini mendukung router awal sebagai metode utama.

Untuk router awal, kuota 10% mengurangi permintaan sekitar 50%. Selisih biaya 10% dikurangi 20% adalah **+0,008** per 100 meme (IK 95% **−0,23 hingga +0,26**) pada campuran caption donor. Ini analisis eksploratif berpasangan setelah hasil awal tersedia. Caption dan prediksi sudah dibuat sebelumnya, sehingga jumlah permintaan mengukur beban simulasi captioner. [Agregat terverifikasi](../../results/cp2/three_actions.json)
