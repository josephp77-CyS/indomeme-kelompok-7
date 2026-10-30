# Sumber penyusunan laporan

`source.md` memuat narasi laporan dan ID persamaan. `formulas.json`, `equations.py`, dan `math_nodes.py` menyimpan rumusan beserta representasi Word OMML serta HTML MathML. `build_writeup.py` menghasilkan laporan Word, HTML mandiri, dan Markdown pada `docs/`.

Untuk menyusun ulang dokumen, instal `requirements.txt` di folder ini pada environment terpisah, lalu jalankan dari root repo:

```console
python src/authoring/build_writeup.py
```

Perintah ini mengganti tiga berkas laporan di `docs/`; simpan perubahan yang ingin dipertahankan melalui Git terlebih dahulu. `build_architecture.py` menggambar ulang diagram arsitektur berdasarkan source dan config yang dipatok. Tiga visual perbandingan lain dipertahankan; notebook memuat analisis dan visualisasi hasil sumbernya.

`render_with_word.ps1` adalah helper QA khusus Windows dengan Microsoft Word. Lokasi Poppler di dalam helper mengikuti runtime laptop penyusun dan perlu disesuaikan jika digunakan pada komputer lain. Helper tersebut tidak diperlukan untuk membaca hasil atau menjalankan analisis.
