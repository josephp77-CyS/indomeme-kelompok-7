# Bekerja pada CP2

Joseph menangani model/fusi dan operasi eksperimen. Rafa menangani input, evaluator, robustness, dan integrasi hasil. Gunakan branch tugas dan commit dengan identitas pelaksana yang sebenarnya, lalu ajukan PR untuk review pasangan.

Mulai dari [status CP2](docs/cp2/mulai-cp2.md) dan spesifikasi CV/NLP. Perubahan kontrak label, input, split, model, loss, dan pemilihan checkpoint harus dicatat sebelum dipakai dalam eksperimen pembanding. Hubungkan hasil ke SHA source yang dijalankan.

Jalankan pemeriksaan dari root checkout dengan environment CP2 aktif.

```bash
export PYTHONPATH="$PWD/src"
python -m pytest -q tests
python verify_submission.py
bash -n scripts/cp2_ai_center_run.sh
bash -n scripts/cp2_mm_run.sh
```

Pada mesin GPU bersama, tes CPU dapat memakai `CUDA_VISIBLE_DEVICES=''`. Smoke dan pilot pretrained berikutnya mengikuti panduan CP2. Cantumkan hardware, scope, seed, config, hasil tes, dan batas pemeriksaan pada PR.

Simpan dataset, prediksi per-ID, checkpoint, dan credential di storage privat di luar checkout. Pilih file yang di-stage secara eksplisit dan periksa diff sebelum push. Notebook analisis publik menampilkan ringkasan agregat. Source dan bukti CP1 yang dikunci tetap diperiksa oleh verifier.

Pakai satu workspace data bersama untuk kunci satu worker CP2. Koordinasikan alokasi GPU melalui mekanisme lab yang berlaku. Setelah review dan merge, sinkronkan checkout bersih sebelum eksperimen selanjutnya. Manifest serta config lama tetap menjadi asal bukti run yang sudah terjadi.
