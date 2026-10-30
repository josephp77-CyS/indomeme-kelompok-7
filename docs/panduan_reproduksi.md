# Panduan data dan reproduksi

Kelompok 7 — Joseph Pratama Lase dan Rafa Maritza Hanasaputri

## Membaca hasil

Buka notebook utama dari README. Semua 52 sel kode sudah memiliki output dari eksekusi lab. Berkas tersebut dipertahankan sebagai bukti hasil; pembaruan identitas pada Markdown tidak mengubah kode maupun output. Receipt eksekusi asal tersimpan pada `results/audit/provenance/`. Receipt asal merujuk hash sebelum pembaruan identitas, sedangkan manifest pengumpulan mencatat berkas versi repo ini.

`results/hasil_utama.csv` berisi sepuluh hasil tugas pada 1.005 meme test. `results/hasil_baseline.csv` berisi empat belas hasil validation pada 804 meme. Konfigurasi, history, raw dan constrained metrics tersedia pada `results/runs/`. Bukti historis di `results/audit/` merupakan snapshot bertanggal; keterangan lama tentang screening kemiripan dilengkapi audit pHash dan penjelasan terbaru pada notebook/laporan.

## Pemeriksaan tanpa data tambahan

Gunakan Python 3.11 atau lebih baru, lalu dari root repo:

```console
python verify_submission.py
python src/reproduce_core.py --id E059
```

Perintah kedua hanya membaca konfigurasi dan memeriksa hash source; tidak memulai training. Pemeriksa dapat menghitung metrik dari confusion matrix tanpa CSV prediksi individual.

## Data dan lisensi

Data rilis INDOMEME berjumlah 5.023 meme, dengan train resmi 4.018 dan test 1.005. Fit 3.214 dan validation 804 dibentuk dari train resmi menggunakan StratifiedGroupKFold, seed 42, dengan pengelompokan gambar identik. Semua kondisi inti memakai pembagian yang sama.

Sumber: https://huggingface.co/datasets/aiatums/INDOMEME pada revisi `3e0cb8cae9cd726582f69afdb122163d0eab108b`. Materi tambahan penerbit diperiksa silang; atribusi paper, dataset, dan model ada pada laporan. Dataset card menyatakan CC BY 4.0. Label kelas 1 `appropriate` di kode berarti inappropriate.

Gambar, checkpoint, dan CSV individual disimpan di arsip pribadi, bukan disalin ke repo. Unduh **paket analisis lengkap** dari https://drive.google.com/drive/folders/1c7cuGs_ij6ghZxLF21D6GZ1Hk3a8nplH dan ekstrak hingga terdapat folder `data`, `runs`, `results`, `evidence`, serta `src` pada satu root. Gunakan paket analisis kecil, bukan arsip model.

## Menjalankan ulang analisis

Buat environment Python terpisah, kemudian:

```console
python -m pip install -r requirements.txt
python src/run_notebook.py --data-bundle "lokasi/paket_analisis"
```

Notebook tidak melatih model atau mengunduh bobot. Runner memeriksa kecocokan tabel hasil, menyalin input analisis ke workspace baru di `results/analysis_runs/`, lalu menjalankan seluruh sel dengan kernel Jupyter. Jalur asli dalam notebook diselesaikan di workspace ini sehingga bukti eksekusi lama dan paket input tidak ditimpa.

Keluaran baru adalah `notebooks/eksperimen_indomeme.rerun.ipynb` dan HTML pendamping. Receipt dengan host, waktu, hash, dan status aktual berada di workspace analisis. Kegagalan menyimpan notebook parsial serta receipt gagal. Notebook utama beroutput lab tetap tersedia. Bila kernel `python3` belum tersedia pada environment, daftarkan ipykernel dari environment yang sama sebelum menjalankan runner.

## Reproduksi training yang disengaja

Kode training menggunakan PyTorch 2.6.0 CUDA 12.4 dan Transformers 4.46.3 pada lingkungan yang tercatat. `requirements.txt` ditujukan untuk analisis; instalasi PyTorch harus disesuaikan dengan hardware dan CUDA. Config tiap run menyimpan revisi sumber serta model. Jangan menganggap pemasangan ulang telah diuji pada semua sistem.

Setelah GPU, environment, penyimpanan, dan cache dikalibrasi, buat direktori induk `workspaces`, lalu pilih folder run yang belum ada:

```console
python src/reproduce_core.py --id E059 --workspace workspaces/E059_baru --run
```

`--run` memulai unduhan dan training baru. Tanpa flag tersebut, runner hanya memeriksa konfigurasi. Kode ilmiah asal tidak diubah pada migrasi struktur; wrapper saja yang menyesuaikan lokasi config ke `results/runs/`. Batch, pooling, resolusi ViT, validation, dan lambda memiliki deviasi terhadap penulis yang dijelaskan pada laporan. Nilai dari training baru dapat berbeda.

Pada server bersama gunakan hanya folder Kelompok 7, jalankan proses panjang melalui tmux, dan jangan memasang agentic tools. Password akun bersama tidak boleh diubah.
