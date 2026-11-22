# Pemantau Lalu Lintas Berbasis Video

Aplikasi Streamlit yang menghitung kendaraan di setiap jalan masuk simpang dari sebuah video:

- **kendaraan di zona**: berapa motor, mobil, truk, dan bus yang sedang berada di zona jalan masuk, diperbarui setiap frame;
- **masuk**: berapa kendaraan yang melewati garis masuk, di ujung zona yang jauh dari simpang;
- **keluar**: berapa kendaraan yang melewati garis keluar, di garis henti.

Data seperti ini menjadi dasar pengaturan lampu lalu lintas (ATCS). Kamera CCTV publik tidak bisa diakses langsung, jadi aplikasi ini memproses video rekaman terlebih dulu, lalu menampilkan video hasil deteksinya seperti tayangan kamera.

## Kenapa Susunan Ini

Susunan deteksi gerakan OpenCV + YOLO26 @960 + ByteTrack dipilih dari benchmark 195 kombinasi di FastTracker Benchmark (7 sequence uji, dihitung di GPU). Hasilnya: F1 hitung garis 0,923, MAE kendaraan di zona 12,0, dan HOTA 0,496. Catatan lengkap semua benchmark ada di `../LOG_BENCHMARK.md`.

## Cara Menjalankan

```bash
cd "Smart Traffic CCTV/code"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tanpa GPU, pasang dulu PyTorch versi CPU supaya unduhannya jauh lebih kecil: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

Model `yolo26n.pt` (±5 MB) dan `yolo26m.pt` (±43 MB) diunduh otomatis oleh Ultralytics saat video pertama kali diproses.

## Video Contoh

Folder `data/` berisi dua rekaman CCTV ATCS Kota Yogyakarta, kamera Simpang Wirosaban arah utara (`cctvjss.jogjakota.go.id`), masing-masing 1280×720, 25 fps:

| File | Waktu rekam | Isi |
|---|---|---|
| `Simpang_Wirosaban_View_Utara_siang.mp4` | 21 Nov 2026, 12.49 WIB, 60 detik | Antrean motor dan mobil di lampu merah, lalu bergerak saat lampu hijau |
| `Simpang_Wirosaban_View_Utara.mp4` | 2 Nov 2026, 21.41 WIB, 60 detik | Malam hari, dengan sudut kamera yang berbeda dari klip siang |

## Cara Pakai

1. **Upload video** CCTV simpang, misalnya salah satu video di `data/`.
2. **Atur jalan masuk** di gambar frame pertama. Gambar ada di kiri, pengaturannya di kanan:
   - isi nama jalan masuk;
   - pilih **Zona**, lalu klik minimal 3 titik di sekeliling lajur yang menuju simpang;
   - pilih **Garis masuk**, lalu klik 2 titik melintang lajur di ujung zona yang jauh dari simpang;
   - pilih **Garis keluar**, lalu klik 2 titik di garis henti;
   - klik **Simpan jalan masuk**, lalu ulangi untuk jalan masuk lain.

   Klik ketiga pada garis memulai garis baru, jadi garis yang salah cukup digambar ulang. Kedua garis boleh tidak digambar.
3. Klik **Mulai proses**. Pratinjau diperbarui setiap detik video.
4. Lihat **Hasil**:
   - **video hasil deteksi**: setiap kendaraan diberi kotak, ID, dan jenisnya. Panel di atas menampilkan jumlah per jenis di zona serta hitungan masuk dan keluar. Kotak berubah warna setelah kendaraannya melewati garis: biru muda untuk garis masuk, ungu untuk garis keluar;
   - **frame diproses** dan **frame dideteksi**: berapa persen frame yang benar-benar dijalankan detektornya;
   - **tabel per jalan masuk**: rata-rata dan maksimum kendaraan di zona, jumlah motor, mobil, truk, dan bus yang melewati garis keluar, serta jumlah masuk dan keluar.

## Cara Kerja

Setiap frame melewati empat tahap:

| Tahap | Tugas | File |
|---|---|---|
| Deteksi gerakan (OpenCV) | Membandingkan frame sekarang dengan frame saat deteksi terakhir (diperkecil, abu-abu). Detektor hanya dijalankan kalau ada cukup perubahan, atau sudah 15 frame tanpa deteksi | `motion.py` |
| Detektor ringan (YOLO26n @960) | Mendeteksi motor, mobil, truk, dan bus | `detector.py` |
| Detektor berat (YOLO26m @960) | Mengulang deteksi hanya kalau ada hasil detektor ringan dengan confidence 0,25–0,5 dalam 80 px dari garis masuk atau garis keluar | `detector.py` |
| Tracker (ByteTrack) | Memberi ID yang sama untuk kendaraan yang sama dari frame ke frame. Ambangnya diturunkan ke 0,3 karena ambang bawaannya (0,6–0,7) terlalu tinggi untuk skor YOLO | `app.py` |

Hasil tracker lalu dihitung di `counter.py`:

- **kendaraan di zona:** titik tengah-bawah kotaknya ada di dalam zona;
- **masuk** dan **keluar:** titik itu berpindah ke seberang garis masuk atau garis keluar, dari arah mana pun. Setiap kendaraan dihitung sekali per garis.

Di frame yang dilewati deteksi gerakan, posisi kendaraan dari deteksi terakhir dipakai ulang.

## File

| File | Isi |
|---|---|
| `app.py` | Tampilan Streamlit, perulangan pemrosesan video, tracker ByteTrack, dan penulisan video hasil (H.264 lewat `imageio`) |
| `motion.py` | Deteksi gerakan dengan OpenCV |
| `detector.py` | Detektor ringan dan berat, nama jenis kendaraan, dan aturan kapan detektor berat dipanggil |
| `counter.py` | Hitungan zona dan garis, serta gambar zona, garis, kotak kendaraan, dan panel ringkasan |

## Batasan

- **Lambat di CPU.** Input 960 jauh lebih berat. Di laptop uji (i7-13620H, tanpa GPU), klip CCTV diproses ±0,1–0,2× real-time: video 1 menit selesai sekitar 5–10 menit.
  - Penyebab utamanya, detektor berat (YOLO26m @960) butuh ±0,4 detik per panggilan di CPU dan sering terpanggil di simpang ramai, terutama di dekat garis masuk yang jauh dari kamera.
  - Dengan GPU NVIDIA dan PyTorch versi CUDA, Ultralytics otomatis memakai GPU. Di RTX 3050, satu frame hanya ±25 ms.
- Di simpang ramai dan malam hari yang hujan, deteksi gerakan jarang bisa melewati frame.
- Kendaraan kecil, kendaraan jauh, dan motor yang berdempetan masih sering terlewat. Jenis "truk" dari model juga mencakup pikap.
- Benchmark memakai CCTV Korea yang didominasi mobil. Hasilnya belum diuji dengan hitungan manual lalu lintas Indonesia.
