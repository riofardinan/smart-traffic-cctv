# Pemantau Lalu Lintas Berbasis Video

Aplikasi Streamlit yang menghitung kendaraan (motor, mobil, truk, bus) di setiap jalan masuk simpang dari sebuah video: yang sedang berada di zona, yang masuk melewati garis masuk, dan yang keluar melewati garis keluar. Data seperti ini menjadi dasar pengaturan lampu lalu lintas (ATCS).

Link: https://smart-traffic-cctv.streamlit.app/

## Cara Menjalankan (local)

```bash
cd "Smart Traffic CCTV/code"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tanpa GPU, pasang dulu PyTorch versi CPU supaya unduhannya jauh lebih kecil: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

Model YOLO diunduh otomatis saat video pertama kali diproses.

## Video Contoh

Folder `data/` berisi dua rekaman CCTV ATCS Kota Yogyakarta, kamera Simpang Wirosaban arah utara (`cctvjss.jogjakota.go.id`), masing-masing 1280×720, 25 fps:

| File | Waktu rekam | Isi |
|---|---|---|
| `Simpang_Wirosaban_View_Utara_siang.mp4` | 21 Nov 2026, 12.49 WIB, 60 detik | Antrean motor dan mobil di lampu merah, lalu bergerak saat lampu hijau |
| `Simpang_Wirosaban_View_Utara.mp4` | 2 Nov 2026, 21.41 WIB, 60 detik | Malam hari, dengan sudut kamera yang berbeda dari klip siang |
