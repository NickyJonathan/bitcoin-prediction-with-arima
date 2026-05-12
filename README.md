# 📈 Prediksi Harga Bitcoin dengan ARIMA

Aplikasi interaktif berbasis Streamlit untuk memprediksi harga Bitcoin menggunakan model **ARIMA (Autoregressive Integrated Moving Average)**. Mengimplementasikan tahapan **KDD (Knowledge Discovery in Databases)** lengkap dengan **metodologi Box-Jenkins**.

Aplikasi ini merupakan pendukung skripsi:
> **Analisis Prediksi Harga Cryptocurrency Menggunakan Model ARIMA Berbasis Python**
> *Nicky Jonathan Purba*

---

## ✨ Fitur Utama

Aplikasi terdiri atas **6 tab** yang mencerminkan tahapan metodologi penelitian:

| Tab | Fungsi | Selaras dengan Skripsi |
|---|---|---|
| 🏠 **Beranda** | Prediksi cepat untuk pengguna akhir | Output aplikasi |
| 📊 **Data** | Statistik deskriptif, plot historis, distribusi, train/test split | Subbab 3.5.1–3.5.2, 4.1 |
| 🔬 **Stasioneritas** | Uji ADF iteratif, differencing, plot ACF & PACF | Subbab 3.5.3, 4.2 |
| ⚙️ **Pemilihan Model** | Manual + Auto Grid Search berbasis AIC, ringkasan koefisien | Subbab 3.5.4–3.5.5, 4.3 |
| ✅ **Diagnostik** | Uji Ljung-Box (multi-lag), Jarque-Bera, plot residual 4-panel | Subbab 3.5.6, 4.4 |
| 🔮 **Peramalan** | Evaluasi RMSE/MAE/MAPE, naive baseline + Theil's U, forecast + CI 95% | Subbab 3.5.7, 4.5–4.6 |

### Tiga Mode Model

1. **Pre-trained (`arima_model.pkl`)** — muat model yang sudah dilatih sebelumnya
2. **Train Ulang Manual** — latih ARIMA dengan parameter (p, d, q) pilihan Anda
3. **Auto Grid Search (AIC)** — pencarian otomatis kombinasi (p, d, q) terbaik berdasarkan kriteria AIC

---

## 📁 Struktur Project

```
bitcoin-arima-app/
├── app.py                  # Entry point Streamlit + orchestration 6 tabs
├── config.py               # Konstanta global (warna, URL, threshold, dll)
├── data_loader.py          # Pengambilan data dari CoinGecko API
├── preprocessing.py        # Pembagian train/test 80:20
├── stationarity.py         # Uji ADF, differencing, ACF/PACF
├── modeling.py             # Training ARIMA, grid search, pre-trained loader
├── diagnostics.py          # Ljung-Box, Jarque-Bera, residual stats
├── evaluation.py           # MAE/RMSE/MAPE, naive baseline, Theil's U
├── forecasting.py          # Forecast dengan interval kepercayaan
├── viz.py                  # Plotly chart builders (8 jenis chart)
├── arima_model.pkl         # Model pre-trained
├── requirements.txt        # Dependency list
└── README.md               # Dokumentasi (file ini)
```

---

## 🚀 Instalasi & Menjalankan

### Prasyarat
- Python 3.10 atau lebih baru
- Koneksi internet (untuk API CoinGecko)

### Langkah-Langkah

**1. Clone atau extract project ke folder lokal.**

**2. (Opsional tapi disarankan) Buat virtual environment:**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Jalankan aplikasi:**

```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

---

## 🎯 Panduan Penggunaan

### Untuk Pengguna Akhir (Quick Start)
1. Buka tab **🏠 Beranda**
2. Pilih **mode model** di sidebar (default: Pre-trained)
3. Atur **horizon prediksi** (7–30 hari)
4. Lihat hasil prediksi langsung dengan grafik dan tabel

### Untuk Demonstrasi Sidang Skripsi
Tunjukkan tab-tab secara berurutan untuk menjelaskan metodologi:

1. **📊 Data** — Tunjukkan data yang digunakan dan pembagiannya
2. **🔬 Stasioneritas** — Jelaskan kenapa perlu differencing (uji ADF)
3. **⚙️ Pemilihan Model** — Pilih **Auto Grid Search** untuk demo otomatis pencarian model terbaik
4. **✅ Diagnostik** — Tunjukkan bahwa model sudah memadai (Ljung-Box lolos)
5. **🔮 Peramalan** — Tunjukkan akurasi pada test set + perbandingan dengan naive baseline (Theil's U)

---

## 🧪 Detail Teknis

### Sumber Data
- **API**: CoinGecko `/coins/bitcoin/ohlc`
- **Periode**: 365 hari terakhir (rolling)
- **Resolusi**: Harian (Close price USD)
- **Cache**: 15 menit (untuk menghindari rate limit)

### Konfigurasi Model
- **Engine**: `statsmodels.tsa.arima.model.ARIMA`
- **Estimasi**: Maximum Likelihood Estimation (MLE)
- **Optimasi**: `maxiter=200`, `enforce_stationarity=False`, `enforce_invertibility=False`
- **Grid search range**: `p ∈ {0..5}`, `d` otomatis dari ADF, `q ∈ {0..5}` (36 kombinasi)
- **Train/Test split**: 80:20 time-based (TIDAK menggunakan random shuffle)
- **Strategi forecast**: One-step ahead rolling pada test set

### Diagnostik
- **Uji ADF**: orde *differencing* ditentukan iteratif (max d=2)
- **Uji Ljung-Box**: pada lag 5, 10, 15, 20, 30
- **Uji Jarque-Bera**: normalitas residual
- **Plot diagnostik**: residual vs waktu, histogram + kurva normal, Q-Q plot, ACF residual

### Evaluasi
- **Metrik**: MAE, RMSE, MAPE (kategori Lewis 1982)
- **Baseline**: Naive forecast `Ŷ_t = Y_{t-1}`
- **Comparison**: Theil's U statistic (U<1 = model lebih baik dari naive)

---

## 🛠️ Troubleshooting

### "Gagal mengambil data dari CoinGecko"
- Periksa koneksi internet
- Coba klik tombol **🔄 Refresh Data** di sidebar
- Jika rate-limited (HTTP 429), tunggu 1–2 menit lalu coba lagi

### "File model pre-trained tidak ditemukan"
- Pastikan file `arima_model.pkl` ada di folder yang sama dengan `app.py`
- Atau pilih mode **Train Ulang Manual** / **Auto Grid Search** sebagai alternatif

### Grid Search lambat
- Wajar untuk 36 kombinasi (~30-60 detik tergantung CPU)
- Progress bar akan menampilkan kemajuan
- Hasil di-cache di session_state — tidak akan dijalankan ulang sampai data berubah

### Plot tidak tampil sempurna
- Coba refresh browser (Ctrl+R)
- Pastikan menggunakan browser modern (Chrome, Firefox, Edge versi terbaru)

---

## ⚠️ Disclaimer

Aplikasi ini bersifat **edukatif dan analitis**, bukan rekomendasi finansial profesional.

- Akurasi historis tidak menjamin akurasi di masa depan
- Hasil prediksi sebaiknya dipadukan dengan analisis fundamental, sentimen pasar, dan manajemen risiko
- Keputusan investasi tetap menjadi tanggung jawab pengguna

Lihat **Subbab 4.6.3** skripsi untuk pembahasan lengkap tentang implikasi investasi dan manajemen risiko.

---

## 📚 Referensi

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Lewis, C. D. (1982). *Industrial and Business Forecasting Methods*. Butterworth Scientific.
- Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P. (1996). From Data Mining to Knowledge Discovery in Databases. *AI Magazine, 17*(3), 37–54.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

---

## 📝 Lisensi & Kontak

Project ini dikembangkan untuk keperluan skripsi S1. Kode bebas digunakan untuk pembelajaran dan referensi akademik.

**Sumber Data**: [CoinGecko API](https://www.coingecko.com/api)
