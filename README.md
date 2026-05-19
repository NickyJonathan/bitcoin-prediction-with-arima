# Prediksi Harga Bitcoin dengan ARIMA

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-Data-150458?style=flat-square&logo=pandas&logoColor=white">
  <img alt="Statsmodels" src="https://img.shields.io/badge/Statsmodels-ARIMA-1f4e79?style=flat-square">
  <img alt="Plotly" src="https://img.shields.io/badge/Plotly-Chart-3F4F75?style=flat-square&logo=plotly&logoColor=white">
  <img alt="CoinGecko" src="https://img.shields.io/badge/CoinGecko-API-8DC647?style=flat-square&logo=coingecko&logoColor=white">
</p>

Aplikasi Streamlit untuk memprediksi harga Bitcoin menggunakan model ARIMA. Project ini mendukung penelitian:

> Analisis Prediksi Harga Cryptocurrency Menggunakan Model ARIMA Berbasis Python  
> Nicky Jonathan Purba

## Fitur

- Mengambil data harga Bitcoin dari CoinGecko API.
- Membagi data latih dan data uji dengan skema 80:20 berbasis waktu.
- Melakukan uji stasioneritas ADF, differencing, ACF, dan PACF.
- Menjalankan model ARIMA tersimpan, manual, atau pencarian otomatis berdasarkan AIC.
- Menampilkan diagnostik residual dengan Ljung-Box dan Jarque-Bera.
- Menampilkan evaluasi MAE, RMSE, MAPE, Theil's U, serta forecast dengan interval 95%.

## Cara Menjalankan

```powershell
cd "c:\Users\Azrah\Documents\Documents\DevProjects\bitcoin"
python -m streamlit run app.py
```

Buka aplikasi di:

```text
http://localhost:8501
```

Jika dependency belum terpasang:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Mode Model

| Mode | Fungsi |
|---|---|
| Model tersimpan | Memakai `arima_model.pkl`. Paling cepat untuk demo. |
| Latih manual | Melatih ARIMA dengan parameter `(p, d, q)` pilihan user. |
| Pencarian otomatis (AIC) | Mencari kombinasi `(p, d, q)` terbaik berdasarkan AIC. |

## Struktur Project

```text
bitcoin/
|-- app.py                  # Entry point Streamlit
|-- config.py               # Konfigurasi global
|-- data_loader.py          # Ambil data CoinGecko
|-- preprocessing.py        # Train/test split
|-- stationarity.py         # ADF, differencing, ACF, PACF
|-- modeling.py             # Training, grid search, load model
|-- diagnostics.py          # Diagnostik residual
|-- evaluation.py           # MAE, RMSE, MAPE, baseline
|-- forecasting.py          # Forecast dan confidence interval
|-- viz.py                  # Visualisasi Plotly
|-- arima_model.pkl         # Model tersimpan
|-- requirements.txt        # Dependency Python
|-- .streamlit/config.toml  # Theme Streamlit
`-- README.md
```

## Panduan Singkat Interface

- `Beranda`: ringkasan harga terakhir dan prediksi cepat.
- `Data`: statistik, grafik harga, distribusi, dan train/test split.
- `Stasioneritas`: uji ADF, differencing, ACF, dan PACF.
- `Pemilihan Model`: informasi model, AIC/BIC, dan koefisien.
- `Diagnostik`: pemeriksaan residual model.
- `Peramalan`: evaluasi model dan prediksi harga ke depan.

## Catatan

- Aplikasi membutuhkan koneksi internet untuk mengambil data CoinGecko.
- Jika sidebar atau font berubah warna di komputer lain, pastikan folder `.streamlit` ikut terkirim.
- Jika terkena rate limit CoinGecko, tunggu 1 sampai 2 menit lalu jalankan ulang atau klik perbarui data.
- Hasil prediksi bersifat edukatif dan bukan saran investasi.
