"""
data_loader.py
Modul pengambilan data harga Bitcoin dari API CoinGecko.

Sesuai Subbab 3.5.1 (Pengumpulan Data) — KDD Tahap 1.
"""
import pandas as pd
import requests
import streamlit as st

from config import (
    API_TIMEOUT,
    CACHE_TTL_SECONDS,
    COINGECKO_DEFAULT_DAYS,
    COINGECKO_OHLC_URL,
    MIN_HISTORY_DAYS,
)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_bitcoin_data(days: int = COINGECKO_DEFAULT_DAYS) -> pd.DataFrame:
    """
    Mengambil data harga harian Bitcoin dari endpoint OHLC CoinGecko.

    Parameter
    ---------
    days : int
        Jumlah hari historis yang diminta (default 365).

    Returns
    -------
    pd.DataFrame
        DataFrame berindex tanggal dengan kolom 'close' (harga penutupan USD).

    Raises
    ------
    ValueError
        Jika respons API kosong atau data kurang dari MIN_HISTORY_DAYS.
    requests.HTTPError
        Jika permintaan HTTP gagal.
    """
    params = {"vs_currency": "usd", "days": days}
    response = requests.get(COINGECKO_OHLC_URL, params=params, timeout=API_TIMEOUT)
    response.raise_for_status()
    raw_data = response.json()

    if not raw_data:
        raise ValueError("CoinGecko tidak mengembalikan data.")

    df = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "close"])
    if "close" not in df.columns:
        raise ValueError("Kolom close tidak ditemukan pada respons API.")

    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df = (
        df[["date", "close"]]
        .dropna()
        .drop_duplicates(subset="date")
        .sort_values("date")
        .set_index("date")
    )

    # Resampling harian + forward fill untuk gap
    daily_close = df["close"].astype(float).resample("D").last().ffill()
    result = daily_close.to_frame(name="close")

    if len(result) < MIN_HISTORY_DAYS:
        raise ValueError(
            f"Data historis kurang dari {MIN_HISTORY_DAYS} hari (didapat {len(result)})."
        )

    return result


def get_descriptive_stats(series: pd.Series) -> dict:
    """
    Statistik deskriptif harga (untuk Tab Eksplorasi Data).
    Subbab 4.1.1.
    """
    return {
        "Jumlah Observasi": int(len(series)),
        "Tanggal Awal": str(series.index.min().date()),
        "Tanggal Akhir": str(series.index.max().date()),
        "Harga Minimum (USD)": float(series.min()),
        "Harga Maksimum (USD)": float(series.max()),
        "Harga Rata-rata (USD)": float(series.mean()),
        "Standar Deviasi (USD)": float(series.std()),
        "Harga Awal Periode (USD)": float(series.iloc[0]),
        "Harga Akhir Periode (USD)": float(series.iloc[-1]),
        "Perubahan Total (%)": float(
            ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100
        ),
    }
