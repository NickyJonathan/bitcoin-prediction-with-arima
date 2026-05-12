"""
forecasting.py
Peramalan harga masa depan beserta interval kepercayaan 95%.

Sesuai Subbab 2.4.7 (teori) dan 3.5.7 (metodologi) — Tahap Peramalan.
"""
from typing import Tuple

import numpy as np
import pandas as pd

from config import CONFIDENCE_LEVEL


def build_forecast_index(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """Buat index tanggal untuk hasil forecast (dimulai dari hari setelah last_date)."""
    start = last_date + pd.Timedelta(days=1)
    return pd.date_range(start=start, periods=horizon, freq="D")


def forecast_with_ci(
    fit_result, horizon: int, alpha: float = CONFIDENCE_LEVEL
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hasilkan forecast titik beserta interval kepercayaan.

    Parameter
    ---------
    fit_result : Fitted ARIMA model
    horizon : int
        Jumlah hari ke depan yang diramalkan.
    alpha : float
        Tingkat signifikansi (default 0.05 = CI 95%).

    Returns
    -------
    (mean, lower, upper) : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Masing-masing array berlength = horizon.
    """
    if hasattr(fit_result, "get_forecast"):
        fc = fit_result.get_forecast(steps=horizon)
        mean = np.asarray(fc.predicted_mean, dtype=float).flatten()
        ci = fc.conf_int(alpha=alpha)
        # ci adalah DataFrame; ambil dua kolom (lower, upper)
        lower = np.asarray(ci.iloc[:, 0], dtype=float).flatten()
        upper = np.asarray(ci.iloc[:, 1], dtype=float).flatten()
    elif hasattr(fit_result, "forecast"):
        # Fallback untuk model lama (tanpa CI)
        mean = np.asarray(fit_result.forecast(steps=horizon), dtype=float).flatten()
        lower = np.full_like(mean, np.nan)
        upper = np.full_like(mean, np.nan)
    else:
        raise TypeError("Model tidak mendukung metode forecast.")

    if len(mean) != horizon:
        raise ValueError("Jumlah nilai forecast tidak sesuai dengan horizon.")

    return mean, lower, upper


def rolling_one_step_forecast(
    fit_result, test: pd.Series
) -> np.ndarray:
    """
    One-step ahead rolling forecast pada data uji.

    Untuk setiap titik t di test:
    1. Prediksi 1 langkah ke depan
    2. Tambahkan nilai aktual ke state model (tanpa refit)
    3. Lanjut ke t+1

    Sesuai Subbab 3.5.7.
    """
    predictions = []
    current_model = fit_result
    for t in range(len(test)):
        fc = current_model.get_forecast(steps=1)
        yhat = float(fc.predicted_mean.iloc[0])
        predictions.append(yhat)
        actual = test.iloc[t]
        current_model = current_model.append([actual], refit=False)
    return np.asarray(predictions)


def build_forecast_dataframe(
    last_date: pd.Timestamp,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> pd.DataFrame:
    """Susun hasil forecast ke DataFrame untuk ditampilkan/diekspor."""
    idx = build_forecast_index(last_date, len(mean))
    return pd.DataFrame(
        {
            "forecast": mean,
            "lower_95": lower,
            "upper_95": upper,
        },
        index=idx,
    )
