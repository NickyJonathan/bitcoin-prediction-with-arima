"""
stationarity.py
Analisis stasioneritas: uji ADF, differencing, ACF, PACF.

Sesuai Subbab 2.3.4–2.3.6 (teori) dan 3.5.3–3.5.4 (metodologi).
"""
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, pacf

from config import GRID_D_MAX


def adf_test(series: pd.Series, autolag: str = "AIC") -> dict:
    """
    Uji Augmented Dickey-Fuller untuk stasioneritas.

    Hipotesis:
        H0: data memiliki unit root (tidak stasioner)
        H1: data tidak memiliki unit root (stasioner)

    Returns
    -------
    dict dengan keys:
        statistic, p_value, lags, n_obs,
        critical_1pct, critical_5pct, critical_10pct,
        is_stationary, conclusion
    """
    s = series.dropna()
    res = adfuller(s, autolag=autolag)
    return {
        "statistic": float(res[0]),
        "p_value": float(res[1]),
        "lags": int(res[2]),
        "n_obs": int(res[3]),
        "critical_1pct": float(res[4]["1%"]),
        "critical_5pct": float(res[4]["5%"]),
        "critical_10pct": float(res[4]["10%"]),
        "is_stationary": res[1] < 0.05,
        "conclusion": (
            "STASIONER (H₀ ditolak, α=0.05)"
            if res[1] < 0.05
            else "TIDAK STASIONER (H₀ tidak ditolak)"
        ),
    }


def find_optimal_d(series: pd.Series, max_d: int = GRID_D_MAX) -> Tuple[int, dict]:
    """
    Cari orde differencing optimal secara iteratif menggunakan uji ADF.

    Mulai dari d=0 (data asli), lakukan differencing hingga stasioner
    atau hingga mencapai max_d.

    Returns
    -------
    (d_optimal, history) : Tuple[int, dict]
        history = {0: adf_result_d0, 1: adf_result_d1, ...}
    """
    history = {}
    current = series.copy()
    for d in range(max_d + 1):
        if d > 0:
            current = current.diff().dropna()
        result = adf_test(current)
        history[d] = result
        if result["is_stationary"]:
            return d, history
    # Paksa pakai max_d jika belum stasioner
    return max_d, history


def difference(series: pd.Series, d: int = 1) -> pd.Series:
    """Lakukan differencing orde d (d=1 atau d=2)."""
    if d == 0:
        return series.copy()
    out = series.copy()
    for _ in range(d):
        out = out.diff().dropna()
    return out


def compute_acf_pacf(series: pd.Series, n_lags: int = 30) -> dict:
    """
    Hitung ACF dan PACF beserta confidence interval untuk plot.

    Returns
    -------
    dict dengan keys: acf_values, pacf_values, conf_lower, conf_upper, lags
    """
    s = series.dropna()
    n = len(s)
    # Confidence interval Bartlett's formula: ±1.96/sqrt(n)
    ci = 1.96 / np.sqrt(n)

    acf_vals = acf(s, nlags=n_lags, fft=False)
    pacf_vals = pacf(s, nlags=n_lags, method="ywm")
    lags = np.arange(n_lags + 1)

    return {
        "acf_values": acf_vals,
        "pacf_values": pacf_vals,
        "conf_lower": -ci,
        "conf_upper": ci,
        "lags": lags,
    }
