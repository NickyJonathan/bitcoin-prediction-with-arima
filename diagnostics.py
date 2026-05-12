"""
diagnostics.py
Diagnostik residual: Ljung-Box, Jarque-Bera, statistik deskriptif.

Sesuai Subbab 2.4.10 (teori) dan 3.5.6 (metodologi) — Tahap Pemeriksaan Diagnostik.
"""
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox

from config import LJUNG_BOX_LAGS


def residual_statistics(residuals: np.ndarray) -> dict:
    """
    Statistik deskriptif residual.
    Subbab 4.4.1.
    """
    r = np.asarray(residuals).flatten()
    r = r[~np.isnan(r)]
    return {
        "Jumlah Residual": int(len(r)),
        "Mean": float(np.mean(r)),
        "Std Deviasi": float(np.std(r, ddof=1)),
        "Minimum": float(np.min(r)),
        "Maximum": float(np.max(r)),
        "Skewness": float(skew(r)),
        "Kurtosis": float(kurtosis(r, fisher=False)),  # Pearson (normal=3)
    }


def ljung_box_multi(residuals: np.ndarray, lags: List[int] = None) -> pd.DataFrame:
    """
    Uji Ljung-Box pada multiple lag.
    Subbab 4.4.2.

    H0: residual tidak berautokorelasi (model memadai)
    H1: residual berautokorelasi

    Returns
    -------
    pd.DataFrame dengan kolom: Lag, Statistik Q, p-value, Kesimpulan
    """
    if lags is None:
        lags = LJUNG_BOX_LAGS

    r = np.asarray(residuals).flatten()
    r = r[~np.isnan(r)]

    rows = []
    for h in lags:
        lb = acorr_ljungbox(r, lags=[h], return_df=True)
        q_stat = float(lb["lb_stat"].iloc[0])
        p_val = float(lb["lb_pvalue"].iloc[0])
        rows.append(
            {
                "Lag": h,
                "Statistik Q": q_stat,
                "p-value": p_val,
                "Kesimpulan (α=0.05)": (
                    "Tidak berautokorelasi ✓" if p_val > 0.05 else "Berautokorelasi ✗"
                ),
            }
        )
    return pd.DataFrame(rows)


def jarque_bera_test(residuals: np.ndarray) -> dict:
    """
    Uji normalitas Jarque-Bera.
    Subbab 4.4.3.

    H0: residual berdistribusi normal
    H1: residual tidak berdistribusi normal
    """
    r = np.asarray(residuals).flatten()
    r = r[~np.isnan(r)]
    stat, p_value = jarque_bera(r)
    return {
        "Statistik JB": float(stat),
        "p-value": float(p_value),
        "Skewness": float(skew(r)),
        "Kurtosis": float(kurtosis(r, fisher=False)),
        "is_normal": p_value > 0.05,
        "conclusion": (
            "NORMAL (H₀ tidak ditolak, α=0.05)"
            if p_value > 0.05
            else "TIDAK NORMAL (H₀ ditolak — residual fat-tailed)"
        ),
    }


def diagnostic_summary(fit_result) -> dict:
    """Ringkasan diagnostik untuk satu fitted model."""
    residuals = np.asarray(fit_result.resid).flatten()
    # Drop initial residuals yang bisa anomali
    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) > 10:
        residuals = residuals[5:]

    return {
        "stats": residual_statistics(residuals),
        "ljung_box": ljung_box_multi(residuals),
        "jarque_bera": jarque_bera_test(residuals),
        "residuals_array": residuals,
    }
