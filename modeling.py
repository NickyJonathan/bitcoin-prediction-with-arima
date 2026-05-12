"""
modeling.py
Pembangunan model ARIMA: training, grid search, dan pemuatan pre-trained.

Sesuai Subbab 3.5.4–3.5.5 (Identifikasi & Estimasi Parameter) — KDD Tahap 3.
"""
import pickle
import warnings
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

try:
    import joblib
except ImportError:
    joblib = None

warnings.filterwarnings("ignore")


# ============================================================
# Training Manual
# ============================================================
def train_arima(series: pd.Series, order: Tuple[int, int, int]):
    """
    Latih model ARIMA dengan orde (p, d, q) tertentu menggunakan MLE.

    Parameter
    ---------
    series : pd.Series
        Runtun waktu (data latih).
    order : Tuple[int, int, int]
        (p, d, q) sebagai orde model.

    Returns
    -------
    Fitted ARIMA model dari statsmodels.
    """
    model = ARIMA(
        series,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    try:
        return model.fit(method_kwargs={"maxiter": 200})
    except TypeError:
        return model.fit()


# ============================================================
# Grid Search
# ============================================================
def grid_search_arima(
    series: pd.Series,
    d: int,
    p_max: int = 5,
    q_max: int = 5,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Pencarian sistematis ARIMA(p, d, q) berdasarkan AIC.

    Parameter
    ---------
    series : pd.Series
        Data latih.
    d : int
        Orde differencing (sudah ditentukan dari uji ADF).
    p_max, q_max : int
        Batas atas p dan q dalam grid.
    progress_callback : callable, optional
        Fungsi yang dipanggil setiap kombinasi diuji,
        signature: callback(current_idx, total).

    Returns
    -------
    pd.DataFrame
        Berisi kolom: p, d, q, order_str, aic, bic, llf, converged.
        Sudah disorting berdasarkan AIC ascending.
    """
    results = []
    total = (p_max + 1) * (q_max + 1)
    idx = 0
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            idx += 1
            if progress_callback:
                progress_callback(idx, total)
            try:
                model = ARIMA(
                    series,
                    order=(p, d, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(method_kwargs={"maxiter": 120})
                results.append(
                    {
                        "p": p,
                        "d": d,
                        "q": q,
                        "order_str": f"({p},{d},{q})",
                        "aic": float(fit.aic),
                        "bic": float(fit.bic),
                        "llf": float(fit.llf),
                        "converged": True,
                    }
                )
            except Exception:
                # Model tidak konvergen — diabaikan dari ranking
                continue

    if not results:
        raise RuntimeError("Tidak ada model yang berhasil dilatih pada grid search.")

    df = pd.DataFrame(results).sort_values("aic").reset_index(drop=True)
    return df


# ============================================================
# Pre-trained Loader
# ============================================================
def load_pretrained_model(model_path: Path):
    """
    Muat model pre-trained dari file pickle/joblib.
    Mencoba pickle dulu, lalu joblib jika gagal.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

    pickle_error = None
    try:
        with model_path.open("rb") as file:
            return pickle.load(file)
    except Exception as err:
        pickle_error = err

    if joblib is not None:
        try:
            return joblib.load(model_path)
        except Exception as joblib_error:
            raise RuntimeError(
                f"Gagal memuat model. pickle error: {pickle_error}; "
                f"joblib error: {joblib_error}"
            ) from joblib_error

    raise RuntimeError(
        f"Gagal memuat model dengan pickle: {pickle_error}"
    ) from pickle_error


@st.cache_resource(show_spinner=False)
def load_pretrained_model_cached(model_path_str: str):
    """Versi cached agar tidak reload setiap rerun Streamlit."""
    return load_pretrained_model(Path(model_path_str))


def apply_pretrained_to_series(model_result, series: pd.Series):
    """
    Selaraskan state model pre-trained dengan data terbaru tanpa refit.
    Diperlukan agar forecast menggunakan informasi terkini.
    """
    if hasattr(model_result, "model") and getattr(model_result.model, "k_exog", 0):
        raise ValueError(
            "Model pre-trained membutuhkan variabel exogenous, "
            "sedangkan aplikasi ini menggunakan data univariat."
        )

    if hasattr(model_result, "apply"):
        try:
            return model_result.apply(series, refit=False)
        except Exception as err:
            raise ValueError(
                f"Model pre-trained tidak kompatibel dengan data terbaru: {err}"
            ) from err

    return model_result


# ============================================================
# Util Format Summary
# ============================================================
def get_model_info(fit_result) -> dict:
    """Ringkasan model untuk ditampilkan di UI."""
    order = fit_result.model.order
    info = {
        "Order (p, d, q)": f"{order}",
        "AIC": f"{fit_result.aic:.4f}",
        "BIC": f"{fit_result.bic:.4f}",
        "Log-Likelihood": f"{fit_result.llf:.4f}",
        "Jumlah Observasi": int(fit_result.nobs),
    }
    return info


def get_coefficients_df(fit_result) -> pd.DataFrame:
    """Tabel koefisien dengan z-stat, p-value, dan CI."""
    summary = fit_result.summary().tables[1]
    # Tabel summary index pertama biasanya berisi header — parse manual
    rows = []
    for row in summary.data[1:]:
        if len(row) >= 6:
            rows.append(
                {
                    "Parameter": str(row[0]).strip(),
                    "Koefisien": str(row[1]).strip(),
                    "Std Error": str(row[2]).strip(),
                    "z-statistik": str(row[3]).strip(),
                    "P>|z|": str(row[4]).strip(),
                    "CI Bawah 95%": str(row[5]).strip(),
                    "CI Atas 95%": str(row[6]).strip() if len(row) > 6 else "",
                }
            )
    return pd.DataFrame(rows)
