"""
evaluation.py
Evaluasi akurasi prediksi: MAE, RMSE, MAPE + perbandingan dengan naive baseline.

Sesuai Subbab 2.5 (teori) dan 3.5.7 (metodologi) — Tahap Evaluasi Pola.
"""
import numpy as np
import pandas as pd

from config import MAPE_CATEGORIES


# ============================================================
# Metrik Dasar
# ============================================================
def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (dalam persen).
    Catatan: tidak terdefinisi jika ada nilai aktual = 0.
    """
    a = np.asarray(actual)
    p = np.asarray(predicted)
    if np.any(a == 0):
        raise ValueError("MAPE tidak terdefinisi untuk nilai aktual = 0.")
    return float(np.mean(np.abs((a - p) / a)) * 100)


def categorize_mape(mape_value: float) -> str:
    """
    Kategori akurasi berdasarkan MAPE menurut Lewis (1982).
    Subbab 2.5.5.
    """
    for threshold, label in MAPE_CATEGORIES:
        if mape_value < threshold:
            return label
    return "Tidak Akurat"


# ============================================================
# Evaluasi Komprehensif
# ============================================================
def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Hitung ketiga metrik sekaligus.

    Returns
    -------
    dict dengan keys: mae, rmse, mape, category
    """
    m = mae(actual, predicted)
    r = rmse(actual, predicted)
    p = mape(actual, predicted)
    return {
        "mae": m,
        "rmse": r,
        "mape": p,
        "category": categorize_mape(p),
    }


# ============================================================
# Naive Baseline
# ============================================================
def naive_predictions(
    train: pd.Series, test: pd.Series
) -> np.ndarray:
    """
    Model naive: Y_hat_t = Y_{t-1}.
    Prediksi untuk hari t = nilai aktual hari t-1.

    Untuk hari pertama test, digunakan nilai terakhir dari train.
    """
    return np.concatenate([[train.iloc[-1]], test.values[:-1]])


def theils_u(actual: np.ndarray, model_pred: np.ndarray, naive_pred: np.ndarray) -> float:
    """
    Theil's U statistic untuk perbandingan dengan naive baseline.
    Subbab 4.5.3.

    U < 1: model lebih baik dari naive
    U = 1: setara
    U > 1: model tidak lebih baik dari naive (data mendekati random walk)
    """
    a = np.asarray(actual)
    num = np.sqrt(np.sum((a - np.asarray(model_pred)) ** 2))
    den = np.sqrt(np.sum((a - np.asarray(naive_pred)) ** 2))
    if den == 0:
        return float("inf")
    return float(num / den)


def compare_with_naive(
    actual: np.ndarray,
    model_pred: np.ndarray,
    naive_pred: np.ndarray,
) -> dict:
    """
    Perbandingan komprehensif: ARIMA vs Naive.

    Returns
    -------
    dict berisi metrik untuk kedua model + Theil's U + interpretasi.
    """
    model_eval = evaluate_predictions(actual, model_pred)
    naive_eval = evaluate_predictions(actual, naive_pred)
    u_stat = theils_u(actual, model_pred, naive_pred)

    if u_stat < 1:
        interp = (
            "ARIMA LEBIH BAIK dari naive baseline. "
            "Model berhasil menangkap struktur temporal yang dapat dieksploitasi."
        )
    elif u_stat > 1:
        interp = (
            "ARIMA TIDAK lebih baik dari naive baseline. "
            "Data sangat mendekati perilaku random walk — konsisten dengan Random Walk Hypothesis."
        )
    else:
        interp = "ARIMA SETARA dengan naive baseline."

    return {
        "model": model_eval,
        "naive": naive_eval,
        "theils_u": u_stat,
        "interpretation": interp,
    }
