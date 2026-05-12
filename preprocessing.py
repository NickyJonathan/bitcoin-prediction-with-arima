"""
preprocessing.py
Pra-pemrosesan data: pembagian train/test sesuai metodologi.

Sesuai Subbab 3.5.2 (Pra-pemrosesan Data) - KDD Tahap 2.
"""
from typing import Tuple

import pandas as pd

from config import TRAIN_TEST_SPLIT_RATIO


def train_test_split_timebased(
    series: pd.Series, ratio: float = TRAIN_TEST_SPLIT_RATIO
) -> Tuple[pd.Series, pd.Series]:
    """
    Pembagian time-based split: data awal sebagai train, data akhir sebagai test.
    TIDAK menggunakan random shuffle agar mempertahankan keterurutan temporal.

    Parameter
    ---------
    series : pd.Series
        Runtun waktu berindex tanggal (urutan ascending).
    ratio : float
        Proporsi data latih (default 0.8 = 80%).

    Returns
    -------
    (train, test) : Tuple[pd.Series, pd.Series]
    """
    if not 0 < ratio < 1:
        raise ValueError("ratio harus di antara 0 dan 1.")

    n = len(series)
    split_idx = int(n * ratio)
    train = series.iloc[:split_idx].copy()
    test = series.iloc[split_idx:].copy()
    return train, test


def get_split_summary(train: pd.Series, test: pd.Series) -> dict:
    """Ringkasan pembagian data untuk ditampilkan di UI."""
    total = len(train) + len(test)
    return {
        "Total Observasi": total,
        "Data Latih": f"{len(train)} ({len(train)/total*100:.1f}%)",
        "Periode Data Latih": f"{train.index.min().date()} sampai {train.index.max().date()}",
        "Data Uji": f"{len(test)} ({len(test)/total*100:.1f}%)",
        "Periode Data Uji": f"{test.index.min().date()} sampai {test.index.max().date()}",
    }
