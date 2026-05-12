"""
viz.py
Pembangun grafik Plotly untuk berbagai komponen aplikasi.

Mendukung tahap Knowledge Representation (Subbab 3.5.8) dari KDD.
"""
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import (
    COLOR_ACTUAL,
    COLOR_CI_FILL,
    COLOR_DANGER,
    COLOR_GRID,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    PLOT_TEMPLATE,
)


# ============================================================
# Tab Data
# ============================================================
def plot_historical(df: pd.DataFrame, title: str = "Data Historis Harga Bitcoin") -> go.Figure:
    """Plot harga historis."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"] if "close" in df.columns else df.iloc[:, 0],
            mode="lines",
            name="Harga Aktual (Close)",
            line={"color": COLOR_PRIMARY, "width": 2.4},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        template=PLOT_TEMPLATE,
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        hovermode="x unified",
    )
    return fig


def plot_train_test_split(train: pd.Series, test: pd.Series) -> go.Figure:
    """Plot pembagian train/test."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train.index, y=train.values,
            mode="lines", name="Data Latih (80%)",
            line={"color": COLOR_PRIMARY, "width": 2.2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test.index, y=test.values,
            mode="lines", name="Data Uji (20%)",
            line={"color": COLOR_SECONDARY, "width": 2.2},
        )
    )
    fig.add_vline(
        x=test.index[0],
        line_width=1.2, line_dash="dash", line_color=COLOR_GRID,
    )
    fig.update_layout(
        title="Pembagian Train/Test 80:20 (Time-based Split)",
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        template=PLOT_TEMPLATE,
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        hovermode="x unified",
    )
    return fig


def plot_distribution(series: pd.Series, nbins: int = 30) -> go.Figure:
    """Histogram distribusi harga."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=series.values,
            nbinsx=nbins,
            marker={"color": COLOR_PRIMARY, "line": {"color": "white", "width": 1}},
            name="Distribusi Harga",
        )
    )
    fig.update_layout(
        title="Distribusi Harga Penutupan Bitcoin",
        xaxis_title="Harga (USD)",
        yaxis_title="Frekuensi",
        template=PLOT_TEMPLATE,
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        showlegend=False,
    )
    return fig


# ============================================================
# Tab Stasioneritas
# ============================================================
def plot_differencing(original: pd.Series, differenced: pd.Series, d: int) -> go.Figure:
    """Plot data sebelum & sesudah differencing (subplot)."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Data Asli (d=0)", f"Setelah Differencing (d={d})"),
    )
    fig.add_trace(
        go.Scatter(
            x=original.index, y=original.values,
            mode="lines", name="Data Asli",
            line={"color": COLOR_PRIMARY, "width": 1.8},
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=differenced.index, y=differenced.values,
            mode="lines", name="Differenced",
            line={"color": COLOR_DANGER, "width": 1.2},
        ),
        row=1, col=2,
    )
    fig.add_hline(y=0, line_width=0.5, line_color="black", row=1, col=2)
    fig.update_layout(
        title="Perbandingan Data Sebelum dan Sesudah Differencing",
        template=PLOT_TEMPLATE,
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
        height=400,
    )
    fig.update_xaxes(title_text="Tanggal", row=1, col=1)
    fig.update_xaxes(title_text="Tanggal", row=1, col=2)
    fig.update_yaxes(title_text="Harga (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Δ Harga (USD)", row=1, col=2)
    return fig


def plot_acf_pacf(acf_data: dict) -> go.Figure:
    """Plot ACF dan PACF berdampingan dengan CI."""
    from plotly.subplots import make_subplots

    lags = acf_data["lags"]
    ci_low = acf_data["conf_lower"]
    ci_up = acf_data["conf_upper"]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))

    for col_idx, (key, color) in enumerate(
        [("acf_values", COLOR_PRIMARY), ("pacf_values", COLOR_DANGER)], start=1
    ):
        values = acf_data[key]
        # Bar chart (stem-like)
        fig.add_trace(
            go.Bar(
                x=lags, y=values,
                marker={"color": color},
                width=0.15,
                name=key.upper().replace("_VALUES", ""),
            ),
            row=1, col=col_idx,
        )
        # Confidence interval
        fig.add_hline(y=ci_low, line_dash="dash", line_color=COLOR_GRID, row=1, col=col_idx)
        fig.add_hline(y=ci_up, line_dash="dash", line_color=COLOR_GRID, row=1, col=col_idx)
        fig.add_hline(y=0, line_width=0.6, line_color="black", row=1, col=col_idx)

    fig.update_layout(
        title="Autocorrelation Function (ACF) dan Partial Autocorrelation Function (PACF)",
        template=PLOT_TEMPLATE,
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
        height=400,
    )
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="Korelasi", row=1, col=1)
    fig.update_yaxes(title_text="Korelasi Parsial", row=1, col=2)
    return fig


# ============================================================
# Tab Diagnostik
# ============================================================
def plot_residual_diagnostics(residuals: np.ndarray, dates: pd.DatetimeIndex = None) -> go.Figure:
    """
    4-panel residual diagnostic:
    (1) Residual vs waktu
    (2) Histogram + kurva normal
    (3) Q-Q plot
    (4) ACF residual
    """
    from plotly.subplots import make_subplots
    from scipy import stats

    r = np.asarray(residuals).flatten()
    r = r[~np.isnan(r)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Residual terhadap Waktu",
            "Histogram Residual (vs Normal)",
            "Q-Q Plot (Normalitas)",
            "ACF Residual",
        ),
    )

    # (1) Residual vs waktu
    x_idx = list(range(len(r))) if dates is None else dates[: len(r)]
    fig.add_trace(
        go.Scatter(x=x_idx, y=r, mode="lines", line={"color": COLOR_PRIMARY, "width": 1.2}),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_width=0.5, line_color="black", row=1, col=1)

    # (2) Histogram + kurva normal
    fig.add_trace(
        go.Histogram(
            x=r, nbinsx=30, histnorm="probability density",
            marker={"color": COLOR_PRIMARY, "opacity": 0.6},
        ),
        row=1, col=2,
    )
    mu, sigma = np.mean(r), np.std(r, ddof=1)
    x_range = np.linspace(r.min(), r.max(), 200)
    norm_pdf = stats.norm.pdf(x_range, mu, sigma)
    fig.add_trace(
        go.Scatter(x=x_range, y=norm_pdf, mode="lines",
                   line={"color": COLOR_DANGER, "width": 2}, name="Normal"),
        row=1, col=2,
    )

    # (3) Q-Q plot
    qq = stats.probplot(r, dist="norm", fit=False)
    theoretical_q, sample_q = qq[0], qq[1]
    fig.add_trace(
        go.Scatter(x=theoretical_q, y=sample_q, mode="markers",
                   marker={"color": COLOR_PRIMARY, "size": 5}),
        row=2, col=1,
    )
    # Reference line (y=x scaled)
    ref_x = np.array([theoretical_q.min(), theoretical_q.max()])
    fig.add_trace(
        go.Scatter(x=ref_x, y=ref_x * sigma + mu, mode="lines",
                   line={"color": COLOR_DANGER, "width": 1.5, "dash": "dash"}),
        row=2, col=1,
    )

    # (4) ACF residual
    from statsmodels.tsa.stattools import acf
    n_lag = min(30, len(r) // 4)
    acf_vals = acf(r, nlags=n_lag, fft=False)
    lags = np.arange(n_lag + 1)
    ci = 1.96 / np.sqrt(len(r))
    fig.add_trace(
        go.Bar(x=lags, y=acf_vals, marker={"color": COLOR_PRIMARY}, width=0.15),
        row=2, col=2,
    )
    fig.add_hline(y=ci, line_dash="dash", line_color=COLOR_GRID, row=2, col=2)
    fig.add_hline(y=-ci, line_dash="dash", line_color=COLOR_GRID, row=2, col=2)
    fig.add_hline(y=0, line_width=0.5, line_color="black", row=2, col=2)

    fig.update_layout(
        title="Diagnostik Residual Model",
        template=PLOT_TEMPLATE,
        showlegend=False,
        height=700,
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    return fig


# ============================================================
# Tab Peramalan
# ============================================================
def plot_actual_vs_predicted(
    train_recent: pd.Series,
    test: pd.Series,
    predictions: np.ndarray,
    mape_value: Optional[float] = None,
) -> go.Figure:
    """Plot aktual vs prediksi pada data uji."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_recent.index, y=train_recent.values,
            mode="lines", name="Data Latih (akhir)",
            line={"color": COLOR_PRIMARY, "width": 1.8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test.index, y=test.values,
            mode="lines", name="Data Uji (Aktual)",
            line={"color": COLOR_ACTUAL, "width": 2.2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test.index, y=predictions,
            mode="lines", name="Prediksi ARIMA",
            line={"color": COLOR_SECONDARY, "width": 2.2, "dash": "dash"},
        )
    )
    fig.add_vline(
        x=test.index[0],
        line_width=1.2, line_dash="dot", line_color=COLOR_GRID,
    )
    title = "Perbandingan Aktual vs Prediksi pada Data Uji"
    if mape_value is not None:
        title += f" (MAPE = {mape_value:.2f}%)"
    fig.update_layout(
        title=title,
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        template=PLOT_TEMPLATE,
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        hovermode="x unified",
    )
    return fig


def plot_forecast_with_ci(
    historical: pd.Series,
    forecast_df: pd.DataFrame,
    n_recent: int = 90,
) -> go.Figure:
    """Plot forecast masa depan dengan shaded CI 95%."""
    recent = historical.iloc[-n_recent:]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent.index, y=recent.values,
            mode="lines", name=f"Data Aktual ({n_recent} hari terakhir)",
            line={"color": COLOR_PRIMARY, "width": 2.2},
        )
    )
    # CI sebagai filled area
    if "upper_95" in forecast_df.columns and not forecast_df["upper_95"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index, y=forecast_df["upper_95"],
                mode="lines", line={"width": 0}, showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index, y=forecast_df["lower_95"],
                mode="lines", line={"width": 0},
                fill="tonexty", fillcolor=COLOR_CI_FILL,
                name="Interval Kepercayaan 95%",
                hoverinfo="skip",
            )
        )
    # Forecast mean
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index, y=forecast_df["forecast"],
            mode="lines", name="Prediksi ARIMA",
            line={"color": COLOR_SECONDARY, "width": 2.4, "dash": "dash"},
        )
    )
    fig.add_vline(
        x=historical.index[-1],
        line_width=1.2, line_dash="dot", line_color=COLOR_GRID,
    )
    fig.update_layout(
        title=f"Peramalan Harga Bitcoin {len(forecast_df)} Hari ke Depan",
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        template=PLOT_TEMPLATE,
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        hovermode="x unified",
    )
    return fig


def plot_naive_comparison(
    test: pd.Series,
    arima_pred: np.ndarray,
    naive_pred: np.ndarray,
) -> go.Figure:
    """Plot perbandingan ARIMA vs Naive baseline."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=test.index, y=test.values,
            mode="lines", name="Aktual",
            line={"color": COLOR_ACTUAL, "width": 2.2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test.index, y=arima_pred,
            mode="lines", name="ARIMA",
            line={"color": COLOR_SECONDARY, "width": 2, "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test.index, y=naive_pred,
            mode="lines", name="Naive (Yₜ = Yₜ₋₁)",
            line={"color": COLOR_DANGER, "width": 1.6, "dash": "dot"},
        )
    )
    fig.update_layout(
        title="Perbandingan: ARIMA vs Naive Baseline",
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        template=PLOT_TEMPLATE,
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        hovermode="x unified",
    )
    return fig
