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


FONT_FAMILY = "Segoe UI, Inter, Calibri, sans-serif"
COLOR_TEXT = "#111827"
COLOR_MUTED = "#64748b"
COLOR_AXIS = "#334155"
COLOR_BORDER = "#e5e7eb"
COLOR_ZERO = "#94a3b8"
CHART_BG = "#ffffff"
COLOR_FORECAST_ZONE = "rgba(255, 247, 237, 0.72)"
COLOR_CI_EDGE = "rgba(217, 119, 6, 0.34)"
COLOR_REFERENCE = "rgba(100, 116, 139, 0.55)"


def _style_figure(
    fig: go.Figure,
    title: str,
    x_title: str = "Tanggal",
    y_title: str = "",
    height: int = 430,
    showlegend: bool = True,
    price_y: bool = False,
) -> go.Figure:
    """Apply one consistent visual system to all Plotly charts."""
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title={
            "text": title,
            "x": 0.0,
            "xanchor": "left",
            "font": {"family": FONT_FAMILY, "size": 17, "color": COLOR_TEXT},
        },
        font={"family": FONT_FAMILY, "size": 12, "color": COLOR_AXIS},
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        height=height,
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": COLOR_BORDER,
            "font": {"family": FONT_FAMILY, "size": 12, "color": COLOR_TEXT},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"family": FONT_FAMILY, "size": 11, "color": COLOR_AXIS},
            "itemsizing": "constant",
        },
        margin={"l": 56, "r": 24, "t": 88 if showlegend else 70, "b": 46},
        showlegend=showlegend,
    )
    fig.update_xaxes(
        title_text=x_title,
        showgrid=False,
        zeroline=False,
        linecolor=COLOR_BORDER,
        tickfont={"color": COLOR_MUTED},
        title_font={"color": COLOR_AXIS, "size": 12},
        ticks="outside",
        tickcolor=COLOR_BORDER,
    )
    fig.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor=COLOR_GRID,
        griddash="solid",
        zeroline=False,
        linecolor=COLOR_BORDER,
        tickfont={"color": COLOR_MUTED},
        title_font={"color": COLOR_AXIS, "size": 12},
        ticks="outside",
        tickcolor=COLOR_BORDER,
    )
    if price_y:
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def _price_hover(label: str) -> str:
    return f"{label}: $%{{y:,.2f}}<extra></extra>"


def _value_hover(label: str) -> str:
    return f"{label}: %{{y:,.4f}}<extra></extra>"


def _fmt_usd(value: float) -> str:
    return f"${value:,.0f}"


def _padded_price_range(values: list[float]) -> list[float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[~np.isnan(clean)]
    if len(clean) == 0:
        return [0, 1]
    low = float(np.min(clean))
    high = float(np.max(clean))
    pad = max((high - low) * 0.14, abs(high) * 0.015, 1)
    return [low - pad, high + pad]


# ============================================================
# Tab Data
# ============================================================
def plot_historical(df: pd.DataFrame, title: str = "Riwayat Harga Bitcoin") -> go.Figure:
    """Plot harga historis."""
    series = df["close"] if "close" in df.columns else df.iloc[:, 0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name="Harga aktual",
            line={"color": COLOR_PRIMARY, "width": 2.6},
            hovertemplate=_price_hover("Harga aktual"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[series.index[-1]],
            y=[series.iloc[-1]],
            mode="markers",
            name="Harga terakhir",
            marker={
                "color": COLOR_PRIMARY,
                "size": 8,
                "line": {"color": "#ffffff", "width": 2},
            },
            showlegend=False,
            hovertemplate=_price_hover("Harga terakhir"),
        )
    )
    return _style_figure(fig, title, y_title="Harga Bitcoin (USD)", price_y=True)


def plot_train_test_split(train: pd.Series, test: pd.Series) -> go.Figure:
    """Plot pembagian train/test."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train.index,
            y=train.values,
            mode="lines",
            name="Data untuk melatih model (80%)",
            line={"color": COLOR_PRIMARY, "width": 2.4},
            hovertemplate=_price_hover("Data latih"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test.values,
            mode="lines",
            name="Data untuk menguji model (20%)",
            line={"color": COLOR_SECONDARY, "width": 2.4},
            hovertemplate=_price_hover("Data uji"),
        )
    )
    fig.add_vline(
        x=test.index[0],
        line_width=1.3,
        line_dash="dash",
        line_color=COLOR_ZERO,
    )
    fig.add_annotation(
        x=test.index[0],
        y=1.0,
        xref="x",
        yref="paper",
        text="Awal data uji",
        showarrow=False,
        yshift=14,
        font={"size": 11, "color": COLOR_MUTED},
    )
    return _style_figure(
        fig,
        "Pembagian Data Latih dan Uji 80:20",
        y_title="Harga Bitcoin (USD)",
        price_y=True,
    )


def plot_distribution(series: pd.Series, nbins: int = 30) -> go.Figure:
    """Histogram distribusi harga."""
    mean_price = float(series.mean())

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=series.values,
            nbinsx=nbins,
            marker={
                "color": COLOR_PRIMARY,
                "opacity": 0.82,
                "line": {"color": "#ffffff", "width": 1},
            },
            name="Harga penutupan",
            hovertemplate="Rentang harga: $%{x:,.2f}<br>Frekuensi: %{y}<extra></extra>",
        )
    )
    fig.add_vline(
        x=mean_price,
        line_width=1.4,
        line_dash="dash",
        line_color=COLOR_SECONDARY,
    )
    fig.add_annotation(
        x=mean_price,
        y=1.0,
        xref="x",
        yref="paper",
        text="Rata-rata",
        showarrow=False,
        yshift=12,
        font={"size": 11, "color": COLOR_MUTED},
    )
    fig.update_layout(bargap=0.08)
    fig = _style_figure(
        fig,
        "Distribusi Harga Penutupan Bitcoin",
        x_title="Harga Bitcoin (USD)",
        y_title="Frekuensi",
        showlegend=False,
        price_y=False,
    )
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    return fig


# ============================================================
# Tab Stasioneritas
# ============================================================
def plot_differencing(original: pd.Series, differenced: pd.Series, d: int) -> go.Figure:
    """Plot data sebelum & sesudah differencing (subplot)."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.09,
        subplot_titles=("Data awal (d=0)", f"Setelah differencing (d={d})"),
    )
    fig.add_trace(
        go.Scatter(
            x=original.index,
            y=original.values,
            mode="lines",
            name="Data awal",
            line={"color": COLOR_PRIMARY, "width": 2.1},
            hovertemplate=_price_hover("Data awal"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=differenced.index,
            y=differenced.values,
            mode="lines",
            name="Data differencing",
            line={"color": COLOR_DANGER, "width": 1.8},
            hovertemplate=_price_hover("Perubahan"),
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line_width=1, line_color=COLOR_ZERO, row=1, col=2)

    fig = _style_figure(
        fig,
        "Perbandingan Data Sebelum dan Sesudah Differencing",
        y_title="",
        height=430,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Tanggal", row=1, col=1)
    fig.update_xaxes(title_text="Tanggal", row=1, col=2)
    fig.update_yaxes(title_text="Harga Bitcoin (USD)", tickprefix="$", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="Perubahan harga (USD)", tickprefix="$", tickformat=",.0f", row=1, col=2)
    return fig


def plot_acf_pacf(acf_data: dict) -> go.Figure:
    """Plot ACF dan PACF berdampingan dengan CI."""
    from plotly.subplots import make_subplots

    lags = acf_data["lags"]
    ci_low = acf_data["conf_lower"]
    ci_up = acf_data["conf_upper"]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.09,
        subplot_titles=("ACF", "PACF"),
    )

    for col_idx, (key, color, label) in enumerate(
        [
            ("acf_values", COLOR_PRIMARY, "ACF"),
            ("pacf_values", COLOR_DANGER, "PACF"),
        ],
        start=1,
    ):
        values = acf_data[key]
        fig.add_hrect(
            y0=ci_low,
            y1=ci_up,
            fillcolor="#eef2f7",
            opacity=0.75,
            line_width=0,
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Bar(
                x=lags,
                y=values,
                marker={"color": color, "line": {"color": "#ffffff", "width": 0.5}},
                width=0.28,
                name=label,
                hovertemplate=_value_hover(label),
            ),
            row=1,
            col=col_idx,
        )
        fig.add_hline(y=ci_low, line_dash="dash", line_color=COLOR_ZERO, row=1, col=col_idx)
        fig.add_hline(y=ci_up, line_dash="dash", line_color=COLOR_ZERO, row=1, col=col_idx)
        fig.add_hline(y=0, line_width=1, line_color=COLOR_ZERO, row=1, col=col_idx)

    fig = _style_figure(
        fig,
        "Autocorrelation Function (ACF) dan Partial Autocorrelation Function (PACF)",
        x_title="Lag",
        y_title="Korelasi",
        height=430,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="Korelasi", range=[-1, 1], row=1, col=1)
    fig.update_yaxes(title_text="Korelasi parsial", range=[-1, 1], row=1, col=2)
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
    from statsmodels.tsa.stattools import acf

    r = np.asarray(residuals).flatten()
    r = r[~np.isnan(r)]

    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.09,
        vertical_spacing=0.13,
        subplot_titles=(
            "Residual terhadap waktu",
            "Histogram residual dan kurva normal",
            "Q-Q plot normalitas",
            "ACF residual",
        ),
    )

    x_idx = list(range(len(r))) if dates is None else dates[: len(r)]
    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=r,
            mode="lines",
            line={"color": COLOR_PRIMARY, "width": 1.5},
            hovertemplate=_value_hover("Residual"),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_width=1, line_color=COLOR_ZERO, row=1, col=1)

    fig.add_trace(
        go.Histogram(
            x=r,
            nbinsx=30,
            histnorm="probability density",
            marker={
                "color": COLOR_PRIMARY,
                "opacity": 0.66,
                "line": {"color": "#ffffff", "width": 1},
            },
            name="Residual",
            hovertemplate="Residual: %{x:,.4f}<br>Densitas: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    mu, sigma = np.mean(r), np.std(r, ddof=1)
    x_range = np.linspace(r.min(), r.max(), 200)
    norm_pdf = stats.norm.pdf(x_range, mu, sigma)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=norm_pdf,
            mode="lines",
            line={"color": COLOR_DANGER, "width": 2},
            name="Kurva normal",
            hovertemplate="Normal: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    qq = stats.probplot(r, dist="norm", fit=False)
    theoretical_q, sample_q = qq[0], qq[1]
    fig.add_trace(
        go.Scatter(
            x=theoretical_q,
            y=sample_q,
            mode="markers",
            marker={
                "color": COLOR_PRIMARY,
                "size": 5,
                "opacity": 0.76,
                "line": {"color": "#ffffff", "width": 0.5},
            },
            hovertemplate="Teoretis: %{x:.4f}<br>Sampel: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    ref_x = np.array([theoretical_q.min(), theoretical_q.max()])
    fig.add_trace(
        go.Scatter(
            x=ref_x,
            y=ref_x * sigma + mu,
            mode="lines",
            line={"color": COLOR_DANGER, "width": 1.6, "dash": "dash"},
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    n_lag = min(30, len(r) // 4)
    acf_vals = acf(r, nlags=n_lag, fft=False)
    lags = np.arange(n_lag + 1)
    ci = 1.96 / np.sqrt(len(r))
    fig.add_hrect(
        y0=-ci,
        y1=ci,
        fillcolor="#eef2f7",
        opacity=0.75,
        line_width=0,
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=lags,
            y=acf_vals,
            marker={"color": COLOR_PRIMARY, "line": {"color": "#ffffff", "width": 0.5}},
            width=0.28,
            hovertemplate="Lag %{x}<br>ACF: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_hline(y=ci, line_dash="dash", line_color=COLOR_ZERO, row=2, col=2)
    fig.add_hline(y=-ci, line_dash="dash", line_color=COLOR_ZERO, row=2, col=2)
    fig.add_hline(y=0, line_width=1, line_color=COLOR_ZERO, row=2, col=2)

    fig = _style_figure(
        fig,
        "Diagnostik Residual Model",
        height=720,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Observasi", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=1, col=1)
    fig.update_xaxes(title_text="Residual", row=1, col=2)
    fig.update_yaxes(title_text="Densitas", row=1, col=2)
    fig.update_xaxes(title_text="Kuantil teoretis", row=2, col=1)
    fig.update_yaxes(title_text="Kuantil sampel", row=2, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=2)
    fig.update_yaxes(title_text="ACF", row=2, col=2)
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
    fig.add_vrect(
        x0=test.index[0],
        x1=test.index[-1],
        fillcolor="rgba(240, 253, 250, 0.62)",
        opacity=1,
        layer="below",
        line_width=0,
    )
    fig.add_trace(
        go.Scatter(
            x=train_recent.index,
            y=train_recent.values,
            mode="lines",
            name="Data latih terbaru",
            line={"color": COLOR_MUTED, "width": 1.8},
            hovertemplate=_price_hover("Data latih"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test.values,
            mode="lines",
            name="Aktual",
            line={"color": COLOR_ACTUAL, "width": 2.6},
            hovertemplate=_price_hover("Aktual"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=predictions,
            mode="lines+markers",
            name="Prediksi model",
            line={"color": COLOR_SECONDARY, "width": 2.6, "dash": "dash"},
            marker={
                "color": COLOR_SECONDARY,
                "size": 4.5,
                "line": {"color": "#ffffff", "width": 0.8},
            },
            hovertemplate=_price_hover("Prediksi model"),
        )
    )
    fig.add_vline(
        x=test.index[0],
        line_width=1.3,
        line_dash="dot",
        line_color=COLOR_ZERO,
    )
    fig.add_annotation(
        x=test.index[0],
        y=1.0,
        xref="x",
        yref="paper",
        text="Mulai data uji",
        showarrow=False,
        yshift=18,
        bgcolor="#ffffff",
        bordercolor=COLOR_BORDER,
        borderwidth=1,
        borderpad=4,
        font={"size": 11, "color": COLOR_TEXT},
    )
    fig.add_annotation(
        x=test.index[-1],
        y=float(test.iloc[-1]),
        text=f"Aktual<br><b>{_fmt_usd(float(test.iloc[-1]))}</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLOR_ACTUAL,
        ax=42,
        ay=-34,
        bgcolor="#ffffff",
        bordercolor=COLOR_BORDER,
        borderwidth=1,
        borderpad=5,
        font={"size": 11, "color": COLOR_TEXT},
    )
    fig.add_annotation(
        x=test.index[-1],
        y=float(predictions[-1]),
        text=f"Prediksi<br><b>{_fmt_usd(float(predictions[-1]))}</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLOR_SECONDARY,
        ax=42,
        ay=42,
        bgcolor="#ffffff",
        bordercolor=COLOR_BORDER,
        borderwidth=1,
        borderpad=5,
        font={"size": 11, "color": COLOR_TEXT},
    )
    title = "Harga Aktual vs Prediksi pada Data Uji"
    if mape_value is not None:
        title += f" (MAPE = {mape_value:.2f}%)"
    all_y = train_recent.tolist() + test.tolist() + list(np.asarray(predictions, dtype=float))
    fig = _style_figure(fig, title, y_title="Harga Bitcoin (USD)", price_y=True, height=470)
    fig.update_yaxes(range=_padded_price_range(all_y))
    fig.update_xaxes(range=[train_recent.index[0], test.index[-1] + pd.Timedelta(days=2)])
    fig.update_layout(margin={"l": 56, "r": 112, "t": 92, "b": 46})
    return fig


def plot_forecast_with_ci(
    historical: pd.Series,
    forecast_df: pd.DataFrame,
    n_recent: int = 90,
) -> go.Figure:
    """Plot forecast masa depan dengan shaded CI 95%."""
    recent = historical.iloc[-n_recent:]
    last_date = historical.index[-1]
    last_price = float(historical.iloc[-1])
    end_date = forecast_df.index[-1]
    end_price = float(forecast_df["forecast"].iloc[-1])
    end_pct = ((end_price - last_price) / last_price) * 100 if last_price else 0.0
    direction_text = "naik" if end_pct >= 0 else "turun"

    forecast_x = [last_date, *forecast_df.index.tolist()]
    forecast_y = [last_price, *forecast_df["forecast"].tolist()]

    fig = go.Figure()

    fig.add_vrect(
        x0=last_date,
        x1=end_date,
        fillcolor=COLOR_FORECAST_ZONE,
        opacity=1,
        layer="below",
        line_width=0,
    )

    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent.values,
            mode="lines",
            name=f"Harga aktual ({n_recent} hari terakhir)",
            line={"color": COLOR_PRIMARY, "width": 2.7},
            hovertemplate=_price_hover("Data aktual"),
        )
    )
    fig.add_hline(
        y=last_price,
        line_width=1.1,
        line_dash="dot",
        line_color=COLOR_REFERENCE,
    )
    fig.add_trace(
        go.Scatter(
            x=[last_date],
            y=[last_price],
            mode="markers",
            name="Harga terakhir",
            marker={
                "color": COLOR_PRIMARY,
                "size": 9,
                "line": {"color": "#ffffff", "width": 2},
            },
            showlegend=False,
            hovertemplate=_price_hover("Harga terakhir"),
        )
    )

    if "upper_95" in forecast_df.columns and not forecast_df["upper_95"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df["upper_95"],
                mode="lines",
                line={"width": 1.2, "color": COLOR_CI_EDGE},
                name="Batas atas rentang",
                showlegend=False,
                hovertemplate=_price_hover("Batas atas 95%"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df["lower_95"],
                mode="lines",
                line={"width": 1.2, "color": COLOR_CI_EDGE},
                fill="tonexty",
                fillcolor=COLOR_CI_FILL,
                name="Rentang perkiraan 95%",
                hovertemplate=_price_hover("Batas bawah 95%"),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode="lines+markers",
            name="Prediksi model",
            line={"color": COLOR_SECONDARY, "width": 3.1, "dash": "dash"},
            marker={
                "color": COLOR_SECONDARY,
                "size": 5,
                "line": {"color": "#ffffff", "width": 1},
            },
            hovertemplate=_price_hover("Prediksi model"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[forecast_df.index[-1]],
            y=[forecast_df["forecast"].iloc[-1]],
            mode="markers",
            name="Prediksi akhir",
            marker={
                "color": COLOR_SECONDARY,
                "size": 9,
                "line": {"color": "#ffffff", "width": 2},
            },
            showlegend=False,
            hovertemplate=_price_hover("Prediksi akhir"),
        )
    )
    fig.add_vline(
        x=last_date,
        line_width=1.3,
        line_dash="dot",
        line_color=COLOR_ZERO,
    )
    fig.add_annotation(
        x=last_date,
        y=last_price,
        text=f"Harga terakhir<br><b>{_fmt_usd(last_price)}</b>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=COLOR_PRIMARY,
        ax=-46,
        ay=-42,
        bgcolor="#ffffff",
        bordercolor=COLOR_BORDER,
        borderwidth=1,
        borderpad=5,
        font={"size": 11, "color": COLOR_TEXT},
    )
    fig.add_annotation(
        x=end_date,
        y=end_price,
        text=(
            f"Prediksi akhir<br><b>{_fmt_usd(end_price)}</b>"
            f"<br><span style='font-size:10px'>{direction_text} {abs(end_pct):.2f}%</span>"
        ),
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=COLOR_SECONDARY,
        ax=42,
        ay=-44 if end_pct >= 0 else 44,
        bgcolor="#ffffff",
        bordercolor=COLOR_BORDER,
        borderwidth=1,
        borderpad=5,
        font={"size": 11, "color": COLOR_TEXT},
    )
    fig.add_annotation(
        x=last_date,
        y=1.0,
        xref="x",
        yref="paper",
        text="Mulai prediksi",
        showarrow=False,
        yshift=18,
        bgcolor="#ffffff",
        bordercolor=COLOR_BORDER,
        borderwidth=1,
        borderpad=4,
        font={"size": 11, "color": COLOR_TEXT},
    )
    fig.add_annotation(
        x=end_date,
        y=1.0,
        xref="x",
        yref="paper",
        text="Area prediksi",
        showarrow=False,
        yshift=18,
        bgcolor="#fff7ed",
        bordercolor=COLOR_CI_EDGE,
        borderwidth=1,
        borderpad=4,
        font={"size": 11, "color": "#92400e"},
    )

    all_y = recent.tolist() + forecast_df["forecast"].tolist()
    if "lower_95" in forecast_df.columns:
        all_y += forecast_df["lower_95"].dropna().tolist()
    if "upper_95" in forecast_df.columns:
        all_y += forecast_df["upper_95"].dropna().tolist()

    fig = _style_figure(
        fig,
        f"Prediksi Harga Bitcoin untuk {len(forecast_df)} Hari ke Depan",
        y_title="Harga Bitcoin (USD)",
        price_y=True,
        height=520,
    )
    fig.update_yaxes(range=_padded_price_range(all_y))
    fig.update_xaxes(range=[recent.index[0], end_date + pd.Timedelta(days=2)])
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
            x=test.index,
            y=test.values,
            mode="lines",
            name="Aktual",
            line={"color": COLOR_ACTUAL, "width": 2.6},
            hovertemplate=_price_hover("Aktual"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=arima_pred,
            mode="lines",
            name="Prediksi model",
            line={"color": COLOR_SECONDARY, "width": 2.4, "dash": "dash"},
            hovertemplate=_price_hover("Prediksi model"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=naive_pred,
            mode="lines",
            name="Baseline sederhana",
            line={"color": COLOR_DANGER, "width": 2.1, "dash": "dot"},
            hovertemplate=_price_hover("Baseline sederhana"),
        )
    )
    return _style_figure(
        fig,
        "Perbandingan Model dan Baseline Sederhana",
        y_title="Harga Bitcoin (USD)",
        price_y=True,
    )
