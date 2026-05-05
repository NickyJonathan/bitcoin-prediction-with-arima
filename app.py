import pickle
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

try:
    import joblib
except Exception:
    joblib = None


warnings.filterwarnings("ignore")

st.set_page_config(page_title="Bitcoin Forecast ARIMA", layout="wide")


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', 'Calibri', sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f4f7fb 0%, #ffffff 30%);
        }

        .main .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid #e5eaf1;
        }

        .subtle-text {
            color: #4e5a6a;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=900, show_spinner=False)
def fetch_bitcoin_data(days: int = 365) -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
    params = {"vs_currency": "usd", "days": days}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    raw_data = response.json()

    if not raw_data:
        raise ValueError("CoinGecko tidak mengembalikan data.")

    df = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "close"])
    if "close" not in df.columns:
        raise ValueError("Kolom close tidak ditemukan pada respons API.")

    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df = df[["date", "close"]].dropna().drop_duplicates(subset="date").sort_values("date")
    df = df.set_index("date")

    daily_close = df["close"].astype(float).resample("D").last().ffill()
    result = daily_close.to_frame(name="close")

    if len(result) < 200:
        raise ValueError(f"Data historis kurang dari 200 hari (didapat {len(result)}).")

    return result


def load_pretrained_model(model_path: Path):
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

    raise RuntimeError(f"Gagal memuat model dengan pickle: {pickle_error}") from pickle_error


@st.cache_resource(show_spinner=False)
def load_pretrained_model_cached(model_path_str: str):
    return load_pretrained_model(Path(model_path_str))


def build_forecast_index(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    start = last_date + pd.Timedelta(days=1)
    return pd.date_range(start=start, periods=horizon, freq="D")


def extract_forecast_values(model_result, steps: int) -> np.ndarray:
    if hasattr(model_result, "get_forecast"):
        forecast = model_result.get_forecast(steps=steps).predicted_mean
    elif hasattr(model_result, "forecast"):
        forecast = model_result.forecast(steps=steps)
    else:
        raise TypeError("Model tidak mendukung metode forecast.")

    values = np.asarray(forecast, dtype=float).reshape(-1)
    if len(values) != steps:
        raise ValueError("Jumlah nilai forecast tidak sesuai dengan horizon.")
    return values


def forecast_with_pretrained(model_result, series: pd.Series, steps: int) -> Tuple[np.ndarray, str]:
    if hasattr(model_result, "model") and getattr(model_result.model, "k_exog", 0):
        raise ValueError(
            "Model pre-trained membutuhkan variabel exogenous, "
            "sedangkan aplikasi ini menggunakan data univariat."
        )

    model_to_use = model_result
    note = "Model pre-trained digunakan langsung."

    if hasattr(model_result, "apply"):
        try:
            model_to_use = model_result.apply(series, refit=False)
            note = "State model pre-trained diselaraskan dengan data terbaru."
        except Exception as err:
            raise ValueError(f"Model pre-trained tidak kompatibel dengan data terbaru: {err}") from err

    values = extract_forecast_values(model_to_use, steps=steps)
    return values, note


def train_arima(series: pd.Series, order: Tuple[int, int, int]):
    model = ARIMA(
        series,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    try:
        return model.fit(method_kwargs={"maxiter": 120})
    except TypeError:
        return model.fit()


def default_settings() -> dict:
    return {
        "history_days": 300,
        "horizon": 14,
        "mode": "Pre-trained (arima_model.pkl)",
        "p": 2,
        "d": 1,
        "q": 2,
    }


def create_historical_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"],
            mode="lines",
            name="Harga Aktual (Close)",
            line={"color": "#2563eb", "width": 2.6},
        )
    )
    fig.update_layout(
        title="Data Historis Harga Penutupan Bitcoin",
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        hovermode="x unified",
    )
    return fig


def create_forecast_chart(actual_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual_df.index,
            y=actual_df["close"],
            mode="lines",
            name="Harga Aktual (Close)",
            line={"color": "#1d4ed8", "width": 2.7},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["forecast"],
            mode="lines",
            name="Prediksi ARIMA",
            line={"color": "#f97316", "width": 2.7, "dash": "dash"},
        )
    )
    fig.add_vline(
        x=forecast_df.index.min(),
        line_width=1.2,
        line_dash="dot",
        line_color="#94a3b8",
    )
    fig.update_layout(
        title="Perbandingan Harga Aktual dan Prediksi Bitcoin",
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        hovermode="x unified",
    )
    return fig


def main() -> None:
    apply_custom_style()

    st.title("Prediksi Harga Bitcoin dengan ARIMA")
    st.markdown(
        "<p class='subtle-text'>"
        "Aplikasi ini menggunakan data real-time CoinGecko untuk memprediksi harga Bitcoin "
        "dengan dua mode: model pre-trained atau training ulang ARIMA."
        "</p>",
        unsafe_allow_html=True,
    )

    st.info(
        "ARIMA (AutoRegressive Integrated Moving Average) adalah model time-series yang "
        "menggabungkan komponen autoregressive (p), differencing (d), dan moving average (q) "
        "untuk memproyeksikan nilai masa depan berdasarkan pola historis."
    )

    saved_settings = st.session_state.get("settings", default_settings())
    if not isinstance(saved_settings, dict):
        saved_settings = default_settings()

    with st.sidebar:
        st.header("Kontrol Prediksi")
        history_days = st.slider(
            "Data historis (hari)",
            min_value=200,
            max_value=365,
            value=int(saved_settings.get("history_days", 300)),
        )
        horizon = st.slider(
            "Horizon prediksi (hari)",
            min_value=7,
            max_value=30,
            value=int(saved_settings.get("horizon", 14)),
        )
        mode_options = ["Pre-trained (arima_model.pkl)", "Train ulang ARIMA"]
        mode = st.radio(
            "Mode model",
            options=mode_options,
            index=0 if saved_settings.get("mode") == mode_options[0] else 1,
        )

        p = int(saved_settings.get("p", 2))
        d = int(saved_settings.get("d", 1))
        q = int(saved_settings.get("q", 2))
        if mode == "Train ulang ARIMA":
            st.markdown("---")
            st.caption("Parameter ARIMA")
            p = st.slider("p (AR)", 0, 5, p)
            d = st.slider("d (Differencing)", 0, 2, d)
            q = st.slider("q (MA)", 0, 5, q)

        st.markdown("---")
        run_prediction = st.button("Jalankan Prediksi", type="primary", use_container_width=True)

        st.markdown("---")
        st.caption("Sumber data: CoinGecko API")

        current_settings = {
            "history_days": history_days,
            "horizon": horizon,
            "mode": mode,
            "p": p,
            "d": d,
            "q": q,
        }

        if run_prediction:
            st.session_state["settings"] = current_settings
            st.session_state["need_recompute"] = True
        elif "settings" not in st.session_state:
            st.session_state["settings"] = current_settings
            st.session_state["need_recompute"] = True
        elif current_settings != st.session_state["settings"]:
            st.info("Perubahan kontrol belum diterapkan. Klik 'Jalankan Prediksi'.")

    active_settings = st.session_state["settings"]

    with st.spinner("Mengambil data harga Bitcoin dari CoinGecko..."):
        try:
            full_df = fetch_bitcoin_data(days=365)
        except Exception as err:
            st.error(f"Gagal mengambil data API: {err}")
            st.stop()

    btc_df = full_df.tail(int(active_settings["history_days"])).copy()

    st.markdown("---")
    st.subheader("Ringkasan Data")
    st.plotly_chart(create_historical_chart(btc_df), use_container_width=True)

    close_series = btc_df["close"]
    horizon = int(active_settings["horizon"])

    active_signature = (
        int(active_settings["history_days"]),
        horizon,
        active_settings["mode"],
        int(active_settings["p"]),
        int(active_settings["d"]),
        int(active_settings["q"]),
        str(btc_df.index[-1].date()),
        float(close_series.iloc[-1]),
    )
    needs_recompute = (
        st.session_state.get("need_recompute", False)
        or "forecast_values" not in st.session_state
        or st.session_state.get("active_signature") != active_signature
    )

    if needs_recompute:
        try:
            if active_settings["mode"] == "Pre-trained (arima_model.pkl)":
                with st.spinner("Memuat model pre-trained dan menghasilkan prediksi..."):
                    model_path = Path(__file__).resolve().with_name("arima_model.pkl")
                    model = load_pretrained_model_cached(str(model_path))
                    forecast_values, model_note = forecast_with_pretrained(
                        model_result=model,
                        series=close_series,
                        steps=horizon,
                    )
            else:
                p, d, q = int(active_settings["p"]), int(active_settings["d"]), int(active_settings["q"])
                with st.spinner(f"Training ARIMA({p},{d},{q}) sedang berjalan..."):
                    trained_model = train_arima(close_series, order=(p, d, q))
                    forecast_values = extract_forecast_values(trained_model, steps=horizon)
                    model_note = f"Model ARIMA({p},{d},{q}) berhasil dilatih ulang dari data terbaru."

            st.session_state["forecast_values"] = forecast_values.tolist()
            st.session_state["model_note"] = model_note
            st.session_state["active_signature"] = active_signature
            st.session_state["need_recompute"] = False
        except Exception as err:
            if "forecast_values" not in st.session_state:
                if active_settings["mode"] == "Pre-trained (arima_model.pkl)":
                    st.error(f"Gagal menggunakan model pre-trained: {err}")
                else:
                    st.error(f"Gagal melakukan training model ARIMA: {err}")
                st.stop()
            st.warning("Prediksi terbaru gagal dihitung. Menampilkan hasil terakhir yang berhasil.")
            st.session_state["need_recompute"] = False

    forecast_values = np.asarray(st.session_state["forecast_values"], dtype=float)
    model_note = st.session_state.get("model_note", "")

    forecast_index = build_forecast_index(btc_df.index[-1], horizon)
    forecast_df = pd.DataFrame({"forecast": forecast_values}, index=forecast_index)

    last_price = float(close_series.iloc[-1])
    next_price = float(forecast_df["forecast"].iloc[0])
    end_price = float(forecast_df["forecast"].iloc[-1])

    next_day_delta_pct = ((next_price - last_price) / last_price) * 100
    horizon_delta_pct = ((end_price - last_price) / last_price) * 100
    trend_direction = "Naik" if horizon_delta_pct >= 0 else "Turun"

    st.markdown("---")
    st.subheader("Hasil Prediksi")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Harga Terakhir Bitcoin (USD)", f"${last_price:,.2f}")
    metric_col2.metric(
        "Prediksi Hari Berikutnya (USD)",
        f"${next_price:,.2f}",
        f"{next_day_delta_pct:+.2f}%",
    )
    metric_col3.metric(
        "Arah Tren",
        trend_direction,
        f"{horizon_delta_pct:+.2f}% ({horizon} hari)",
    )

    st.caption(model_note)

    st.plotly_chart(create_forecast_chart(btc_df, forecast_df), use_container_width=True)

    st.markdown("### Tabel Forecast")
    result_table = forecast_df.reset_index().rename(
        columns={"index": "Tanggal", "forecast": "Prediksi Harga (USD)"}
    )
    result_table["Prediksi Harga (USD)"] = result_table["Prediksi Harga (USD)"].map(
        lambda value: f"${value:,.2f}"
    )
    st.dataframe(
        result_table,
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
