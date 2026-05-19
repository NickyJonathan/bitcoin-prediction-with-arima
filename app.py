"""
app.py
Aplikasi prediksi harga Bitcoin dengan ARIMA - entry point.

Aplikasi ini mengimplementasikan tahapan KDD (Knowledge Discovery in Databases)
sesuai metodologi penelitian:
    1. Pengumpulan data       - Tab Data
    2. Pra-pemrosesan         - Tab Data
    3. Uji stasioneritas      - Tab Stasioneritas
    4. Identifikasi model     - Tab Pemilihan Model
    5. Estimasi parameter     - Tab Pemilihan Model
    6. Diagnostik residual    - Tab Diagnostik
    7. Peramalan dan evaluasi - Tab Peramalan
"""
import warnings

import pandas as pd
import streamlit as st

import viz
from config import (
    DEFAULT_D,
    DEFAULT_HORIZON,
    DEFAULT_P,
    DEFAULT_Q,
    GRID_D_MAX,
    GRID_P_MAX,
    GRID_Q_MAX,
    MAX_HORIZON,
    MIN_HORIZON,
    MODE_AUTO,
    MODE_MANUAL,
    MODE_OPTIONS,
    MODE_PRETRAINED,
    PRETRAINED_MODEL_PATH,
)
from data_loader import fetch_bitcoin_data, get_descriptive_stats
from diagnostics import diagnostic_summary
from evaluation import compare_with_naive, naive_predictions
from forecasting import (
    build_forecast_dataframe,
    forecast_with_ci,
    rolling_one_step_forecast,
)
from modeling import (
    apply_pretrained_to_series,
    get_coefficients_df,
    get_model_info,
    grid_search_arima,
    load_pretrained_model_cached,
    train_arima,
)
from preprocessing import get_split_summary, train_test_split_timebased
from stationarity import compute_acf_pacf, difference, find_optimal_d

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Dashboard Bitcoin ARIMA",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS Kustom
# ============================================================
def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
        }
        html, body, [data-testid="stAppViewContainer"]  {
            font-family: 'Segoe UI', 'Inter', 'Calibri', sans-serif;
            color: #111827;
            letter-spacing: 0;
        }
        [data-testid="stAppViewContainer"] {
            background: #f6f8fb;
            color: #111827;
        }
        [data-testid="stHeader"] {
            background: rgba(246, 248, 251, 0.96);
        }
        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div,
        [data-testid="stSidebarContent"] {
            background: #ffffff;
            color: #111827;
            border-right: 1px solid #e5e7eb;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
            color: #111827 !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] small {
            color: #526070 !important;
        }
        h1, h2, h3 {
            color: #111827;
            letter-spacing: 0;
        }
        h1 {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        h2 {
            font-size: 1.42rem;
            font-weight: 700;
            margin-top: 0.6rem;
        }
        h3 {
            font-size: 1.05rem;
            font-weight: 650;
            margin-top: 1.2rem;
        }
        .subtle-text {
            color: #526070;
            font-size: 0.96rem;
            line-height: 1.55;
            max-width: 880px;
        }
        .guide-panel {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 1rem 0 0.35rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .guide-title {
            color: #111827;
            font-size: 0.98rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }
        .guide-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
        }
        .guide-item {
            color: #526070;
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .guide-item strong {
            color: #111827;
            display: block;
            font-size: 0.92rem;
            margin-bottom: 0.2rem;
        }
        .legend-swatch {
            display: inline-block;
            width: 22px;
            height: 3px;
            border-radius: 999px;
            margin-right: 0.45rem;
            vertical-align: middle;
        }
        .legend-actual { background: #2563eb; }
        .legend-forecast {
            background: repeating-linear-gradient(
                90deg,
                #d97706 0,
                #d97706 7px,
                transparent 7px,
                transparent 12px
            );
        }
        .legend-range {
            height: 10px;
            background: rgba(217, 119, 6, 0.18);
            border: 1px solid rgba(217, 119, 6, 0.25);
        }
        .chart-note {
            background: #f8fafc;
            border-left: 3px solid #1f4e79;
            border-radius: 6px;
            color: #475569;
            font-size: 0.92rem;
            line-height: 1.5;
            padding: 0.75rem 0.9rem;
            margin: 0.45rem 0 0.9rem;
        }
        @media (max-width: 760px) {
            .guide-grid {
                grid-template-columns: 1fr;
            }
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid #e5e7eb;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            font-weight: 600;
            padding: 0.65rem 1rem;
        }
        .stTabs [aria-selected="true"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-bottom-color: #ffffff;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.95rem 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stMetricLabel"] {
            color: #5b6472;
        }
        [data-testid="stMetricValue"] {
            color: #111827;
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.2;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        [data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            overflow: hidden;
        }
        .stButton > button {
            border-radius: 6px;
            border: 1px solid #1f4e79;
            background: #1f4e79;
            color: #ffffff;
            font-weight: 600;
        }
        .stButton > button * {
            color: #ffffff !important;
        }
        .stButton > button:hover {
            border-color: #173c5f;
            background: #173c5f;
            color: #ffffff;
        }
        hr {
            margin: 1.5rem 0;
            border-color: #e5e7eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Load Data + Split
# ============================================================
def load_initial_data():
    if "data" in st.session_state:
        return

    with st.spinner("Mengambil data harga Bitcoin dari CoinGecko..."):
        try:
            df = fetch_bitcoin_data()
        except Exception as err:
            st.error(f"Gagal mengambil data dari CoinGecko: {err}")
            st.info("Pastikan koneksi internet aktif dan API CoinGecko dapat diakses.")
            st.stop()

    series = df["close"]
    train, test = train_test_split_timebased(series)
    st.session_state["data"] = {"full": series, "train": train, "test": test}


# ============================================================
# Sidebar Global Controls
# ============================================================
def sidebar_controls():
    with st.sidebar:
        st.markdown("### Pengaturan")
        st.caption("Atur mode model dan horizon prediksi.")

        current_mode = st.session_state.get("mode", MODE_PRETRAINED)
        if current_mode not in MODE_OPTIONS:
            current_mode = MODE_PRETRAINED

        mode = st.radio(
            "Mode model",
            options=MODE_OPTIONS,
            index=MODE_OPTIONS.index(current_mode),
            help=(
                "**Model tersimpan**: menggunakan arima_model.pkl.\n\n"
                "**Latih manual**: melatih ARIMA dengan parameter (p, d, q) pilihan Anda.\n\n"
                "**Pencarian otomatis**: memilih kombinasi (p, d, q) terbaik berdasarkan AIC."
            ),
        )
        st.session_state["mode"] = mode

        if mode == MODE_MANUAL:
            st.markdown("---")
            st.caption("**Parameter ARIMA manual**")
            p = st.slider("Orde p (AR)", 0, GRID_P_MAX, st.session_state.get("p", DEFAULT_P))
            d = st.slider("Orde d (differencing)", 0, GRID_D_MAX, st.session_state.get("d", DEFAULT_D))
            q = st.slider("Orde q (MA)", 0, GRID_Q_MAX, st.session_state.get("q", DEFAULT_Q))
            st.session_state["p"], st.session_state["d"], st.session_state["q"] = p, d, q

        st.markdown("---")
        horizon = st.slider(
            "Horizon prediksi (hari)",
            MIN_HORIZON, MAX_HORIZON,
            st.session_state.get("horizon", DEFAULT_HORIZON),
        )
        st.session_state["horizon"] = horizon

        st.markdown("---")
        if st.button("Perbarui data CoinGecko", width="stretch"):
            for key in [
                "data",
                "trained_model",
                "model_meta",
                "model_signature",
                "full_history_model",
                "full_history_model_signature",
            ]:
                st.session_state.pop(key, None)
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.caption("Sumber data: CoinGecko API")
        st.caption("Metodologi: KDD + Box-Jenkins")


# ============================================================
# Active Model Helper (cached via session_state)
# ============================================================
def get_active_model():
    mode = st.session_state["mode"]
    train = st.session_state["data"]["train"]

    if mode == MODE_PRETRAINED:
        sig = ("pretrained", str(train.index[-1].date()), len(train))
    elif mode == MODE_MANUAL:
        sig = (
            "manual",
            st.session_state["p"], st.session_state["d"], st.session_state["q"],
            str(train.index[-1].date()), len(train),
        )
    else:
        sig = ("auto", str(train.index[-1].date()), len(train))

    if st.session_state.get("model_signature") == sig:
        return st.session_state["trained_model"], st.session_state["model_meta"]

    # Build model
    if mode == MODE_PRETRAINED:
        if not PRETRAINED_MODEL_PATH.exists():
            st.error(
                f"File model tersimpan tidak ditemukan: {PRETRAINED_MODEL_PATH.name}. "
                "Silakan pilih mode lain."
            )
            st.stop()
        with st.spinner("Memuat model tersimpan..."):
            base_model = load_pretrained_model_cached(str(PRETRAINED_MODEL_PATH))
            model = apply_pretrained_to_series(base_model, train)
        meta = {"mode": mode, "note": "Model tersimpan dimuat dari arima_model.pkl."}

    elif mode == MODE_MANUAL:
        p, d, q = st.session_state["p"], st.session_state["d"], st.session_state["q"]
        with st.spinner(f"Melatih ARIMA({p},{d},{q}) pada data latih..."):
            model = train_arima(train, (p, d, q))
        meta = {
            "mode": mode,
            "note": f"Model ARIMA({p},{d},{q}) dilatih ulang menggunakan data latih.",
        }

    else:  # AUTO
        with st.spinner("Menentukan d optimal (uji ADF)..."):
            d_opt, _ = find_optimal_d(train)

        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        def cb(idx, total):
            progress_bar.progress(idx / total)
            progress_text.caption(f"Pencarian otomatis: {idx}/{total} kombinasi")

        results_df = grid_search_arima(
            train, d=d_opt, p_max=GRID_P_MAX, q_max=GRID_Q_MAX, progress_callback=cb
        )
        progress_bar.empty()
        progress_text.empty()

        best = results_df.iloc[0]
        with st.spinner(
            f"Melatih ARIMA({int(best['p'])},{int(best['d'])},{int(best['q'])}) terbaik..."
        ):
            model = train_arima(train, (int(best["p"]), int(best["d"]), int(best["q"])))
        meta = {
            "mode": mode,
            "note": (
                f"Pencarian otomatis memilih ARIMA({int(best['p'])},{int(best['d'])},{int(best['q'])}) "
                f"dengan AIC={best['aic']:.2f}."
            ),
            "grid_results": results_df,
            "d_optimal": d_opt,
        }

    st.session_state["trained_model"] = model
    st.session_state["model_meta"] = meta
    st.session_state["model_signature"] = sig
    return model, meta


def get_full_history_model(active_model, meta: dict, full: pd.Series):
    """Prepare a forecast model whose origin is the latest available data point."""
    if meta["mode"] == MODE_PRETRAINED:
        sig = ("full", "pretrained", str(full.index[-1].date()), len(full))
    else:
        sig = (
            "full",
            "retrained",
            active_model.model.order,
            str(full.index[-1].date()),
            len(full),
        )

    if st.session_state.get("full_history_model_signature") == sig:
        return st.session_state["full_history_model"]

    if meta["mode"] == MODE_PRETRAINED:
        base_model = load_pretrained_model_cached(str(PRETRAINED_MODEL_PATH))
        full_model = apply_pretrained_to_series(base_model, full)
    else:
        full_model = train_arima(full, active_model.model.order)

    st.session_state["full_history_model"] = full_model
    st.session_state["full_history_model_signature"] = sig
    return full_model


def fmt_usd(value: float) -> str:
    return f"${value:,.2f}"


def render_chart_guide() -> None:
    st.markdown(
        """
        <div class="guide-panel">
            <div class="guide-title">Panduan cepat membaca grafik</div>
            <div class="guide-grid">
                <div class="guide-item">
                    <strong><span class="legend-swatch legend-actual"></span>Garis biru</strong>
                    Harga Bitcoin yang benar-benar terjadi pada data historis.
                </div>
                <div class="guide-item">
                    <strong><span class="legend-swatch legend-forecast"></span>Garis oranye</strong>
                    Perkiraan harga dari model untuk beberapa hari ke depan.
                </div>
                <div class="guide-item">
                    <strong><span class="legend-swatch legend-range"></span>Area krem</strong>
                    Rentang perkiraan. Semakin lebar areanya, semakin besar ketidakpastiannya.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart_note(text: str) -> None:
    st.markdown(f"<div class='chart-note'>{text}</div>", unsafe_allow_html=True)


# ============================================================
# TAB 1: BERANDA
# ============================================================
def render_tab_beranda():
    st.markdown("## Ringkasan Prediksi")
    st.markdown(
        "<p class='subtle-text'>Pantau harga terakhir, arah proyeksi, "
        "dan interval prediksi berdasarkan model ARIMA aktif.</p>",
        unsafe_allow_html=True,
    )

    data = st.session_state["data"]
    full = data["full"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Harga terakhir", fmt_usd(full.iloc[-1]))
    c2.metric("Periode data", f"{full.index.min().date()} sampai {full.index.max().date()}")
    c3.metric("Jumlah hari", f"{len(full)}")

    render_chart_guide()

    st.markdown("---")
    st.markdown("### Grafik harga 90 hari terakhir")
    render_chart_note(
        "Grafik ini menunjukkan pergerakan harga terbaru. Fokus utamanya adalah arah umum "
        "dan besar naik-turunnya harga, bukan hanya satu titik harga tertentu."
    )
    st.plotly_chart(
        viz.plot_historical(full.tail(90).to_frame("close")),
        width="stretch",
        key="beranda_historical_90",
    )

    st.markdown("---")
    st.markdown("### Hasil prediksi")
    model, meta = get_active_model()
    st.caption(meta["note"])

    horizon = st.session_state["horizon"]
    with st.spinner(f"Membuat prediksi {horizon} hari ke depan dari data terbaru..."):
        forecast_model = get_full_history_model(model, meta, full)
        mean, lower, upper = forecast_with_ci(forecast_model, horizon=horizon)
        forecast_df = build_forecast_dataframe(full.index[-1], mean, lower, upper)

    last_price = float(full.iloc[-1])
    next_price = float(forecast_df["forecast"].iloc[0])
    end_price = float(forecast_df["forecast"].iloc[-1])
    next_pct = ((next_price - last_price) / last_price) * 100
    end_pct = ((end_price - last_price) / last_price) * 100
    direction = "Naik" if end_pct >= 0 else "Turun"

    m1, m2, m3 = st.columns(3)
    m1.metric("Harga terakhir", fmt_usd(last_price))
    m2.metric("Prediksi hari berikutnya", fmt_usd(next_price), f"{next_pct:+.2f}%")
    m3.metric(f"Arah proyeksi ({horizon} hari)", direction, f"{end_pct:+.2f}%")

    render_chart_note(
        f"Dalam {horizon} hari ke depan, model memperkirakan harga bergerak "
        f"{direction.lower()} sekitar {abs(end_pct):.2f}% dari harga terakhir. "
        "Gunakan area rentang perkiraan sebagai batas ketidakpastian, bukan hanya garis prediksi."
    )
    st.plotly_chart(
        viz.plot_forecast_with_ci(full, forecast_df),
        width="stretch",
        key="beranda_forecast",
    )

    with st.expander("Tabel detail prediksi"):
        disp = forecast_df.reset_index().rename(columns={"index": "Tanggal"})
        disp["forecast"] = disp["forecast"].map(fmt_usd)
        disp["lower_95"] = disp["lower_95"].map(fmt_usd)
        disp["upper_95"] = disp["upper_95"].map(fmt_usd)
        disp.columns = ["Tanggal", "Prediksi", "Rentang bawah 95%", "Rentang atas 95%"]
        st.dataframe(disp, width="stretch", hide_index=True)

    st.markdown("---")
    st.warning(
        "Disclaimer: Prediksi ini berbasis model statistik dan data harga historis. "
        "Hasilnya bukan saran investasi, dan keputusan finansial tetap menjadi tanggung jawab pengguna."
    )


# ============================================================
# TAB 2: DATA
# ============================================================
def render_tab_data():
    st.markdown("## Data dan Statistik")
    st.markdown(
        "<p class='subtle-text'>Ringkasan data harga penutupan Bitcoin harian "
        "yang digunakan untuk pelatihan dan evaluasi model.</p>",
        unsafe_allow_html=True,
    )

    full = st.session_state["data"]["full"]
    train = st.session_state["data"]["train"]
    test = st.session_state["data"]["test"]

    st.markdown("### Statistik deskriptif")
    stats = get_descriptive_stats(full)

    def _fmt(k, v):
        if isinstance(v, float):
            if "USD" in k:
                return f"${v:,.2f}"
            if "%" in k:
                return f"{v:+.2f}%"
            return f"{v:.4f}"
        return v

    stats_df = pd.DataFrame(
        [{"Statistik": k, "Nilai": _fmt(k, v)} for k, v in stats.items()]
    )
    stats_df["Nilai"] = stats_df["Nilai"].astype(str)
    st.dataframe(stats_df, width="stretch", hide_index=True)

    st.markdown("### Riwayat harga")
    render_chart_note(
        "Bagian ini memperlihatkan harga penutupan harian dalam seluruh periode data. "
        "Garis yang lebih sering berubah tajam menandakan volatilitas yang lebih tinggi."
    )
    st.plotly_chart(
        viz.plot_historical(full.to_frame("close")),
        width="stretch",
        key="data_historical_full",
    )

    st.markdown("### Distribusi harga")
    render_chart_note(
        "Distribusi menunjukkan seberapa sering harga berada pada rentang tertentu. "
        "Batang yang lebih tinggi berarti harga lebih sering muncul di rentang tersebut."
    )
    st.plotly_chart(
        viz.plot_distribution(full),
        width="stretch",
        key="data_distribution",
    )

    st.markdown("### Pembagian data latih dan uji (80:20)")
    split_summary = get_split_summary(train, test)
    split_df = pd.DataFrame(
        [{"Keterangan": key, "Nilai": value} for key, value in split_summary.items()]
    )
    split_df["Nilai"] = split_df["Nilai"].astype(str)
    st.dataframe(split_df, width="stretch", hide_index=True)
    render_chart_note(
        "Data latih dipakai untuk membangun model. Data uji dipakai untuk melihat apakah "
        "model tetap masuk akal saat dibandingkan dengan data yang belum dipakai saat pelatihan."
    )
    st.plotly_chart(
        viz.plot_train_test_split(train, test),
        width="stretch",
        key="data_train_test_split",
    )


# ============================================================
# TAB 3: STASIONERITAS
# ============================================================
def render_tab_stationarity():
    st.markdown("## Analisis Stasioneritas")
    st.markdown(
        "<p class='subtle-text'>Uji ADF, differencing, ACF, dan PACF "
        "untuk memeriksa kesiapan deret waktu sebelum pemodelan ARIMA.</p>",
        unsafe_allow_html=True,
    )

    train = st.session_state["data"]["train"]

    st.markdown("### Uji Augmented Dickey-Fuller (ADF)")
    st.markdown(
        "**Hipotesis:**\n"
        "- H0: data memiliki *unit root* (tidak stasioner)\n"
        "- H1: data tidak memiliki *unit root* (stasioner)\n\n"
        "Differencing dilakukan hingga p-value < 0,05."
    )

    with st.spinner("Menjalankan uji ADF iteratif..."):
        d_opt, history = find_optimal_d(train)

    rows = []
    for d, res in history.items():
        rows.append({
            "Orde d": d,
            "Statistik ADF": f"{res['statistic']:.4f}",
            "p-value": f"{res['p_value']:.6f}",
            "Nilai kritis 5%": f"{res['critical_5pct']:.4f}",
            "Jumlah observasi": res["n_obs"],
            "Status": "Stasioner" if res["is_stationary"] else "Tidak stasioner",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.success(f"Orde differencing optimal: d = {d_opt}")

    st.markdown("### Visualisasi Differencing")
    if d_opt > 0:
        differenced = difference(train, d_opt)
        render_chart_note(
            "Differencing membantu mengubah pola harga yang terus bergerak menjadi perubahan harian "
            "yang lebih stabil, sehingga model ARIMA lebih mudah membaca pola."
        )
        st.plotly_chart(
            viz.plot_differencing(train, differenced, d_opt),
            width="stretch",
            key="stationarity_differencing",
        )
    else:
        st.info("Data sudah stasioner pada level (d=0), tidak perlu differencing.")
        differenced = train

    st.markdown("### Analisis ACF dan PACF")
    st.caption("Pola cut-off dan tailing-off pada plot ACF/PACF memberi indikasi awal orde p dan q.")
    render_chart_note(
        "Batang yang keluar dari area abu-abu menunjukkan hubungan antar-lag yang masih cukup kuat. "
        "Bagian ini lebih teknis dan dipakai untuk mendukung pemilihan parameter model."
    )
    n_lags = st.slider("Jumlah lag", 10, 50, 30)
    acf_data = compute_acf_pacf(differenced if d_opt > 0 else train, n_lags=n_lags)
    st.plotly_chart(
        viz.plot_acf_pacf(acf_data),
        width="stretch",
        key="stationarity_acf_pacf",
    )


# ============================================================
# TAB 4: PEMILIHAN MODEL
# ============================================================
def render_tab_model():
    st.markdown("## Pemilihan Model")
    st.markdown(
        "<p class='subtle-text'>Ringkasan mode model, hasil pencarian parameter, "
        "dan estimasi koefisien ARIMA yang sedang digunakan.</p>",
        unsafe_allow_html=True,
    )

    mode = st.session_state["mode"]
    st.info(f"Mode aktif: {mode}")

    model, meta = get_active_model()
    st.caption(meta["note"])

    if mode == MODE_AUTO and "grid_results" in meta:
        st.markdown("### Hasil pencarian parameter berbasis AIC")
        st.markdown(f"Orde differencing: **d = {meta['d_optimal']}** (dari uji ADF).")
        st.markdown(f"Total kombinasi yang berhasil konvergen: **{len(meta['grid_results'])} model**.")

        display_grid = meta["grid_results"].head(10)[["order_str", "aic", "bic", "llf"]].copy()
        display_grid.columns = ["ARIMA(p,d,q)", "AIC", "BIC", "Log-Likelihood"]
        display_grid["AIC"] = display_grid["AIC"].map(lambda x: f"{x:.4f}")
        display_grid["BIC"] = display_grid["BIC"].map(lambda x: f"{x:.4f}")
        display_grid["Log-Likelihood"] = display_grid["Log-Likelihood"].map(lambda x: f"{x:.4f}")
        st.markdown("**10 model terbaik:**")
        st.dataframe(display_grid, width="stretch", hide_index=True)

    st.markdown("### Informasi model terpilih")
    info = get_model_info(model)
    info_df = pd.DataFrame([{"Parameter": k, "Nilai": v} for k, v in info.items()])
    info_df["Nilai"] = info_df["Nilai"].astype(str)
    st.dataframe(info_df, width="stretch", hide_index=True)

    st.markdown("### Estimasi koefisien model")
    st.caption("Estimasi menggunakan Maximum Likelihood Estimation (MLE).")
    try:
        coef_df = get_coefficients_df(model)
        st.dataframe(coef_df, width="stretch", hide_index=True)
    except Exception as err:
        st.warning(f"Tidak dapat menampilkan tabel koefisien: {err}")

    with st.expander("Ringkasan lengkap statsmodels"):
        st.text(str(model.summary()))


# ============================================================
# TAB 5: DIAGNOSTIK
# ============================================================
def render_tab_diagnostics():
    st.markdown("## Diagnostik Residual")
    st.markdown(
        "<p class='subtle-text'>Pemeriksaan residual untuk menilai apakah model "
        "sudah cukup memadai setelah proses estimasi.</p>",
        unsafe_allow_html=True,
    )

    model, _ = get_active_model()
    diag = diagnostic_summary(model)

    st.markdown("### Statistik deskriptif residual")
    stats_df = pd.DataFrame([
        {"Statistik": k, "Nilai": (f"{v:.4f}" if isinstance(v, float) else v)}
        for k, v in diag["stats"].items()
    ])
    stats_df["Nilai"] = stats_df["Nilai"].astype(str)
    st.dataframe(stats_df, width="stretch", hide_index=True)

    st.markdown("### Uji Ljung-Box pada beberapa lag")
    st.markdown(
        "**Hipotesis:**\n"
        "- H0: residual tidak berautokorelasi (model memadai)\n"
        "- H1: residual berautokorelasi"
    )
    lb_df = diag["ljung_box"].copy()
    lb_df["Statistik Q"] = lb_df["Statistik Q"].map(lambda x: f"{x:.4f}")
    lb_df["p-value"] = lb_df["p-value"].map(lambda x: f"{x:.4f}")
    st.dataframe(lb_df, width="stretch", hide_index=True)

    n_pass = (diag["ljung_box"]["p-value"] > 0.05).sum()
    total = len(diag["ljung_box"])
    if n_pass == total:
        st.success(f"Residual tidak berautokorelasi pada seluruh {total} lag yang diuji.")
    else:
        st.warning(f"Residual berautokorelasi pada {total - n_pass} dari {total} lag.")

    st.markdown("### Uji normalitas Jarque-Bera")
    jb = diag["jarque_bera"]
    jb_df = pd.DataFrame([
        {"Statistik": "Statistik JB", "Nilai": f"{jb['Statistik JB']:.4f}"},
        {"Statistik": "p-value", "Nilai": f"{jb['p-value']:.6f}"},
        {"Statistik": "Skewness", "Nilai": f"{jb['Skewness']:.4f}"},
        {"Statistik": "Kurtosis", "Nilai": f"{jb['Kurtosis']:.4f} (normal=3)"},
    ])
    st.dataframe(jb_df, width="stretch", hide_index=True)
    if jb["is_normal"]:
        st.success(jb["conclusion"])
    else:
        st.warning(
            f"{jb['conclusion']}. Penyimpangan normalitas umum terjadi pada data finansial "
            "dan tidak selalu mengganggu prediksi titik."
        )

    st.markdown("### Visualisasi diagnostik residual")
    render_chart_note(
        "Residual adalah selisih antara nilai aktual dan nilai yang diperkirakan model. "
        "Model yang baik biasanya memiliki residual yang tidak membentuk pola jelas."
    )
    st.plotly_chart(
        viz.plot_residual_diagnostics(diag["residuals_array"]),
        width="stretch",
        key="diagnostics_residual",
    )


# ============================================================
# TAB 6: PERAMALAN
# ============================================================
def render_tab_forecast():
    st.markdown("## Peramalan dan Evaluasi")
    st.markdown(
        "<p class='subtle-text'>Evaluasi kinerja model pada data uji, perbandingan "
        "dengan baseline sederhana, dan prediksi harga untuk horizon yang dipilih.</p>",
        unsafe_allow_html=True,
    )

    model, meta = get_active_model()
    train = st.session_state["data"]["train"]
    test = st.session_state["data"]["test"]
    full = st.session_state["data"]["full"]

    st.markdown("### Evaluasi pada data uji")
    st.caption("Strategi evaluasi: one-step ahead rolling, dengan nilai aktual ditambahkan setelah tiap prediksi.")

    with st.spinner("Menghasilkan prediksi rolling pada data uji..."):
        predictions = rolling_one_step_forecast(model, test)

    naive_pred = naive_predictions(train, test)
    comparison = compare_with_naive(test.values, predictions, naive_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", fmt_usd(comparison["model"]["mae"]))
    c2.metric("RMSE", fmt_usd(comparison["model"]["rmse"]))
    c3.metric("MAPE", f"{comparison['model']['mape']:.2f}%")
    c4.metric("Kategori akurasi", comparison["model"]["category"])

    render_chart_note(
        "Pada grafik evaluasi, garis aktual dan garis prediksi yang saling berdekatan "
        "menunjukkan model lebih mampu mengikuti pergerakan harga pada data uji."
    )
    st.plotly_chart(
        viz.plot_actual_vs_predicted(
            train.iloc[-60:], test, predictions, mape_value=comparison["model"]["mape"]
        ),
        width="stretch",
        key="forecast_actual_vs_predicted",
    )

    st.markdown("### Perbandingan dengan baseline sederhana")
    st.markdown(
        "Baseline sederhana menggunakan nilai aktual hari sebelumnya sebagai prediksi. "
        "Pembanding ini membantu menilai apakah ARIMA memberi nilai tambah terhadap pola historis sederhana."
    )

    comp_df = pd.DataFrame({
        "Metrik": ["MAE", "RMSE", "MAPE"],
        "ARIMA": [
            fmt_usd(comparison["model"]["mae"]),
            fmt_usd(comparison["model"]["rmse"]),
            f"{comparison['model']['mape']:.4f}%",
        ],
        "Baseline sederhana": [
            fmt_usd(comparison["naive"]["mae"]),
            fmt_usd(comparison["naive"]["rmse"]),
            f"{comparison['naive']['mape']:.4f}%",
        ],
        "Selisih (model - baseline)": [
            fmt_usd(comparison["model"]["mae"] - comparison["naive"]["mae"]),
            fmt_usd(comparison["model"]["rmse"] - comparison["naive"]["rmse"]),
            f"{comparison['model']['mape'] - comparison['naive']['mape']:+.4f}%",
        ],
    })
    st.dataframe(comp_df, width="stretch", hide_index=True)

    u = comparison["theils_u"]
    st.metric(
        "Theil's U Statistic", f"{u:.4f}",
        help="U < 1: model lebih baik dari baseline sederhana. U > 1: baseline sederhana lebih baik.",
    )
    if u < 1:
        st.success(comparison["interpretation"])
    else:
        st.warning(comparison["interpretation"])

    render_chart_note(
        "Baseline sederhana adalah pembanding paling dasar: harga besok dianggap sama dengan harga hari ini. "
        "Model ARIMA perlu mengungguli pembanding ini agar manfaatnya terlihat jelas."
    )
    st.plotly_chart(
        viz.plot_naive_comparison(test, predictions, naive_pred),
        width="stretch",
        key="forecast_baseline_comparison",
    )

    st.markdown("---")
    st.markdown("### Prediksi harga ke depan")
    horizon = st.session_state["horizon"]

    if meta["mode"] == MODE_PRETRAINED:
        full_model = get_full_history_model(model, meta, full)
    else:
        with st.spinner(f"Melatih ulang ARIMA{model.model.order} pada seluruh data..."):
            full_model = get_full_history_model(model, meta, full)

    mean, lower, upper = forecast_with_ci(full_model, horizon=horizon)
    forecast_df = build_forecast_dataframe(full.index[-1], mean, lower, upper)
    render_chart_note(
        "Garis prediksi menunjukkan perkiraan utama. Area rentang perkiraan menunjukkan kemungkinan "
        "naik-turun di sekitar prediksi, sehingga keputusan sebaiknya melihat rentangnya juga."
    )
    st.plotly_chart(
        viz.plot_forecast_with_ci(full, forecast_df),
        width="stretch",
        key="forecast_future",
    )

    with st.expander("Tabel detail prediksi"):
        disp = forecast_df.reset_index().rename(columns={"index": "Tanggal"})
        disp["forecast"] = disp["forecast"].map(fmt_usd)
        disp["lower_95"] = disp["lower_95"].map(fmt_usd)
        disp["upper_95"] = disp["upper_95"].map(fmt_usd)
        disp.columns = ["Tanggal", "Prediksi", "Rentang bawah 95%", "Rentang atas 95%"]
        st.dataframe(disp, width="stretch", hide_index=True)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    apply_custom_style()

    st.title("Prediksi Harga Bitcoin dengan ARIMA")
    st.markdown(
        "<p class='subtle-text'>Dashboard analitik untuk mengevaluasi data historis Bitcoin, "
        "memilih model ARIMA, memeriksa diagnostik residual, dan melihat prediksi harga "
        "dengan interval kepercayaan.</p>",
        unsafe_allow_html=True,
    )

    load_initial_data()
    sidebar_controls()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Beranda",
        "Data",
        "Stasioneritas",
        "Pemilihan Model",
        "Diagnostik",
        "Peramalan",
    ])

    with tab1: render_tab_beranda()
    with tab2: render_tab_data()
    with tab3: render_tab_stationarity()
    with tab4: render_tab_model()
    with tab5: render_tab_diagnostics()
    with tab6: render_tab_forecast()


if __name__ == "__main__":
    main()
