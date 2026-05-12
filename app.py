"""
app.py
Aplikasi Prediksi Harga Bitcoin dengan ARIMA — Entry Point.

Aplikasi ini mengimplementasikan seluruh tahapan KDD (Knowledge Discovery in Databases)
sesuai dengan metodologi yang dijabarkan pada Bab 3 skripsi:
    1. Pengumpulan Data       → Tab Data
    2. Pra-pemrosesan         → Tab Data
    3. Uji Stasioneritas      → Tab Stasioneritas
    4. Identifikasi Model     → Tab Pemilihan Model
    5. Estimasi Parameter     → Tab Pemilihan Model
    6. Diagnostik Residual    → Tab Diagnostik
    7. Peramalan & Evaluasi   → Tab Peramalan
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
    page_title="Prediksi Bitcoin ARIMA",
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
        html, body, [class*="css"]  { font-family: 'Segoe UI', 'Calibri', sans-serif; }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f4f7fb 0%, #ffffff 30%);
        }
        .main .block-container {
            max-width: 1180px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] { border-right: 1px solid #e5eaf1; }
        .subtle-text { color: #4e5a6a; font-size: 0.95rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] { font-weight: 500; }
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
        st.markdown("### ⚙️ Kontrol Global")
        st.caption("Pengaturan berlaku untuk semua tab.")

        mode = st.radio(
            "Mode Model",
            options=MODE_OPTIONS,
            index=MODE_OPTIONS.index(st.session_state.get("mode", MODE_PRETRAINED)),
            help=(
                "**Pre-trained**: muat model yang sudah dilatih.\n\n"
                "**Train Ulang Manual**: latih ARIMA dengan (p,d,q) pilihan Anda.\n\n"
                "**Auto Grid Search**: cari (p,d,q) terbaik berdasarkan AIC."
            ),
        )
        st.session_state["mode"] = mode

        if mode == MODE_MANUAL:
            st.markdown("---")
            st.caption("**Parameter ARIMA Manual**")
            p = st.slider("p (AR)", 0, GRID_P_MAX, st.session_state.get("p", DEFAULT_P))
            d = st.slider("d (Differencing)", 0, GRID_D_MAX, st.session_state.get("d", DEFAULT_D))
            q = st.slider("q (MA)", 0, GRID_Q_MAX, st.session_state.get("q", DEFAULT_Q))
            st.session_state["p"], st.session_state["d"], st.session_state["q"] = p, d, q

        st.markdown("---")
        horizon = st.slider(
            "Horizon Prediksi (hari)",
            MIN_HORIZON, MAX_HORIZON,
            st.session_state.get("horizon", DEFAULT_HORIZON),
        )
        st.session_state["horizon"] = horizon

        st.markdown("---")
        if st.button("🔄 Refresh Data CoinGecko", use_container_width=True):
            for key in ["data", "trained_model", "model_meta", "model_signature"]:
                st.session_state.pop(key, None)
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.caption("**Sumber Data:** CoinGecko API")
        st.caption("**Metodologi:** KDD + Box-Jenkins")


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
                f"File model pre-trained tidak ditemukan: {PRETRAINED_MODEL_PATH.name}. "
                "Silakan pilih mode lain."
            )
            st.stop()
        with st.spinner("Memuat model pre-trained..."):
            base_model = load_pretrained_model_cached(str(PRETRAINED_MODEL_PATH))
            model = apply_pretrained_to_series(base_model, train)
        meta = {"mode": mode, "note": "Model pre-trained dimuat dari arima_model.pkl."}

    elif mode == MODE_MANUAL:
        p, d, q = st.session_state["p"], st.session_state["d"], st.session_state["q"]
        with st.spinner(f"Melatih ARIMA({p},{d},{q}) pada data latih..."):
            model = train_arima(train, (p, d, q))
        meta = {"mode": mode, "note": f"Model ARIMA({p},{d},{q}) dilatih ulang."}

    else:  # AUTO
        with st.spinner("Menentukan d optimal (uji ADF)..."):
            d_opt, _ = find_optimal_d(train)

        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        def cb(idx, total):
            progress_bar.progress(idx / total)
            progress_text.caption(f"Grid search... {idx}/{total} kombinasi")

        results_df = grid_search_arima(
            train, d=d_opt, p_max=GRID_P_MAX, q_max=GRID_Q_MAX, progress_callback=cb
        )
        progress_bar.empty()
        progress_text.empty()

        best = results_df.iloc[0]
        with st.spinner(
            f"Melatih model terbaik ARIMA({int(best['p'])},{int(best['d'])},{int(best['q'])})..."
        ):
            model = train_arima(train, (int(best["p"]), int(best["d"]), int(best["q"])))
        meta = {
            "mode": mode,
            "note": (
                f"Grid search menemukan ARIMA({int(best['p'])},{int(best['d'])},{int(best['q'])}) "
                f"dengan AIC={best['aic']:.2f}."
            ),
            "grid_results": results_df,
            "d_optimal": d_opt,
        }

    st.session_state["trained_model"] = model
    st.session_state["model_meta"] = meta
    st.session_state["model_signature"] = sig
    return model, meta


def fmt_usd(value: float) -> str:
    return f"${value:,.2f}"


# ============================================================
# TAB 1: BERANDA
# ============================================================
def render_tab_beranda():
    st.markdown("## 🏠 Beranda — Prediksi Cepat")
    st.markdown(
        "<p class='subtle-text'>Halaman utama untuk pengguna akhir. "
        "Pilih mode model di sidebar, atur horizon prediksi, lihat hasil prediksi.</p>",
        unsafe_allow_html=True,
    )

    data = st.session_state["data"]
    full, train = data["full"], data["train"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Harga Terakhir", fmt_usd(full.iloc[-1]))
    c2.metric("Periode Data", f"{full.index.min().date()} → {full.index.max().date()}")
    c3.metric("Jumlah Hari", f"{len(full)}")

    st.markdown("---")
    st.markdown("### 📈 Grafik 90 Hari Terakhir")
    st.plotly_chart(viz.plot_historical(full.tail(90).to_frame("close")), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔮 Hasil Prediksi")
    model, meta = get_active_model()
    st.caption(f"ℹ️ {meta['note']}")

    horizon = st.session_state["horizon"]
    with st.spinner(f"Membuat forecast {horizon} hari ke depan..."):
        mean, lower, upper = forecast_with_ci(model, horizon=horizon)
        forecast_df = build_forecast_dataframe(train.index[-1], mean, lower, upper)

    last_price = float(train.iloc[-1])
    next_price = float(forecast_df["forecast"].iloc[0])
    end_price = float(forecast_df["forecast"].iloc[-1])
    next_pct = ((next_price - last_price) / last_price) * 100
    end_pct = ((end_price - last_price) / last_price) * 100
    direction = "📈 Naik" if end_pct >= 0 else "📉 Turun"

    m1, m2, m3 = st.columns(3)
    m1.metric("Harga Terakhir", fmt_usd(last_price))
    m2.metric("Prediksi Hari Berikutnya", fmt_usd(next_price), f"{next_pct:+.2f}%")
    m3.metric(f"Arah Tren ({horizon} hari)", direction, f"{end_pct:+.2f}%")

    st.plotly_chart(viz.plot_forecast_with_ci(full, forecast_df), use_container_width=True)

    with st.expander("📋 Tabel Forecast Detail"):
        disp = forecast_df.reset_index().rename(columns={"index": "Tanggal"})
        disp["forecast"] = disp["forecast"].map(fmt_usd)
        disp["lower_95"] = disp["lower_95"].map(fmt_usd)
        disp["upper_95"] = disp["upper_95"].map(fmt_usd)
        disp.columns = ["Tanggal", "Prediksi", "CI Bawah 95%", "CI Atas 95%"]
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.warning(
        "⚠️ **Disclaimer:** Prediksi ini berdasarkan model statistik dan data harga historis. "
        "Bukan saran investasi. Keputusan investasi tetap menjadi tanggung jawab pengguna."
    )


# ============================================================
# TAB 2: DATA
# ============================================================
def render_tab_data():
    st.markdown("## 📊 Eksplorasi Data")
    st.markdown(
        "<p class='subtle-text'>Sesuai Subbab 3.5.1, 3.5.2, dan 4.1.</p>",
        unsafe_allow_html=True,
    )

    full = st.session_state["data"]["full"]
    train = st.session_state["data"]["train"]
    test = st.session_state["data"]["test"]

    st.markdown("### 📋 Statistik Deskriptif")
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
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.markdown("### 📈 Plot Historis")
    st.plotly_chart(viz.plot_historical(full.to_frame("close")), use_container_width=True)

    st.markdown("### 📊 Distribusi Harga")
    st.plotly_chart(viz.plot_distribution(full), use_container_width=True)

    st.markdown("### ✂️ Pembagian Train/Test (80:20)")
    st.json(get_split_summary(train, test))
    st.plotly_chart(viz.plot_train_test_split(train, test), use_container_width=True)


# ============================================================
# TAB 3: STASIONERITAS
# ============================================================
def render_tab_stationarity():
    st.markdown("## 🔬 Analisis Stasioneritas")
    st.markdown(
        "<p class='subtle-text'>Sesuai Subbab 3.5.3 dan 4.2.</p>",
        unsafe_allow_html=True,
    )

    train = st.session_state["data"]["train"]

    st.markdown("### Uji Augmented Dickey-Fuller (ADF) Iteratif")
    st.markdown(
        "**Hipotesis:**\n"
        "- H₀: data memiliki *unit root* (tidak stasioner)\n"
        "- H₁: data tidak memiliki *unit root* (stasioner)\n\n"
        "Differencing dilakukan hingga p-value < 0,05."
    )

    with st.spinner("Menjalankan uji ADF iteratif..."):
        d_opt, history = find_optimal_d(train)

    rows = []
    for d, res in history.items():
        rows.append({
            "Orde d": d,
            "ADF Statistic": f"{res['statistic']:.4f}",
            "p-value": f"{res['p_value']:.6f}",
            "Critical 5%": f"{res['critical_5pct']:.4f}",
            "n Observasi": res["n_obs"],
            "Status": "✅ Stasioner" if res["is_stationary"] else "❌ Tidak Stasioner",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.success(f"**Orde differencing optimal: d = {d_opt}**")

    st.markdown("### Visualisasi Differencing")
    if d_opt > 0:
        differenced = difference(train, d_opt)
        st.plotly_chart(
            viz.plot_differencing(train, differenced, d_opt), use_container_width=True
        )
    else:
        st.info("Data sudah stasioner pada level (d=0), tidak perlu differencing.")
        differenced = train

    st.markdown("### Analisis ACF dan PACF")
    st.caption("Pola cut-off / tailing-off pada plot ACF dan PACF memberikan indikasi awal orde p dan q.")
    n_lags = st.slider("Jumlah Lag", 10, 50, 30)
    acf_data = compute_acf_pacf(differenced if d_opt > 0 else train, n_lags=n_lags)
    st.plotly_chart(viz.plot_acf_pacf(acf_data), use_container_width=True)


# ============================================================
# TAB 4: PEMILIHAN MODEL
# ============================================================
def render_tab_model():
    st.markdown("## ⚙️ Pemilihan Model")
    st.markdown(
        "<p class='subtle-text'>Sesuai Subbab 3.5.4 dan 3.5.5. Mode model dipilih dari sidebar.</p>",
        unsafe_allow_html=True,
    )

    mode = st.session_state["mode"]
    st.info(f"**Mode aktif:** {mode}")

    model, meta = get_active_model()
    st.caption(f"ℹ️ {meta['note']}")

    if mode == MODE_AUTO and "grid_results" in meta:
        st.markdown("### Hasil Grid Search Berbasis AIC")
        st.markdown(f"Orde differencing: **d = {meta['d_optimal']}** (dari uji ADF).")
        st.markdown(f"Total kombinasi yang berhasil konvergen: **{len(meta['grid_results'])} model**.")

        display_grid = meta["grid_results"].head(10)[["order_str", "aic", "bic", "llf"]].copy()
        display_grid.columns = ["ARIMA(p,d,q)", "AIC", "BIC", "Log-Likelihood"]
        display_grid["AIC"] = display_grid["AIC"].map(lambda x: f"{x:.4f}")
        display_grid["BIC"] = display_grid["BIC"].map(lambda x: f"{x:.4f}")
        display_grid["Log-Likelihood"] = display_grid["Log-Likelihood"].map(lambda x: f"{x:.4f}")
        st.markdown("**Top 10 Model:**")
        st.dataframe(display_grid, use_container_width=True, hide_index=True)

    st.markdown("### Informasi Model Terpilih")
    info = get_model_info(model)
    info_df = pd.DataFrame([{"Parameter": k, "Nilai": v} for k, v in info.items()])
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    st.markdown("### Estimasi Koefisien Model")
    st.caption("Estimasi dengan MLE. Lihat Subbab 2.4.9 (teori) dan 4.3.3–4.3.4 (interpretasi).")
    try:
        coef_df = get_coefficients_df(model)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
    except Exception as err:
        st.warning(f"Tidak dapat menampilkan tabel koefisien: {err}")

    with st.expander("📄 Lihat Summary Lengkap (statsmodels)"):
        st.text(str(model.summary()))


# ============================================================
# TAB 5: DIAGNOSTIK
# ============================================================
def render_tab_diagnostics():
    st.markdown("## ✅ Diagnostik Residual")
    st.markdown(
        "<p class='subtle-text'>Sesuai Subbab 3.5.6 dan 4.4.</p>",
        unsafe_allow_html=True,
    )

    model, _ = get_active_model()
    diag = diagnostic_summary(model)

    st.markdown("### Statistik Deskriptif Residual")
    stats_df = pd.DataFrame([
        {"Statistik": k, "Nilai": (f"{v:.4f}" if isinstance(v, float) else v)}
        for k, v in diag["stats"].items()
    ])
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.markdown("### Uji Ljung-Box pada Multiple Lag")
    st.markdown(
        "**Hipotesis:**\n"
        "- H₀: residual tidak berautokorelasi (model memadai)\n"
        "- H₁: residual berautokorelasi"
    )
    lb_df = diag["ljung_box"].copy()
    lb_df["Statistik Q"] = lb_df["Statistik Q"].map(lambda x: f"{x:.4f}")
    lb_df["p-value"] = lb_df["p-value"].map(lambda x: f"{x:.4f}")
    st.dataframe(lb_df, use_container_width=True, hide_index=True)

    n_pass = (diag["ljung_box"]["p-value"] > 0.05).sum()
    total = len(diag["ljung_box"])
    if n_pass == total:
        st.success(f"✅ Residual tidak berautokorelasi pada seluruh {total} lag yang diuji.")
    else:
        st.warning(f"⚠️ Residual berautokorelasi pada {total - n_pass} dari {total} lag.")

    st.markdown("### Uji Normalitas (Jarque-Bera)")
    jb = diag["jarque_bera"]
    jb_df = pd.DataFrame([
        {"Statistik": "Statistik JB", "Nilai": f"{jb['Statistik JB']:.4f}"},
        {"Statistik": "p-value", "Nilai": f"{jb['p-value']:.6f}"},
        {"Statistik": "Skewness", "Nilai": f"{jb['Skewness']:.4f}"},
        {"Statistik": "Kurtosis", "Nilai": f"{jb['Kurtosis']:.4f} (normal=3)"},
    ])
    st.dataframe(jb_df, use_container_width=True, hide_index=True)
    if jb["is_normal"]:
        st.success(f"✅ {jb['conclusion']}")
    else:
        st.warning(
            f"⚠️ {jb['conclusion']}. Pelanggaran ringan terhadap normalitas umum pada "
            "data finansial dan tidak fatal untuk peramalan titik."
        )

    st.markdown("### Visualisasi Diagnostik (4-Panel)")
    st.plotly_chart(
        viz.plot_residual_diagnostics(diag["residuals_array"]),
        use_container_width=True,
    )


# ============================================================
# TAB 6: PERAMALAN
# ============================================================
def render_tab_forecast():
    st.markdown("## 🔮 Peramalan & Evaluasi")
    st.markdown(
        "<p class='subtle-text'>Sesuai Subbab 3.5.7 dan 4.5–4.6.</p>",
        unsafe_allow_html=True,
    )

    model, meta = get_active_model()
    train = st.session_state["data"]["train"]
    test = st.session_state["data"]["test"]
    full = st.session_state["data"]["full"]

    st.markdown("### Evaluasi pada Data Uji (One-Step Ahead Rolling)")
    st.caption("Model memprediksi 1 hari ke depan; setelah evaluasi, nilai aktual ditambahkan ke state.")

    with st.spinner("Menghasilkan prediksi rolling pada test set..."):
        predictions = rolling_one_step_forecast(model, test)

    naive_pred = naive_predictions(train, test)
    comparison = compare_with_naive(test.values, predictions, naive_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", fmt_usd(comparison["model"]["mae"]))
    c2.metric("RMSE", fmt_usd(comparison["model"]["rmse"]))
    c3.metric("MAPE", f"{comparison['model']['mape']:.2f}%")
    c4.metric("Kategori (Lewis)", comparison["model"]["category"])

    st.plotly_chart(
        viz.plot_actual_vs_predicted(
            train.iloc[-60:], test, predictions, mape_value=comparison["model"]["mape"]
        ),
        use_container_width=True,
    )

    st.markdown("### Perbandingan dengan Naive Baseline")
    st.markdown(
        "Model naive: $\\hat{Y}_t = Y_{t-1}$. "
        "Jika ARIMA tidak mengungguli naive, data mendekati *random walk*."
    )

    comp_df = pd.DataFrame({
        "Metrik": ["MAE", "RMSE", "MAPE"],
        "ARIMA": [
            fmt_usd(comparison["model"]["mae"]),
            fmt_usd(comparison["model"]["rmse"]),
            f"{comparison['model']['mape']:.4f}%",
        ],
        "Naive": [
            fmt_usd(comparison["naive"]["mae"]),
            fmt_usd(comparison["naive"]["rmse"]),
            f"{comparison['naive']['mape']:.4f}%",
        ],
        "Selisih (ARIMA - Naive)": [
            fmt_usd(comparison["model"]["mae"] - comparison["naive"]["mae"]),
            fmt_usd(comparison["model"]["rmse"] - comparison["naive"]["rmse"]),
            f"{comparison['model']['mape'] - comparison['naive']['mape']:+.4f}%",
        ],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    u = comparison["theils_u"]
    st.metric(
        "Theil's U Statistic", f"{u:.4f}",
        help="U < 1 = ARIMA lebih baik dari naive; U > 1 = naive lebih baik",
    )
    if u < 1:
        st.success(comparison["interpretation"])
    else:
        st.warning(comparison["interpretation"])

    st.plotly_chart(viz.plot_naive_comparison(test, predictions, naive_pred), use_container_width=True)

    st.markdown("---")
    st.markdown("### Peramalan Masa Depan")
    horizon = st.session_state["horizon"]

    if meta["mode"] == MODE_PRETRAINED:
        full_model = apply_pretrained_to_series(
            load_pretrained_model_cached(str(PRETRAINED_MODEL_PATH)), full
        )
    else:
        order = model.model.order
        with st.spinner(f"Re-fit ARIMA{order} pada seluruh data (train+test)..."):
            full_model = train_arima(full, order)

    mean, lower, upper = forecast_with_ci(full_model, horizon=horizon)
    forecast_df = build_forecast_dataframe(full.index[-1], mean, lower, upper)
    st.plotly_chart(viz.plot_forecast_with_ci(full, forecast_df), use_container_width=True)

    with st.expander("📋 Tabel Detail Forecast"):
        disp = forecast_df.reset_index().rename(columns={"index": "Tanggal"})
        disp["forecast"] = disp["forecast"].map(fmt_usd)
        disp["lower_95"] = disp["lower_95"].map(fmt_usd)
        disp["upper_95"] = disp["upper_95"].map(fmt_usd)
        disp.columns = ["Tanggal", "Prediksi", "CI Bawah 95%", "CI Atas 95%"]
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    apply_custom_style()

    st.title("📈 Prediksi Harga Bitcoin dengan ARIMA")
    st.markdown(
        "<p class='subtle-text'>Aplikasi pendukung skripsi: "
        "<em>Analisis Prediksi Harga Cryptocurrency Menggunakan Model ARIMA Berbasis Python</em>. "
        "Mengimplementasikan tahapan KDD lengkap dengan metodologi Box-Jenkins.</p>",
        unsafe_allow_html=True,
    )

    load_initial_data()
    sidebar_controls()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Beranda",
        "📊 Data",
        "🔬 Stasioneritas",
        "⚙️ Pemilihan Model",
        "✅ Diagnostik",
        "🔮 Peramalan",
    ])

    with tab1: render_tab_beranda()
    with tab2: render_tab_data()
    with tab3: render_tab_stationarity()
    with tab4: render_tab_model()
    with tab5: render_tab_diagnostics()
    with tab6: render_tab_forecast()


if __name__ == "__main__":
    main()
