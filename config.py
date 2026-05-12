"""
config.py
Konstanta global aplikasi prediksi harga Bitcoin dengan ARIMA.
"""
from pathlib import Path

# ============================================================
# Path & File
# ============================================================
APP_DIR = Path(__file__).resolve().parent
PRETRAINED_MODEL_PATH = APP_DIR / "arima_model.pkl"

# ============================================================
# API CoinGecko
# ============================================================
COINGECKO_OHLC_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
COINGECKO_DEFAULT_DAYS = 365
API_TIMEOUT = 30
CACHE_TTL_SECONDS = 900  # 15 menit

# ============================================================
# Data & Split
# ============================================================
MIN_HISTORY_DAYS = 200
MAX_HISTORY_DAYS = 365
DEFAULT_HISTORY_DAYS = 365
TRAIN_TEST_SPLIT_RATIO = 0.8

# ============================================================
# Model & Forecast
# ============================================================
MIN_HORIZON = 7
MAX_HORIZON = 30
DEFAULT_HORIZON = 14

# Default parameter manual
DEFAULT_P, DEFAULT_D, DEFAULT_Q = 2, 1, 2

# Range grid search
GRID_P_MAX = 5
GRID_Q_MAX = 5
GRID_D_MAX = 2

# Diagnostik
LJUNG_BOX_LAGS = [5, 10, 15, 20, 30]
CONFIDENCE_LEVEL = 0.05  # untuk CI 95%

# Kategori MAPE (Lewis 1982)
MAPE_CATEGORIES = [
    (10, "Sangat Baik"),
    (20, "Baik"),
    (50, "Cukup"),
    (float("inf"), "Tidak Akurat"),
]

# ============================================================
# Mode Model
# ============================================================
MODE_PRETRAINED = "Pre-trained (arima_model.pkl)"
MODE_MANUAL = "Train Ulang Manual"
MODE_AUTO = "Auto Grid Search (AIC)"
MODE_OPTIONS = [MODE_PRETRAINED, MODE_MANUAL, MODE_AUTO]

# ============================================================
# Visualisasi
# ============================================================
COLOR_PRIMARY = "#1d4ed8"      # biru
COLOR_SECONDARY = "#f97316"    # oranye
COLOR_ACTUAL = "#16a34a"       # hijau
COLOR_GRID = "#94a3b8"         # abu-abu
COLOR_DANGER = "#dc2626"       # merah
COLOR_CI_FILL = "rgba(249, 115, 22, 0.18)"  # oranye transparan

PLOT_TEMPLATE = "plotly_white"
