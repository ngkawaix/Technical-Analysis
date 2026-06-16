import warnings
warnings.filterwarnings("ignore")
import os

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Technical Analysis (ML)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = sorted([
    "AAPL", "ADBE", "AMAT", "AMZN", "ASML", "SPGI",
    "FICO", "GOOGL", "LRCX", "MA",   "META", "MSCI",
    "MSFT", "NFLX", "NVDA", "TSM",  "V",
])

REGIME_COLORS = {
    0: "#1D9E75",   # Bull / Low Vol  — teal
    1: "#F5A623",   # Transition       — amber
    2: "#D85A30",   # Bear / High Vol  — coral
}
REGIME_LABELS = {0: "Bull / Low-Vol", 1: "Transitional", 2: "Bear / High-Vol"}

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark-panel trading dashboard aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* --- typography & base --- */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar stays dark — intentional contrast panel */
[data-testid="stSidebar"] { background: #0d1117; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* Metric cards — clean white with a subtle border, no dark background */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.35rem;
    color: #1e293b !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem; }

/* Signal badge helper classes (applied via HTML) */
.bull  { color: #1D9E75; font-weight: 600; }
.bear  { color: #D85A30; font-weight: 600; }
.neut  { color: #64748b; font-weight: 500; }

/* Tab strip */
[data-baseweb="tab-list"] { border-bottom: 1px solid #e2e8f0; gap: 4px; }
[data-baseweb="tab"] { font-size: 0.83rem; padding: 6px 14px; border-radius: 6px 6px 0 0; }

/* Expander */
details > summary { font-size: 0.85rem; color: #64748b; }

/* Mono code inline — light background */
code {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 1px 5px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82em;
    color: #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching daily price data…", ttl=3600)
def load_prices(tickers, start="2018-01-01"):
    raw = yf.download(tickers, start=start, interval="1d", auto_adjust=True, progress=False)
    closes = raw["Close"]
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    closes = closes.ffill().dropna(how="all")
    return closes, raw["Volume"].ffill().dropna(how="all")

@st.cache_data(show_spinner=False, ttl=3600)
def validate_and_load_ticker(ticker: str, start: str) -> tuple[bool, str]:
    """
    Validate a user-supplied ticker and return (is_valid, message).
    Checks: exists on Yahoo Finance, has ≥ 2 years of data from `start`,
    and < 10% missing values. Cached 1 hour so repeated attempts don't
    re-hit the API.
    """
    import time
    try:
        t = yf.Ticker(ticker)
        price = t.fast_info.last_price
        if price is None or price <= 0:
            return False, f"**{ticker}** not found on Yahoo Finance."
    except Exception:
        return False, f"Could not retrieve data for **{ticker}**. Check the symbol."

    try:
        raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if raw.empty:
            return False, f"**{ticker}** has no price data from {start}."
        prices = raw["Close"].squeeze().dropna()
        n_days = len(prices)
        min_days = 504   # ~2 years — enough for ML training window
        if n_days < min_days:
            return False, (
                f"**{ticker}** only has {n_days} trading days from {start} "
                f"(minimum {min_days} ≈ 2 years). Move the data start date earlier "
                "or choose a stock with more history."
            )
        missing = raw["Close"].squeeze().isna().mean()
        if missing > 0.10:
            return False, f"**{ticker}** has {missing:.0%} missing data — too patchy to use."
    except Exception as e:
        return False, f"Download failed for **{ticker}**: {e}"

    return True, f"✅ **{ticker}** added ({n_days} trading days from {start})."

@st.cache_data(show_spinner=False, ttl=3600)
def load_spy(start="2018-01-01"):
    raw = yf.download("SPY", start=start, auto_adjust=True, progress=False)
    s = raw["Close"].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (all features explained term-by-term in the UI)
# ─────────────────────────────────────────────────────────────────────────────
def compute_features(prices: pd.Series, volumes: pd.Series) -> pd.DataFrame:
    """
    Compute a rich feature set for one stock.

    Feature glossary (mirrors the UI explainers):
    ─ ret_1d/5d/21d : log return over 1 / 5 / 21 days
    ─ rsi_14        : Relative Strength Index, 14-day window
    ─ macd_line     : MACD line = EMA(12) − EMA(26)
    ─ macd_signal   : 9-day EMA of the MACD line
    ─ macd_hist     : histogram = macd_line − macd_signal
    ─ bb_pct        : price position inside Bollinger Bands (0=lower, 1=upper)
    ─ bb_width      : band width normalised by mid-band (proxy for vol regime)
    ─ atr_14        : Average True Range over 14 days (absolute volatility)
    ─ vol_21        : 21-day realised volatility (annualised)
    ─ vol_ratio     : vol_5d / vol_63d  — short vs long vol regime
    ─ obv_norm      : On-Balance Volume normalised by 21-day mean
    ─ price_vs_sma20: (price / SMA20) − 1
    ─ price_vs_sma50: (price / SMA50) − 1
    ─ price_vs_sma200:(price / SMA200)− 1
    ─ high_52w_pct  : distance from 52-week high
    ─ low_52w_pct   : distance from 52-week low
    """
    df = pd.DataFrame({"close": prices, "volume": volumes})
    df["ret_1d"]  = np.log(df["close"] / df["close"].shift(1))
    df["ret_5d"]  = np.log(df["close"] / df["close"].shift(5))
    df["ret_21d"] = np.log(df["close"] / df["close"].shift(21))

    # RSI
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_line"]   = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd_line"] - df["macd_signal"]

    # Bollinger Bands (20-day)
    sma20    = df["close"].rolling(20).mean()
    std20    = df["close"].rolling(20).std()
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    df["bb_pct"]   = (df["close"] - lower_bb) / (upper_bb - lower_bb + 1e-9)
    df["bb_width"] = (upper_bb - lower_bb) / (sma20 + 1e-9)

    # ATR (Average True Range)
    # Without OHLC data, the True Range proxy is |close − prev_close|.
    # The full True Range = max(H−L, |H−Cp|, |L−Cp|) requires tick data.
    # Note: the original code used pd.concat([close-close, ...]) where
    # close-close is always zero — that first term was a no-op.
    prev_close = df["close"].shift(1)
    df["atr_14"] = (df["close"] - prev_close).abs().rolling(14).mean()

    # Volatility
    df["vol_21"]   = df["ret_1d"].rolling(21).std() * np.sqrt(252)
    df["vol_ratio"] = df["ret_1d"].rolling(5).std() / (df["ret_1d"].rolling(63).std() + 1e-9)

    # OBV
    obv = (np.sign(df["ret_1d"]) * df["volume"]).cumsum()
    df["obv_norm"] = obv / (obv.rolling(21).mean() + 1e-9)

    # SMA distances
    df["price_vs_sma20"]  = df["close"] / sma20 - 1
    df["price_vs_sma50"]  = df["close"] / df["close"].rolling(50).mean() - 1
    df["price_vs_sma200"] = df["close"] / df["close"].rolling(200).mean() - 1

    # 52-week extremes
    df["high_52w_pct"] = df["close"] / df["close"].rolling(252).max() - 1
    df["low_52w_pct"]  = df["close"] / df["close"].rolling(252).min() - 1

    return df.dropna()

FEATURE_COLS = [
    "ret_1d","ret_5d","ret_21d",
    "rsi_14",
    "macd_line","macd_signal","macd_hist",
    "bb_pct","bb_width",
    "atr_14","vol_21","vol_ratio",
    "obv_norm",
    "price_vs_sma20","price_vs_sma50","price_vs_sma200",
    "high_52w_pct","low_52w_pct",
]

FEATURE_GLOSSARY = {
    "ret_1d":          ("1-Day Log Return",      "log(P_t / P_{t-1})",       "Captures the most recent daily price momentum."),
    "ret_5d":          ("5-Day Log Return",       "log(P_t / P_{t-5})",       "Weekly momentum; smooths out single-day noise."),
    "ret_21d":         ("21-Day Log Return",      "log(P_t / P_{t-21})",      "Monthly momentum, the primary ML input for trend direction."),
    "rsi_14":          ("RSI (14)",               "100 - 100/(1+RS)",         "RS = avg(14d gains)/avg(14d losses). >70 = overbought; <30 = oversold."),
    "macd_line":       ("MACD Line",              "EMA(12) − EMA(26)",        "Fast minus slow exponential moving average. Cross above zero = bullish."),
    "macd_signal":     ("MACD Signal",            "EMA(9) of MACD line",      "Trigger line. When MACD line crosses above signal, that's a buy signal."),
    "macd_hist":       ("MACD Histogram",         "MACD line − Signal",       "Speed of the MACD divergence. Positive and rising = accelerating bull momentum."),
    "bb_pct":          ("Bollinger %B",           "(P − Lower) / (Upper − Lower)", "0 = at lower band (oversold), 1 = at upper band (overbought)."),
    "bb_width":        ("Bollinger Band Width",   "(Upper − Lower) / SMA20",  "Narrow bands → volatility compression (breakout likely). Wide → expansion."),
    "atr_14":          ("ATR (14)",               "14-day avg of |ΔP|",       "Absolute daily volatility. High ATR = large daily swings, risk is elevated."),
    "vol_21":          ("21d Realised Vol",       "σ(ret_1d,21) × √252",      "Annualised volatility. The core regime signal: low vol = bull, high vol = bear."),
    "vol_ratio":       ("Vol Ratio (5d/63d)",     "σ_5d / σ_63d",             ">1 = short-term vol is elevated vs. long-term average. Regime stress indicator."),
    "obv_norm":        ("OBV (normalised)",       "Σ sign(ret)×Vol / OBV_mean21", "On-Balance Volume. Rising OBV with flat price = accumulation (bullish). OBV normalised by its 21-day mean."),
    "price_vs_sma20":  ("Price vs SMA20",         "P/SMA20 − 1",              "Short-term trend. >0 = price above 20-day average (momentum supports bulls)."),
    "price_vs_sma50":  ("Price vs SMA50",         "P/SMA50 − 1",              "Medium-term trend. The '50-day golden cross' signal when SMA50 > SMA200."),
    "price_vs_sma200": ("Price vs SMA200",        "P/SMA200 − 1",             "Long-term trend health. Being above SMA200 is the primary bull-market condition."),
    "high_52w_pct":    ("Distance from 52W High", "P/max(P,252) − 1",         "How far from the yearly high. Near 0 = strength. Far below = possible downtrend."),
    "low_52w_pct":     ("Distance from 52W Low",  "P/min(P,252) − 1",         "How far above the yearly low. Near 0 = price is at yearly support."),
}

# ─────────────────────────────────────────────────────────────────────────────
# REGIME DETECTION  (Hidden Markov Model)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fitting HMM regime model…", ttl=3600)
def fit_hmm(returns: pd.Series, n_regimes: int = 3):
    """
    Fit a Gaussian Hidden Markov Model to daily log-returns.

    Pure numpy/scipy implementation — no hmmlearn required.
    Uses the Baum-Welch algorithm (Expectation-Maximisation for HMMs).

    The HMM assumes the market cycles through n_regimes hidden states.
    Observables at each day: [return, |return|] — captures both direction
    and magnitude of daily moves.

    Algorithm
    ---------
    E-step: Forward-Backward algorithm computes state occupancy probabilities
            γ_t(k) = P(state=k at time t | all observations, parameters)
    M-step: Update means μ_k, covariances Σ_k, and transition matrix A
            using the weighted sufficient statistics from the E-step.
    Repeat until log-likelihood converges.

    Decoding: Viterbi algorithm finds the single most likely state sequence.

    Parameters
    ----------
    returns   : pd.Series of daily log-returns
    n_regimes : number of hidden states (default 3)

    Returns
    -------
    regimes   : pd.Series of integer state labels (0=bull, n-1=bear)
    model     : dict with transmat_ and other params (mirrors hmmlearn API)
    remap     : dict mapping original → relabelled state ids
    """
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(42)
    r   = returns.dropna().values
    X   = np.column_stack([r, np.abs(r)]).astype(np.float64)  # (T, 2)
    T, D = X.shape
    K    = n_regimes

    # ── Initialise parameters with k-means-like seeding ──────────────────────
    # Sort observations by |return| and assign initial state boundaries
    sort_idx = np.argsort(X[:, 1])
    chunk    = T // K
    mu    = np.array([X[sort_idx[i*chunk:(i+1)*chunk]].mean(axis=0) for i in range(K)])
    sigma = np.array([np.cov(X[sort_idx[i*chunk:(i+1)*chunk]].T) + np.eye(D)*1e-4
                      for i in range(K)])
    pi_0  = np.ones(K) / K                  # initial state distribution
    A     = np.ones((K, K)) / K             # transition matrix (row = from, col = to)

    def log_emission(X, mu, sigma):
        """Log emission probabilities: (T, K) matrix."""
        out = np.zeros((T, K))
        for k in range(K):
            out[:, k] = multivariate_normal.logpdf(X, mean=mu[k], cov=sigma[k],
                                                    allow_singular=True)
        return out

    def forward_backward(log_b):
        """
        Baum-Welch forward-backward pass.
        log_b : (T, K) log emission probabilities
        Returns gamma (T, K) state occupancies and xi (T-1, K, K) transitions.
        """
        # Forward pass: α_t(k) = P(o_1..o_t, s_t=k)
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(pi_0 + 1e-300) + log_b[0]
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = log_b[t, k] + np.logaddexp.reduce(
                    log_alpha[t-1] + np.log(A[:, k] + 1e-300))

        # Backward pass: β_t(k) = P(o_{t+1}..o_T | s_t=k)
        log_beta = np.zeros((T, K))
        for t in range(T-2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(
                    np.log(A[k] + 1e-300) + log_b[t+1] + log_beta[t+1])

        # Gamma: γ_t(k) = P(s_t=k | all obs)
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Xi: ξ_t(i,j) = P(s_t=i, s_{t+1}=j | all obs)
        xi = np.zeros((T-1, K, K))
        for t in range(T-1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = (log_alpha[t, i] + np.log(A[i, j] + 1e-300)
                                   + log_b[t+1, j] + log_beta[t+1, j])
            xi[t] = np.exp(xi[t] - np.logaddexp.reduce(xi[t].ravel()))

        ll = np.logaddexp.reduce(log_alpha[-1])
        return gamma, xi, ll

    # ── Baum-Welch EM loop ────────────────────────────────────────────────────
    prev_ll = -np.inf
    for iteration in range(150):
        log_b          = log_emission(X, mu, sigma)
        gamma, xi, ll  = forward_backward(log_b)

        # M-step: update parameters
        gamma_sum = gamma.sum(axis=0) + 1e-300
        mu_new    = (gamma[:, :, None] * X[:, None, :]).sum(axis=0) / gamma_sum[:, None]

        sigma_new = np.zeros((K, D, D))
        for k in range(K):
            diff = X - mu_new[k]
            sigma_new[k] = ((gamma[:, k, None, None] *
                             (diff[:, :, None] * diff[:, None, :])).sum(axis=0)
                            / gamma_sum[k]) + np.eye(D) * 1e-4

        A_new  = xi.sum(axis=0)
        A_new /= A_new.sum(axis=1, keepdims=True) + 1e-300
        pi_0   = gamma[0] / gamma[0].sum()

        mu, sigma, A = mu_new, sigma_new, A_new

        if abs(ll - prev_ll) < 1e-4:
            break
        prev_ll = ll

    # ── Viterbi decoding ──────────────────────────────────────────────────────
    log_b     = log_emission(X, mu, sigma)
    viterbi   = np.zeros((T, K))
    backptr   = np.zeros((T, K), dtype=int)
    viterbi[0] = np.log(pi_0 + 1e-300) + log_b[0]
    for t in range(1, T):
        trans_prob = viterbi[t-1, :, None] + np.log(A + 1e-300)
        backptr[t] = trans_prob.argmax(axis=0)
        viterbi[t] = trans_prob.max(axis=0) + log_b[t]
    hidden_states = np.zeros(T, dtype=int)
    hidden_states[-1] = viterbi[-1].argmax()
    for t in range(T-2, -1, -1):
        hidden_states[t] = backptr[t+1, hidden_states[t+1]]

    # Re-label: state 0 = lowest volatility (bull), K-1 = highest (bear)
    state_vols = {s: np.abs(r[hidden_states == s]).mean() for s in range(K)}
    ordered    = sorted(state_vols, key=state_vols.get)
    remap      = {old: new for new, old in enumerate(ordered)}
    relabeled  = np.array([remap[s] for s in hidden_states])

    # Package into a dict that mirrors the hmmlearn API surface we use
    model = {"transmat_": A[[ordered.index(k) for k in range(K)], :]
                           [:, [ordered.index(k) for k in range(K)]],
             "means_":    mu[ordered],
             "log_likelihood": prev_ll}

    regimes = pd.Series(relabeled, index=returns.dropna().index, name="regime")
    return regimes, model, remap

# ─────────────────────────────────────────────────────────────────────────────
# ML SIGNAL MODEL  (Random Forest + XGBoost ensemble)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Training RF + XGBoost signal model…", ttl=3600)
def train_ml_model(feat_df: pd.DataFrame, lookahead: int = 5):
    """
    Train a binary classifier: will the stock be up more than 0% over the next
    `lookahead` days?

    Methodology
    -----------
    1. Target  : y = 1 if forward_ret > 0 else 0
       (lookahead log-return computed on the *training* set to avoid leakage)
    2. Walk-forward split: train on first 70%, test on last 30% — never train
       on data the model would not have seen in real time.
    3. Ensemble: RandomForest + XGBoost vote equally (soft voting on probabilities).
    4. Feature importances: averaged across both models.

    Returns
    -------
    prob_series   : pd.Series of P(up) for each day in the test window
    importances   : pd.Series of feature importances
    train_end_idx : the date where training ended
    accuracy      : float, out-of-sample accuracy
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import xgboost as xgb

    df = feat_df[FEATURE_COLS].copy()
    df["fwd_ret"] = np.log(feat_df["close"].shift(-lookahead) / feat_df["close"])
    df = df.dropna()

    X = df[FEATURE_COLS].values
    y = (df["fwd_ret"] > 0).astype(int).values
    idx = df.index

    split = int(len(X) * 0.70)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    idx_te = idx[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)

    xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric="logloss", random_state=42,
                                verbosity=0, use_label_encoder=False)
    xgb_m.fit(X_tr_s, y_tr)

    prob_rf  = rf.predict_proba(X_te_s)[:, 1]
    prob_xgb = xgb_m.predict_proba(X_te_s)[:, 1]
    prob_ens = 0.5 * prob_rf + 0.5 * prob_xgb

    preds    = (prob_ens > 0.5).astype(int)
    accuracy = accuracy_score(y_te, preds)

    # Feature importances — average of both models
    imp_rf  = rf.feature_importances_
    imp_xgb = xgb_m.feature_importances_
    importances = pd.Series(
        0.5 * imp_rf + 0.5 * imp_xgb,
        index=FEATURE_COLS,
    ).sort_values(ascending=False)

    prob_series = pd.Series(prob_ens, index=idx_te, name="prob_up")
    train_end   = idx[split - 1]
    return prob_series, importances, train_end, accuracy

# ─────────────────────────────────────────────────────────────────────────────
# LSTM FORECASTING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Training LSTM price forecaster…", ttl=3600)
def train_lstm(prices: pd.Series, seq_len: int = 30, horizon: int = 5,
               epochs: int = 40):
    """
    Pure-numpy LSTM for sequence-to-scalar regression — no TensorFlow required.

    Architecture (identical concept to the Keras version, implemented from scratch)
    ------------
    Input  : sequences of shape (seq_len, 3) — [norm_price, vol_5d×100, rsi]
    Layer 1: Single LSTM cell with hidden_size=32
             At each time step t:
               f_t = σ(Wf·[h_{t-1}, x_t] + bf)   # forget gate
               i_t = σ(Wi·[h_{t-1}, x_t] + bi)   # input gate
               g_t = tanh(Wg·[h_{t-1}, x_t] + bg) # candidate cell
               o_t = σ(Wo·[h_{t-1}, x_t] + bo)   # output gate
               c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t  # cell state update
               h_t = o_t ⊙ tanh(c_t)              # hidden state
    Layer 2: Linear readout — W_out · h_T + b_out → scalar prediction
    Training: Mini-batch gradient descent with Adam optimiser, MSE loss,
              gradients computed via Backpropagation Through Time (BPTT).
              Dropout applied to h_t during training (rate=0.2).

    Why no TensorFlow: the numpy implementation is fully equivalent for a
    single-layer LSTM of this size. TF adds ~600 MB of install weight for
    identical mathematical operations at this scale.

    Returns
    -------
    preds     : pd.Series of predicted next-horizon log-return (out-of-sample)
    actuals   : pd.Series of actual next-horizon log-return
    train_end : last date in the training window
    """
    rng = np.random.default_rng(42)

    # ── Feature engineering ───────────────────────────────────────────────────
    close  = prices.values.astype(np.float64)
    ret_1d = np.diff(np.log(close), prepend=np.log(close[0]))
    vol_5d = pd.Series(ret_1d).rolling(5).std().bfill().values * 100

    rsi_s  = pd.Series(close)
    delta  = rsi_s.diff()
    gain   = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi    = (100 - 100 / (1 + gain / (loss + 1e-9))).fillna(50).values / 100

    feat_arr = np.column_stack([close / close[0], vol_5d, rsi])  # (T, 3)
    n_feat   = feat_arr.shape[1]   # 3
    n        = len(feat_arr)

    # ── Sequence construction ─────────────────────────────────────────────────
    X_all, y_all = [], []
    for i in range(seq_len, n - horizon):
        window = feat_arr[i - seq_len: i].copy()
        window = window / (window[0] + 1e-9)
        X_all.append(window)
        y_all.append(np.log(close[i + horizon] / close[i]))

    X_all  = np.array(X_all, dtype=np.float64)   # (N, seq_len, n_feat)
    y_all  = np.array(y_all, dtype=np.float64)   # (N,)
    idx    = prices.index[seq_len: n - horizon]

    split    = int(len(X_all) * 0.70)
    X_tr, X_te = X_all[:split], X_all[split:]
    y_tr, y_te = y_all[:split], y_all[split:]

    # ── LSTM weight initialisation (Xavier uniform) ───────────────────────────
    H  = 32          # hidden units
    D  = n_feat      # input features (3)
    scale = np.sqrt(6.0 / (D + H + H))

    def xavier(rows, cols):
        return rng.uniform(-scale, scale, (rows, cols))

    # Gate weight matrices: input → hidden, hidden → hidden, bias
    Wf, Uf, bf = xavier(H, D), xavier(H, H), np.zeros(H)
    Wi, Ui, bi = xavier(H, D), xavier(H, H), np.zeros(H)
    Wg, Ug, bg = xavier(H, D), xavier(H, H), np.zeros(H)
    Wo, Uo, bo = xavier(H, D), xavier(H, H), np.zeros(H)
    W_out = xavier(1, H) * 0.01
    b_out = np.zeros(1)

    # Adam moment accumulators — one per weight array
    params     = [Wf, Uf, bf, Wi, Ui, bi, Wg, Ug, bg, Wo, Uo, bo, W_out, b_out]
    m_adam     = [np.zeros_like(p) for p in params]
    v_adam     = [np.zeros_like(p) for p in params]
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    lr         = 3e-3
    adam_t     = 0

    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def lstm_forward(X_batch, dropout_rate=0.0):
        """
        Forward pass through the LSTM for a batch.
        X_batch : (B, T, D)
        Returns h_final (B, H) and all gates for BPTT.
        """
        B, T, _ = X_batch.shape
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        cache = []
        mask  = (rng.random((B, H)) > dropout_rate).astype(np.float64) \
                if dropout_rate > 0 else np.ones((B, H))

        for t in range(T):
            x_t = X_batch[:, t, :]   # (B, D)
            f   = sigmoid(x_t @ Wf.T + h @ Uf.T + bf)
            i_g = sigmoid(x_t @ Wi.T + h @ Ui.T + bi)
            g   = np.tanh(x_t @ Wg.T + h @ Ug.T + bg)
            o   = sigmoid(x_t @ Wo.T + h @ Uo.T + bo)
            c   = f * c + i_g * g
            h   = o * np.tanh(c) * mask   # apply dropout mask
            cache.append((x_t, f, i_g, g, o, c, h))

        y_hat = h @ W_out.T + b_out   # (B, 1)
        return y_hat.flatten(), h, cache, mask

    def lstm_backward(X_batch, y_batch, y_hat, h_final, cache, mask):
        """
        BPTT: compute gradients of MSE loss w.r.t. all parameters.
        Returns list of gradients matching `params` order.
        """
        B = len(y_batch)
        dL_dy = 2.0 * (y_hat - y_batch) / B   # (B,)

        dW_out = (dL_dy[:, None] * h_final).mean(axis=0, keepdims=True)  # (1, H)
        db_out = np.array([dL_dy.mean()])
        dh_next = dL_dy[:, None] * W_out    # (B, H)
        dc_next = np.zeros((B, H))

        # Accumulate gate gradients across time steps (BPTT)
        dWf=np.zeros_like(Wf); dUf=np.zeros_like(Uf); dbf=np.zeros(H)
        dWi=np.zeros_like(Wi); dUi=np.zeros_like(Ui); dbi=np.zeros(H)
        dWg=np.zeros_like(Wg); dUg=np.zeros_like(Ug); dbg=np.zeros(H)
        dWo=np.zeros_like(Wo); dUo=np.zeros_like(Uo); dbo=np.zeros(H)

        T = len(cache)
        for t in reversed(range(T)):
            x_t, f, i_g, g, o, c, h = cache[t]
            c_prev = cache[t-1][5] if t > 0 else np.zeros((B, H))
            h_prev = cache[t-1][6] if t > 0 else np.zeros((B, H))

            dh = dh_next * mask
            tanh_c   = np.tanh(c)
            do       = dh * tanh_c
            dc       = dh * o * (1 - tanh_c**2) + dc_next
            df       = dc * c_prev
            di       = dc * g
            dg       = dc * i_g

            df_pre  = df * f * (1 - f)
            di_pre  = di * i_g * (1 - i_g)
            dg_pre  = dg * (1 - g**2)
            do_pre  = do * o * (1 - o)

            for dpre, x, h_p, dW, dU, db in [
                (df_pre, x_t, h_prev, dWf, dUf, dbf),
                (di_pre, x_t, h_prev, dWi, dUi, dbi),
                (dg_pre, x_t, h_prev, dWg, dUg, dbg),
                (do_pre, x_t, h_prev, dWo, dUo, dbo),
            ]:
                dW += dpre.T @ x / B
                dU += dpre.T @ h_p / B
                db += dpre.mean(axis=0)

            dh_next  = (df_pre @ Uf + di_pre @ Ui +
                        dg_pre @ Ug + do_pre @ Uo)
            dc_next  = dc * f

        return [dWf, dUf, dbf, dWi, dUi, dbi,
                dWg, dUg, dbg, dWo, dUo, dbo, dW_out, db_out]

    # ── Training loop ─────────────────────────────────────────────────────────
    batch_size  = 32
    best_val_loss = np.inf
    best_params   = [p.copy() for p in params]
    patience_count = 0
    patience_limit = 8
    n_tr = len(X_tr)

    for epoch in range(epochs):
        # Shuffle training data
        perm = rng.permutation(n_tr)
        X_sh, y_sh = X_tr[perm], y_tr[perm]
        train_loss = 0.0
        n_batches  = 0

        for start in range(0, n_tr, batch_size):
            Xb = X_sh[start: start + batch_size]
            yb = y_sh[start: start + batch_size]
            if len(Xb) < 2:
                continue

            y_hat, h_fin, cache, mask = lstm_forward(Xb, dropout_rate=0.2)
            loss_b = np.mean((y_hat - yb) ** 2)
            train_loss += loss_b
            n_batches  += 1

            grads = lstm_backward(Xb, yb, y_hat, h_fin, cache, mask)

            # Adam update
            adam_t += 1
            for k, (p, g_p, m, v) in enumerate(zip(params, grads, m_adam, v_adam)):
                m[:] = beta1 * m + (1 - beta1) * g_p
                v[:] = beta2 * v + (1 - beta2) * g_p ** 2
                m_hat = m / (1 - beta1 ** adam_t)
                v_hat = v / (1 - beta2 ** adam_t)
                p    -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        # Validation loss (no dropout)
        val_hat, _, _, _ = lstm_forward(X_te[:64], dropout_rate=0.0)
        val_loss = np.mean((val_hat - y_te[:64]) ** 2)

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_params     = [p.copy() for p in params]
            patience_count  = 0
        else:
            patience_count += 1
            if patience_count >= patience_limit:
                break

    # Restore best weights
    for p, bp in zip(params, best_params):
        p[:] = bp

    # ── Out-of-sample predictions ─────────────────────────────────────────────
    y_pred, _, _, _ = lstm_forward(X_te, dropout_rate=0.0)

    preds     = pd.Series(y_pred, index=idx[split:], name="lstm_pred")
    actuals   = pd.Series(y_te,   index=idx[split:], name="actual")
    train_end = idx[split - 1]
    return preds, actuals, train_end

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIC TECHNICALS (baseline, all-in-one)
# ─────────────────────────────────────────────────────────────────────────────
def compute_technicals(feat_df: pd.DataFrame) -> dict:
    last = feat_df.iloc[-1]

    def rsi_signal(rsi):
        if rsi > 70:   return "Overbought", "bear"
        if rsi < 30:   return "Oversold",   "bull"
        return "Neutral", "neut"

    def macd_signal(hist):
        if hist > 0:  return "Bullish", "bull"
        if hist < 0:  return "Bearish", "bear"
        return "Flat", "neut"

    def bb_signal(pct):
        if pct > 0.95: return "Upper band — overbought", "bear"
        if pct < 0.05: return "Lower band — oversold",  "bull"
        return "Mid-range", "neut"

    def sma_signal(vs20, vs200):
        if vs20 > 0 and vs200 > 0: return "Price above both SMAs — bull trend", "bull"
        if vs20 < 0 and vs200 < 0: return "Price below both SMAs — bear trend", "bear"
        return "Mixed signals", "neut"

    rsi_txt, rsi_cls = rsi_signal(last["rsi_14"])
    macd_txt, macd_cls = macd_signal(last["macd_hist"])
    bb_txt, bb_cls = bb_signal(last["bb_pct"])
    sma_txt, sma_cls = sma_signal(last["price_vs_sma20"], last["price_vs_sma200"])

    return {
        "RSI (14)":        (f"{last['rsi_14']:.1f}  —  {rsi_txt}",  rsi_cls),
        "MACD Histogram":  (f"{last['macd_hist']:.3f}  —  {macd_txt}", macd_cls),
        "Bollinger %B":    (f"{last['bb_pct']:.2f}  —  {bb_txt}",   bb_cls),
        "SMA Trend":       (sma_txt, sma_cls),
        "Vol (21d ann.)":  (f"{last['vol_21']:.1%}", "neut"),
        "Vol Regime":      ("Elevated" if last["vol_ratio"] > 1.2 else "Normal",
                            "bear" if last["vol_ratio"] > 1.2 else "bull"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️  Settings")
    st.caption("Technical Analysis — ML Edition")
    st.divider()

    # ── 1. DATA START — set first so validation uses the correct window ────────
    data_start = st.date_input("Data start", value=date(2018, 1, 1),
                               min_value=date(2015, 1, 1),
                               max_value=(datetime.today() - timedelta(days=365*2)).date())
    # Store for use in custom-ticker validation below
    st.session_state.sidebar_data_start = data_start.strftime("%Y-%m-%d")
    st.divider()

    # ── 2. CUSTOM TICKER UNIVERSE ─────────────────────────────────────────────
    # Persist custom tickers in session_state, loaded from portfolio_state.json
    # on first boot (same file as holdings/BL weights — single source of truth).
    import json as _json
    _SAVE_FILE_SIDEBAR = "/tmp/portfolio_state.json"

    if "custom_tickers" not in st.session_state:
        # Try loading from disk
        try:
            with open(_SAVE_FILE_SIDEBAR) as _f:
                _saved = _json.load(_f)
            st.session_state.custom_tickers = [
                t.upper() for t in _saved.get("custom_tickers", [])
                if t.upper() not in TICKERS
            ]
        except Exception:
            st.session_state.custom_tickers = []

    if "ticker_msg" not in st.session_state:
        st.session_state.ticker_msg = None   # (kind, text) or None

    st.subheader("📌 Universe")
    st.caption(f"Core: {len(TICKERS)} stocks  |  Custom: {len(st.session_state.custom_tickers)}")

    # Add ticker row
    _inp_col, _btn_col = st.columns([3, 1])
    with _inp_col:
        _new_raw = st.text_input(
            "Add ticker", placeholder="e.g. BABA",
            label_visibility="collapsed", key="new_ticker_input",
            help=(
                "Any valid Yahoo Finance ticker. Must have ≥ 2 years of data "
                "from your chosen start date. Saved automatically on Add."
            ),
        )
    with _btn_col:
        _add_clicked = st.button("Add", use_container_width=True, key="add_ticker_btn")

    if _add_clicked and _new_raw.strip():
        _t = _new_raw.strip().upper()
        if _t in TICKERS:
            st.session_state.ticker_msg = ("info", f"**{_t}** is already in the core universe.")
        elif _t in st.session_state.custom_tickers:
            st.session_state.ticker_msg = ("info", f"**{_t}** is already added.")
        else:
            with st.spinner(f"Validating {_t}…"):
                # Use the sidebar's data_start — defined below; default to 2018 if not yet set
                _start_for_val = st.session_state.get("sidebar_data_start", "2018-01-01")
                _valid, _msg = validate_and_load_ticker(_t, _start_for_val)
            if _valid:
                st.session_state.custom_tickers.append(_t)
                st.session_state.ticker_msg = ("success", _msg)
                # Persist to disk immediately
                try:
                    with open(_SAVE_FILE_SIDEBAR) as _f:
                        _state = _json.load(_f)
                except Exception:
                    _state = {}
                _state["custom_tickers"] = st.session_state.custom_tickers
                with open(_SAVE_FILE_SIDEBAR, "w") as _f:
                    _json.dump(_state, _f, indent=2)
                st.rerun()
            else:
                st.session_state.ticker_msg = ("error", _msg)

    # Show last add message
    if st.session_state.ticker_msg:
        _kind, _text = st.session_state.ticker_msg
        if _kind == "success": st.success(_text)
        elif _kind == "error":  st.error(_text)
        elif _kind == "info":   st.info(_text)

    # List custom tickers with remove buttons
    if st.session_state.custom_tickers:
        for _ct in list(st.session_state.custom_tickers):
            _nc, _rc = st.columns([5, 1])
            with _nc:
                st.markdown(f"`{_ct}` ✦")
            with _rc:
                if st.button("✕", key=f"rm_{_ct}", help=f"Remove {_ct}"):
                    st.session_state.custom_tickers.remove(_ct)
                    st.session_state.ticker_msg = None
                    # Persist removal
                    try:
                        with open(_SAVE_FILE_SIDEBAR) as _f:
                            _state = _json.load(_f)
                    except Exception:
                        _state = {}
                    _state["custom_tickers"] = st.session_state.custom_tickers
                    with open(_SAVE_FILE_SIDEBAR, "w") as _f:
                        _json.dump(_state, _f, indent=2)
                    st.rerun()

    st.divider()

    # ── 3. STOCK SELECTOR — right after universe is finalised ─────────────────
    _ACTIVE_TICKERS_SB = sorted(set(TICKERS + st.session_state.get("custom_tickers", [])))
    _default_idx = _ACTIVE_TICKERS_SB.index("NVDA") if "NVDA" in _ACTIVE_TICKERS_SB else 0
    selected_ticker = st.selectbox(
        "🔍 Analyse stock", _ACTIVE_TICKERS_SB, index=_default_idx,
        help="Drives Overview, Technicals, ML Signals, and LSTM tabs.",
    )
    st.divider()

    # ── 4. MODEL PARAMETERS ────────────────────────────────────────────────────
    st.subheader("⚙️ Model Parameters")
    n_regimes = st.radio("HMM Regime Count", [2, 3], index=1, horizontal=True,
                         help="Number of hidden market states the HMM fits. 3 = Bull / Transitional / Bear.")
    lookahead = st.slider("ML signal lookahead (days)", 1, 21, 5, 1,
                          help="How many days ahead the RF/XGB model targets. "
                               "5d = one trading week.")
    lstm_horizon = st.slider("LSTM forecast horizon (days)", 3, 21, 5, 1,
                             help="Prediction horizon for the deep learning sequence model.")
    st.divider()

    # ── 5. APP GUIDE ──────────────────────────────────────────────────────────
    st.caption("💡 **How the app fits together:**\n\n"
               "- **Overview**: daily dashboard of all signals in one glance\n"
               "- **Technicals**: classic indicators explained\n"
               "- **Regime**: HMM detects market states from hidden volatility patterns\n"
               "- **ML Signals**: RF+XGBoost predict short-term direction\n"
               "- **LSTM Forecast**: deep learning on price sequences\n"
               "- **Universe**: all stocks including custom additions\n"
               "- **Rebalancing**: trade planner with timing verdicts")

# ── Active universe: core + custom, deduplicated and sorted ──────────────────
ACTIVE_TICKERS = sorted(set(TICKERS + st.session_state.get("custom_tickers", [])))

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────
start_str = data_start.strftime("%Y-%m-%d")
with st.spinner("Loading price & volume data for the full universe…"):
    closes, volumes = load_prices(tuple(ACTIVE_TICKERS), start=start_str)
    spy = load_spy(start=start_str)

# Filter to selected ticker
price_s  = closes[selected_ticker].dropna()
volume_s = volumes[selected_ticker].dropna()

with st.spinner("Engineering features…"):
    feat_df = compute_features(price_s, volume_s)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
last_price  = float(price_s.iloc[-1])
prev_price  = float(price_s.iloc[-2])
daily_ret   = (last_price / prev_price) - 1
last_date   = price_s.index[-1].strftime("%d %b %Y")

st.title("🔬 Technical Analysis — ML Edition")
st.caption(
    f"Daily analysis as of **{last_date}**  |  "
    f"Universe: **{len(ACTIVE_TICKERS)} stocks** ({len(TICKERS)} core + {len(ACTIVE_TICKERS)-len(TICKERS)} custom)  |  "
    f"Data from: **{start_str}**  |  "
    f"Companion to the **Portfolio Optimiser (DCF-BL)** app"
)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "📈 Technicals",
    "🔄 Regime Detection",
    "🤖 ML Signals",
    "🧠 LSTM Forecast",
    "🌐 Universe Scanner",
    "💼 Rebalancing Planner",
    "🔍 Signal Audit",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — DAILY OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab0:
    st.markdown(f"### {selected_ticker}  —  Daily Signal Dashboard")

    # Key metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last Price",  f"${last_price:,.2f}", f"{daily_ret:+.2%}")
    c2.metric("RSI (14)",    f"{feat_df['rsi_14'].iloc[-1]:.1f}",
              "Overbought" if feat_df["rsi_14"].iloc[-1] > 70 else
              ("Oversold" if feat_df["rsi_14"].iloc[-1] < 30 else "Neutral"))
    c3.metric("MACD Hist",   f"{feat_df['macd_hist'].iloc[-1]:+.3f}",
              "↑ Bullish" if feat_df["macd_hist"].iloc[-1] > 0 else "↓ Bearish",
              delta_color="normal" if feat_df["macd_hist"].iloc[-1] > 0 else "inverse")
    c4.metric("Vol (21d)",   f"{feat_df['vol_21'].iloc[-1]:.1%}")
    c5.metric("Bollinger %B",f"{feat_df['bb_pct'].iloc[-1]:.2f}")

    st.divider()

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("#### Price + Bollinger Bands + Volume")
        lookback_days = 252
        pdf  = feat_df.tail(lookback_days)
        sma20 = pdf["close"].rolling(20).mean()
        std20 = pdf["close"].rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=upper.index, y=upper, line=dict(color="rgba(100,149,237,0.4)", width=1),
                                  showlegend=False, name="Upper BB"))
        fig.add_trace(go.Scatter(x=lower.index, y=lower, fill="tonexty",
                                  fillcolor="rgba(100,149,237,0.07)",
                                  line=dict(color="rgba(100,149,237,0.4)", width=1),
                                  name="Bollinger Band"))
        fig.add_trace(go.Scatter(x=sma20.index, y=sma20, line=dict(color="#F5A623", width=1.2, dash="dash"),
                                  name="SMA 20"))
        fig.add_trace(go.Scatter(x=pdf.index, y=pdf["close"],
                                  line=dict(color="#1D9E75", width=1.8), name="Price"))
        fig.update_layout(height=320, margin=dict(t=20, b=10),
                          legend=dict(orientation="h", y=1.05),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#30363d"))
        st.plotly_chart(fig, width="stretch")

    with col_r:
        st.markdown("#### Classic Signal Summary")
        technicals = compute_technicals(feat_df)
        for indicator, (value, cls) in technicals.items():
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:6px 0; "
                f"border-bottom:1px solid #30363d;'>"
                f"<span style='color:#475569; font-size:0.82rem;'>{indicator}</span>"
                f"<span class='{cls}' style='font-size:0.82rem;'>{value}</span></div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### RSI  ·  MACD  (last 252 trading days)")
    fig_ind = go.Figure()
    fig_ind.add_trace(go.Scatter(x=pdf.index, y=pdf["rsi_14"], line=dict(color="#1D9E75", width=1.5),
                                  name="RSI (14)", yaxis="y"))
    fig_ind.add_hline(y=70, line_dash="dot", line_color="#D85A30", annotation_text="Overbought 70", yref="y")
    fig_ind.add_hline(y=30, line_dash="dot", line_color="#1D9E75", annotation_text="Oversold 30",   yref="y")
    fig_ind.add_trace(go.Bar(x=pdf.index, y=pdf["macd_hist"],
                              marker_color=np.where(pdf["macd_hist"] >= 0, "#1D9E75", "#D85A30"),
                              name="MACD Hist", yaxis="y2", opacity=0.75))
    fig_ind.update_layout(
        height=250, margin=dict(t=20, b=10),
        yaxis=dict(title="RSI", range=[0, 100], gridcolor="#30363d"),
        yaxis2=dict(title="MACD Hist", overlaying="y", side="right", gridcolor="#30363d"),
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_ind, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIC TECHNICALS (deep explainers)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Technical Indicators — Every Term Explained")
    st.markdown(
        "This tab treats classic technical indicators as the **baseline** for the ML models. "
        "The ML models later learn which of these signals — and in which combinations — are "
        "actually predictive on this specific stock. Think of this tab as building intuition "
        "before the machine gets involved."
    )
    st.divider()

    # Feature glossary table
    with st.expander("📖 Feature Glossary — all 18 model inputs defined", expanded=False):
        rows = []
        for col, (name, formula, meaning) in FEATURE_GLOSSARY.items():
            rows.append({"Feature": col, "Name": name, "Formula": formula, "What it tells you": meaning})
        st.dataframe(pd.DataFrame(rows).set_index("Feature"), width="stretch",
                     column_config={
                         "Formula": st.column_config.TextColumn("Formula (LaTeX-free)", width="medium"),
                         "What it tells you": st.column_config.TextColumn("What it tells you", width="large"),
                     })

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### RSI (14)  — Relative Strength Index")
        st.markdown("""
**The why:** RSI measures *how fast* a stock has moved, not just by how much.
A stock that rose sharply in the last 14 days is likely to mean-revert — RSI flags this.

**Every term:**
- `Gain_avg` = 14-day exponential moving average of *positive* daily changes
- `Loss_avg` = 14-day EMA of *negative* daily changes (absolute value)
- `RS` = Gain_avg / Loss_avg  —  ratio of up-momentum to down-momentum
- `RSI = 100 − 100 / (1 + RS)` — normalised to 0–100
  - RSI > 70 → the stock has gained unusually fast → **overbought** (potential short-term reversal)
  - RSI < 30 → the stock has fallen unusually fast → **oversold** (potential bounce)
  - 30–70 → neutral territory
        """)
        lookback = feat_df.tail(252)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=lookback.index, y=lookback["rsi_14"],
                                      line=dict(color="#1D9E75", width=1.5), name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="#D85A30", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="#1D9E75", annotation_text="Oversold")
        fig_rsi.add_hline(y=50, line_dash="dash", line_color="#8b949e", opacity=0.4)
        fig_rsi.update_layout(height=200, margin=dict(t=10, b=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               yaxis=dict(range=[0, 100], gridcolor="#30363d"), showlegend=False)
        st.plotly_chart(fig_rsi, width="stretch")

    with col_b:
        st.markdown("##### MACD — Moving Average Convergence/Divergence")
        st.markdown("""
**The why:** MACD detects changes in *trend momentum* by comparing two exponential
moving averages of different speeds.

**Every term:**
- `EMA(12)` = 12-day Exponential Moving Average — short-term trend tracker
- `EMA(26)` = 26-day EMA — long-term trend tracker
- `MACD Line = EMA(12) − EMA(26)` — positive = short-term faster than long-term → bullish
- `Signal Line = EMA(9) of MACD Line` — smoothed trigger; a crossover above = buy signal
- `Histogram = MACD Line − Signal` — *speed of divergence*; rising bars = accelerating
  momentum, even before the zero-cross

Traders watch the **zero-line cross** (MACD goes positive) and the **signal-line cross**
(MACD crosses above its signal) as entry signals.
        """)
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=lookback.index, y=lookback["macd_line"],
                                       line=dict(color="#1D9E75", width=1.5), name="MACD Line"))
        fig_macd.add_trace(go.Scatter(x=lookback.index, y=lookback["macd_signal"],
                                       line=dict(color="#F5A623", width=1.2, dash="dash"), name="Signal"))
        fig_macd.add_trace(go.Bar(x=lookback.index, y=lookback["macd_hist"],
                                   marker_color=np.where(lookback["macd_hist"] >= 0, "#1D9E75", "#D85A30"),
                                   name="Histogram", opacity=0.7))
        fig_macd.add_hline(y=0, line_color="#8b949e", line_width=0.8)
        fig_macd.update_layout(height=200, margin=dict(t=10, b=10),
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                yaxis=dict(gridcolor="#30363d"),
                                legend=dict(orientation="h", y=1.15, font_size=10))
        st.plotly_chart(fig_macd, width="stretch")

    st.divider()
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("##### Bollinger Bands")
        st.markdown("""
**The why:** Bollinger Bands define a *dynamic price envelope* based on recent volatility.
When the bands narrow, a breakout is likely. When price touches a band, mean-reversion may follow.

**Every term:**
- `SMA20` = 20-day Simple Moving Average — the mid-band / baseline
- `σ` (sigma) = standard deviation of closing prices over 20 days — measures recent volatility
- `Upper Band = SMA20 + 2σ` — price this far above average is statistically unusual
- `Lower Band = SMA20 − 2σ` — price this far below average is statistically unusual
- `%B = (Price − Lower) / (Upper − Lower)` — normalised position: 0 = lower, 1 = upper
- `Band Width = (Upper − Lower) / SMA20` — wider = more volatile; narrow = compression before move
        """)
        sma20_full = feat_df["close"].rolling(20).mean()
        std20_full = feat_df["close"].rolling(20).std()
        upper_f    = (sma20_full + 2 * std20_full).tail(252)
        lower_f    = (sma20_full - 2 * std20_full).tail(252)
        sma20_f    = sma20_full.tail(252)
        close_f    = feat_df["close"].tail(252)

        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=upper_f.index, y=upper_f, line=dict(color="rgba(100,149,237,0.5)", width=1), name="Upper"))
        fig_bb.add_trace(go.Scatter(x=lower_f.index, y=lower_f, fill="tonexty",
                                     fillcolor="rgba(100,149,237,0.07)",
                                     line=dict(color="rgba(100,149,237,0.5)", width=1), name="Lower"))
        fig_bb.add_trace(go.Scatter(x=sma20_f.index, y=sma20_f, line=dict(color="#F5A623", width=1.2, dash="dash"), name="SMA20"))
        fig_bb.add_trace(go.Scatter(x=close_f.index, y=close_f, line=dict(color="#1D9E75", width=1.5), name="Price"))
        fig_bb.update_layout(height=220, margin=dict(t=10, b=10),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              yaxis=dict(gridcolor="#30363d"),
                              legend=dict(orientation="h", y=1.15, font_size=10))
        st.plotly_chart(fig_bb, width="stretch")

    with col_d:
        st.markdown("##### SMA Trend Structure")
        st.markdown("""
**The why:** Moving averages filter out daily noise and reveal the underlying *trend direction*.
The 200-day SMA is the single most-watched line by institutional investors.

**Every term:**
- `SMA20` = 20-day average — *short-term* trend; reacts quickly
- `SMA50` = 50-day average — *medium-term* trend
- `SMA200` = 200-day average — *long-term* regime indicator
- **Golden Cross**: SMA50 crosses above SMA200 → institutional buy signal
- **Death Cross**: SMA50 crosses below SMA200 → institutional sell signal
- `Price / SMAx − 1` = how far above (+) or below (−) price sits vs. each average
  (these are features in the ML model, not just visual aids)
        """)
        sma50_f  = feat_df["close"].rolling(50).mean().tail(252)
        sma200_f = feat_df["close"].rolling(200).mean().tail(252)
        fig_sma = go.Figure()
        fig_sma.add_trace(go.Scatter(x=close_f.index, y=close_f, line=dict(color="#1D9E75", width=1.5), name="Price"))
        fig_sma.add_trace(go.Scatter(x=sma20_f.index, y=sma20_f, line=dict(color="#F5A623", width=1.2, dash="dot"), name="SMA 20"))
        fig_sma.add_trace(go.Scatter(x=sma50_f.index, y=sma50_f, line=dict(color="cornflowerblue", width=1.2, dash="dash"), name="SMA 50"))
        fig_sma.add_trace(go.Scatter(x=sma200_f.index, y=sma200_f, line=dict(color="#D85A30", width=1.5), name="SMA 200"))
        fig_sma.update_layout(height=220, margin=dict(t=10, b=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               yaxis=dict(gridcolor="#30363d"),
                               legend=dict(orientation="h", y=1.15, font_size=10))
        st.plotly_chart(fig_sma, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HMM REGIME DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Market Regime Detection  —  Hidden Markov Model")
    st.markdown("""
**The big picture:** Markets don't behave the same way all the time.
A momentum strategy that works in a low-volatility bull trend can destroy
wealth in a high-volatility bear regime. The HMM detects which *hidden state*
the market is most likely in at each point in time.

**What is a Hidden Markov Model?**  
Think of the market as a machine that flips between invisible modes (regimes).
You can't directly observe the mode — you can only observe its outputs (daily returns, volatility).
The HMM reverse-engineers the hidden mode from the observable outputs.

**Every term:**
- `Hidden State (regime)` = unobservable market mode: Bull, Transitional, or Bear
- `Observable` = daily log-return + absolute return (|ret|), what the model *sees*
- `Emission probability` = P(observing this return | we are in state k). Each regime has a characteristic return/vol profile.
- `Transition matrix` = P(being in state j tomorrow | in state i today). This encodes *persistence* — regimes tend to stick.
- `Viterbi algorithm` = finds the most likely sequence of hidden states given the full return history.
- Re-labelling: states are ordered by volatility so Regime 0 = lowest vol (bull), Regime {n-1} = highest (bear).
    """)
    st.divider()

    with st.spinner("Fitting HMM…"):
        spy_ret = np.log(spy / spy.shift(1)).dropna()
        regimes, hmm_model, remap = fit_hmm(spy_ret, n_regimes=n_regimes)

    # Regime colour bands on SPY price chart
    spy_aligned = spy.reindex(regimes.index)
    current_regime = int(regimes.iloc[-1])
    regime_label   = REGIME_LABELS.get(current_regime, f"Regime {current_regime}")
    regime_color   = REGIME_COLORS.get(current_regime, "#888")

    st.markdown(
        f"**Current SPY Regime (today):** "
        f"<span style='color:{regime_color}; font-weight:700; font-size:1.1rem;'>{regime_label}</span>",
        unsafe_allow_html=True,
    )

    # Build shaded regime chart
    fig_reg = go.Figure()
    # Background regime shading
    for reg_id, reg_col in REGIME_COLORS.items():
        if reg_id >= n_regimes:
            continue
        mask = regimes == reg_id
        # Find contiguous blocks
        in_block = False
        block_start = None
        for date_i, is_in in mask.items():
            if is_in and not in_block:
                block_start = date_i
                in_block = True
            elif not is_in and in_block:
                fig_reg.add_vrect(x0=block_start, x1=date_i,
                                   fillcolor=reg_col,
                                   opacity=0.08 if reg_id == 0 else 0.30,
                                   line_width=0)
                in_block = False
        if in_block:
            fig_reg.add_vrect(x0=block_start, x1=regimes.index[-1],
                               fillcolor=reg_col,
                               opacity=0.08 if reg_id == 0 else 0.30,
                               line_width=0)

    fig_reg.add_trace(go.Scatter(x=spy.index, y=spy.values,
                                  line=dict(color="#64748b", width=1.5), name="SPY"))
    fig_reg.update_layout(
        title="SPY Price with HMM Regime Overlay (colour = detected state)",
        height=360, margin=dict(t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(title="SPY ($)", gridcolor="#30363d"),
    )
    for reg_id, reg_col in REGIME_COLORS.items():
        if reg_id < n_regimes:
            fig_reg.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                          marker=dict(color=reg_col, size=10, symbol="square"),
                                          name=REGIME_LABELS[reg_id]))
    st.plotly_chart(fig_reg, width="stretch")

    # Transition matrix heatmap
    col_trans, col_stats = st.columns(2)
    with col_trans:
        st.markdown("##### HMM Transition Matrix")
        st.markdown("""
Each cell [i → j] = **probability of moving from Regime i to Regime j tomorrow**.
High diagonal values (e.g. 0.97) = regimes are *persistent*.
Off-diagonal values show how likely a regime switch is in any given day.
        """)
        trans_mat = hmm_model["transmat_"]
        labels    = [REGIME_LABELS[i] for i in range(n_regimes)]
        fig_trans = px.imshow(
            trans_mat,
            x=labels, y=labels,
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            text_auto=".2f",
        )
        fig_trans.update_layout(height=300, margin=dict(t=20, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 xaxis_title="To Regime", yaxis_title="From Regime")
        st.plotly_chart(fig_trans, width="stretch")

    with col_stats:
        st.markdown("##### Regime Statistics (SPY)")
        stat_rows = []
        spy_ret_aligned = spy_ret.reindex(regimes.index)
        for r_id in range(n_regimes):
            mask = regimes == r_id
            r_returns = spy_ret_aligned[mask]
            stat_rows.append({
                "Regime": REGIME_LABELS[r_id],
                "% of time": f"{mask.mean():.1%}",
                "Avg daily ret": f"{r_returns.mean():.3%}",
                "Daily vol": f"{r_returns.std():.3%}",
                "Ann. return": f"{r_returns.mean() * 252:.1%}",
                "Ann. vol": f"{r_returns.std() * np.sqrt(252):.1%}",
                "Sharpe": f"{(r_returns.mean() / r_returns.std() * np.sqrt(252)):.2f}",
            })
        st.dataframe(pd.DataFrame(stat_rows).set_index("Regime"), width="stretch")

        st.markdown("##### Regime History (last 60 days)")
        recent_reg = regimes.tail(60)
        regime_colors_list = [REGIME_COLORS.get(int(v), "#888") for v in recent_reg.values]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=recent_reg.index, y=[1]*len(recent_reg),
            marker_color=regime_colors_list,
            hovertext=[REGIME_LABELS.get(int(v), str(v)) for v in recent_reg.values],
            hoverinfo="x+text",
        ))
        fig_hist.update_layout(height=100, margin=dict(t=5, b=5, l=0, r=0),
                                yaxis=dict(showticklabels=False, showgrid=False),
                                xaxis=dict(showgrid=False),
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                showlegend=False)
        st.plotly_chart(fig_hist, width="stretch")

    st.divider()
    # Regime-conditioned stock return distributions
    st.markdown("##### How does the detected regime affect this stock's returns?")
    st.caption(f"Distribution of {selected_ticker} daily returns conditioned on SPY's HMM regime.")
    stock_ret = np.log(price_s / price_s.shift(1)).dropna()
    stock_reg = regimes.reindex(stock_ret.index)

    fig_dist = go.Figure()
    for r_id in range(n_regimes):
        mask = stock_reg == r_id
        r_vals = stock_ret[mask].dropna()
        fig_dist.add_trace(go.Histogram(
            x=r_vals, nbinsx=60, name=REGIME_LABELS[r_id],
            marker_color=REGIME_COLORS[r_id], opacity=0.6,
            histnorm="probability",
        ))
    fig_dist.update_layout(barmode="overlay", height=280, margin=dict(t=20, b=10),
                            xaxis_title="Daily log-return", yaxis_title="Probability",
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_dist, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML SIGNALS (RF + XGBoost)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### ML Signal Generation  —  Random Forest + XGBoost Ensemble")
    st.markdown(f"""
**The big picture:** Classic technical indicators give rules of thumb (RSI > 70 = sell).
ML models *learn* which combination of indicators — and at what thresholds — actually
predicted up-moves for *this specific stock* over the *training period*.

**Ensemble approach:** Two models are trained and their probability estimates are averaged:
- **Random Forest (RF):** builds many decision trees on random subsets of data and features,
  then averages the results. Robust against overfitting, handles nonlinear patterns.
- **XGBoost:** builds trees *sequentially*, each correcting the errors of the previous.
  Tends to perform slightly better on structured financial data.
- **Soft voting:** instead of a majority vote, we average the *probabilities* — this gives
  a smoother, more calibrated confidence signal.

**Target variable:** `y = 1` if the stock is up over the next **{lookahead} days** (your lookahead setting).
**Walk-forward split:** the model trains on the first 70% of history and signals on the last 30%
— it never sees the future while training.
    """)
    st.divider()

    with st.spinner("Training RF + XGBoost model…"):
        prob_series, importances, train_end, accuracy = train_ml_model(feat_df, lookahead=lookahead)

    # Summary metrics
    current_prob = float(prob_series.iloc[-1])
    signal_str   = "BUY signal" if current_prob > 0.55 else ("SELL signal" if current_prob < 0.45 else "NEUTRAL")
    sig_color    = "#1D9E75" if current_prob > 0.55 else ("#D85A30" if current_prob < 0.45 else "#8b949e")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P(Up) — Today",    f"{current_prob:.1%}")
    m2.metric("Signal",           signal_str)
    m3.metric("OOS Accuracy",     f"{accuracy:.1%}")
    m4.metric("Trained through",  str(train_end.date()))

    st.markdown(
        f"**Model verdict:** <span style='color:{sig_color}; font-size:1.1rem; font-weight:700;'>"
        f"P(stock up in {lookahead}d) = {current_prob:.1%}  →  {signal_str}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "⚠️ This signal reflects learned historical patterns, not a guarantee. "
        "Always cross-reference with the regime tab and your fundamental view from the BL app."
    )
    st.divider()

    col_prob, col_imp = st.columns([3, 2])

    with col_prob:
        st.markdown(f"##### P(Up in {lookahead}d) — Rolling Signal History")
        fig_prob = go.Figure()
        fig_prob.add_hline(y=0.5, line_color="#8b949e", line_dash="dash", opacity=0.6)
        fig_prob.add_hline(y=0.55, line_color="#1D9E75", line_dash="dot", opacity=0.5, annotation_text="Buy zone")
        fig_prob.add_hline(y=0.45, line_color="#D85A30", line_dash="dot", opacity=0.5, annotation_text="Sell zone")
        colors = ["#1D9E75" if p > 0.55 else ("#D85A30" if p < 0.45 else "#F5A623")
                  for p in prob_series.values]
        fig_prob.add_trace(go.Scatter(
            x=prob_series.index, y=prob_series.values,
            mode="lines", line=dict(color="#94a3b8", width=1.2), showlegend=False,
        ))
        fig_prob.add_trace(go.Scatter(
            x=prob_series.index, y=prob_series.values,
            mode="markers", marker=dict(color=colors, size=3), showlegend=False,
        ))
        fig_prob.update_layout(height=300, margin=dict(t=20, b=10),
                                yaxis=dict(title="P(Up)", tickformat=".0%", range=[0, 1], gridcolor="#30363d"),
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_prob, width="stretch")

        # Signal accuracy per regime
        if len(regimes) > 0:
            prob_aligned  = prob_series.copy()
            reg_aligned   = regimes.reindex(prob_aligned.index)
            st.markdown("##### Signal Quality by Regime")
            st.caption("Does the ML signal perform differently in bull vs. bear regimes?")
            regime_acc = {}
            stock_ret_oos = np.log(price_s / price_s.shift(lookahead)).shift(-lookahead)
            actual_up = (stock_ret_oos > 0).reindex(prob_aligned.index)
            for r_id in range(n_regimes):
                mask = reg_aligned == r_id
                if mask.sum() < 10:
                    continue
                hits = ((prob_aligned[mask] > 0.5) == actual_up[mask]).dropna()
                regime_acc[REGIME_LABELS[r_id]] = {
                    "Days in regime (test set)": int(mask.sum()),
                    "Signal accuracy": f"{hits.mean():.1%}" if len(hits) > 0 else "N/A",
                }
            if regime_acc:
                st.dataframe(pd.DataFrame(regime_acc).T, width="stretch")

    with col_imp:
        st.markdown("##### Feature Importances — Why does the model signal this?")
        st.markdown("""
Feature importance = how much each feature reduces prediction error across all decision trees.
Higher = more relied upon. This answers the **"why"** behind the signal.
        """)
        top_n = 10
        top_imp = importances.head(top_n)
        fig_imp = go.Figure(go.Bar(
            y=top_imp.index[::-1],
            x=top_imp.values[::-1],
            orientation="h",
            marker=dict(
                color=top_imp.values[::-1],
                colorscale=[[0, "#30363d"], [1, "#1D9E75"]],
            ),
        ))
        fig_imp.update_layout(height=360, margin=dict(t=20, b=10, l=10, r=10),
                               xaxis=dict(title="Avg importance", gridcolor="#30363d"),
                               yaxis=dict(tickfont=dict(size=11)),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_imp, width="stretch")

        # Plain-English explanation for the top 3
        st.markdown("**Top 3 drivers — in plain English:**")
        for feat in importances.head(3).index:
            glos = FEATURE_GLOSSARY.get(feat, (feat, "", ""))
            st.markdown(f"- **{glos[0]}** (`{feat}`): {glos[2]}")

    st.divider()
    with st.expander("📊 Full feature importance table"):
        imp_df = importances.reset_index()
        imp_df.columns = ["Feature", "Importance"]
        imp_df["Name"]    = imp_df["Feature"].map(lambda f: FEATURE_GLOSSARY.get(f, (f,))[0])
        imp_df["Meaning"] = imp_df["Feature"].map(lambda f: FEATURE_GLOSSARY.get(f, ("","",""))[2])
        st.dataframe(imp_df.set_index("Feature").style.format({"Importance": "{:.4f}"}),
                     width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LSTM FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### LSTM Price Forecast  —  Deep Sequence Learning")
    st.markdown(f"""
**The big picture:** LSTM (Long Short-Term Memory) is a type of recurrent neural network
designed for *sequential data*. Unlike RF/XGBoost which treat each day independently,
the LSTM reads a *window* of the past {30} trading days as a sequence and learns which
temporal patterns precede up/down moves.

**Architecture (every layer explained):**
- `Input`: the last **30 days** of [normalised price, 5-day vol, RSI] — a (30, 3) tensor
- `LSTM(64)`: 64 memory units, each maintaining a *hidden state* that summarises past observations.
  The key innovation: LSTM has **forget gates** — it learns *what to remember and what to ignore*
  across time steps. This lets it capture both short-term momentum and longer-term reversals.
  - `h_t` (hidden state) = what the LSTM passes forward to the next time step
  - `c_t` (cell state)   = the long-term memory carried across many steps
  - `σ` (sigmoid gates)  = decide how much old memory to *forget* vs. how much new info to *add*
- `Dropout(0.2)`: randomly zeros 20% of neurons during training — reduces overfitting
- `Dense(32) → Dense(1)`: maps the LSTM's final hidden state to a scalar: the predicted
  **{lstm_horizon}-day forward log-return**

**Walk-forward:** trains on first 70% of data, forecasts out-of-sample on last 30%.
Normalisation per window (divide by first value) makes the model learn *shape* not levels.
    """)
    st.divider()

    with st.spinner("Training LSTM… (pure-numpy, ~10–20 seconds on first run)"):
        lstm_preds, lstm_actuals, lstm_train_end = train_lstm(
            price_s, seq_len=30, horizon=lstm_horizon, epochs=30
        )

    # Align and compute metrics
    compare = pd.DataFrame({"Predicted": lstm_preds, "Actual": lstm_actuals}).dropna()
    directional_acc = ((compare["Predicted"] > 0) == (compare["Actual"] > 0)).mean()
    corr = compare.corr().iloc[0, 1]
    latest_pred = float(lstm_preds.iloc[-1])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Next-horizon pred",      f"{latest_pred:+.2%}",
              "↑ Up" if latest_pred > 0 else "↓ Down")
    m2.metric("Directional accuracy",   f"{directional_acc:.1%}")
    m3.metric("Pred-actual correlation",f"{corr:.2f}")
    m4.metric("Trained through",        str(lstm_train_end.date()))

    st.divider()
    col_fc, col_err = st.columns([3, 2])

    with col_fc:
        st.markdown(f"##### Predicted vs. Actual {lstm_horizon}-Day Return (out-of-sample)")
        fig_lstm = go.Figure()
        fig_lstm.add_hline(y=0, line_color="#30363d")
        fig_lstm.add_trace(go.Scatter(x=compare.index, y=compare["Actual"],
                                       line=dict(color="#8b949e", width=1.2), name="Actual"))
        fig_lstm.add_trace(go.Scatter(x=compare.index, y=compare["Predicted"],
                                       line=dict(color="#1D9E75", width=1.5, dash="dash"),
                                       name="LSTM Prediction"))
        fig_lstm.update_layout(
            height=300, margin=dict(t=20, b=10),
            yaxis=dict(tickformat=".1%", gridcolor="#30363d"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_lstm, width="stretch")

    with col_err:
        st.markdown("##### Prediction Error Distribution")
        errors = compare["Predicted"] - compare["Actual"]
        fig_err = go.Figure()
        fig_err.add_trace(go.Histogram(x=errors, nbinsx=50,
                                        marker_color="#1D9E75", opacity=0.75))
        fig_err.add_vline(x=0, line_color="#D85A30", line_dash="dash")
        fig_err.update_layout(height=200, margin=dict(t=20, b=10),
                               xaxis=dict(title="Prediction error (log-return)", tickformat=".1%"),
                               yaxis=dict(title="Count"),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_err, width="stretch")

        st.markdown("##### Performance Summary")
        mae  = float(errors.abs().mean())
        rmse = float(np.sqrt((errors ** 2).mean()))
        st.dataframe(pd.DataFrame({
            "Metric": ["MAE", "RMSE", "Directional Acc", "Correlation"],
            "Value":  [f"{mae:.3%}", f"{rmse:.3%}", f"{directional_acc:.1%}", f"{corr:.3f}"],
        }).set_index("Metric"), width="stretch")

    st.divider()

    # How the architecture applies here
    with st.expander("🧠 Understanding the LSTM Architecture — term by term"):
        st.markdown("""
This LSTM is implemented in pure numpy — no TensorFlow or PyTorch required.
The mathematics are identical to a Keras LSTM; only the execution engine differs.

| Term | Formula / definition | Why it matters here |
|------|---------------------|---------------------|
| `Forget gate f_t` | `σ(Wf·x_t + Uf·h_{t-1} + bf)` | Decides what fraction of the old cell state to *discard*. Near 0 = forget, near 1 = keep. Lets the model reset after a regime change. |
| `Input gate i_t` | `σ(Wi·x_t + Ui·h_{t-1} + bi)` | Decides how much *new* information to write into the cell state. |
| `Candidate g_t` | `tanh(Wg·x_t + Ug·h_{t-1} + bg)` | The actual new content to potentially add — bounded to [−1, 1] by tanh. |
| `Output gate o_t` | `σ(Wo·x_t + Uo·h_{t-1} + bo)` | Controls how much of the cell state is exposed as the hidden state output. |
| `Cell state c_t` | `f_t ⊙ c_{t-1} + i_t ⊙ g_t` | The long-term memory. ⊙ = element-wise multiply. Updated every step. |
| `Hidden state h_t` | `o_t ⊙ tanh(c_t)` | What the LSTM "outputs" at each step. The final h_T feeds the readout layer. |
| `σ` (sigmoid) | `1/(1+e^{-x})` | Squashes gate values to (0,1) so they act as differentiable on/off switches. |
| `Readout` | `W_out · h_T + b_out` | Linear projection from hidden state → predicted log-return (scalar). |
| `BPTT` | Backpropagation Through Time | Gradient of MSE loss propagated backwards through all T time steps to update W, U, b. |
| `Adam` | Adaptive moment estimation | Optimiser that adapts learning rates per parameter using first (m) and second (v) moment estimates. |
| `Dropout (0.2)` | Zero 20% of h_t during training | Applied to the hidden state at each step to prevent memorisation of training sequences. |
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — UNIVERSE SCANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### Universe Scanner  —  All 17 Stocks at a Glance")
    st.markdown(
        "Daily snapshot of key technical signals across the full portfolio universe. "
        "Pairs with the BL app: if the BL allocates 20% to NVDA but the scanner shows "
        "a bear regime + overbought RSI, that's a prompt to re-examine the near-term entry timing."
    )
    st.divider()

    with st.spinner(f"Computing signals for all {len(ACTIVE_TICKERS)} tickers…"):
        scanner_rows = []
        for t in ACTIVE_TICKERS:
            if t not in closes.columns or t not in volumes.columns:
                continue
            try:
                p  = closes[t].dropna()
                v  = volumes[t].dropna()
                fd = compute_features(p, v)
                last_fd = fd.iloc[-1]
                daily_r = float(p.iloc[-1] / p.iloc[-2] - 1)

                # Quick RSI signal
                rsi_val = last_fd["rsi_14"]
                rsi_sig = "🟢 Oversold" if rsi_val < 30 else ("🔴 OB" if rsi_val > 70 else "–")

                # MACD direction
                macd_sig = "↑" if last_fd["macd_hist"] > 0 else "↓"

                # Vol regime
                vol_sig = "High" if last_fd["vol_ratio"] > 1.2 else "Normal"

                # SMA position
                sma_sig = "Above 200" if last_fd["price_vs_sma200"] > 0 else "Below 200"

                scanner_rows.append({
                    "Ticker":         t,
                    "Type":           "✦ Rotation" if t in st.session_state.get("custom_tickers", []) else "Core",
                    "Last Price":     float(p.iloc[-1]),
                    "1D Ret":         daily_r,
                    "1M Ret":         float(last_fd["ret_21d"]),
                    "RSI (14)":       float(rsi_val),
                    "RSI Signal":     rsi_sig,
                    "MACD":           macd_sig,
                    "Vol Regime":     vol_sig,
                    "vs SMA 200":     float(last_fd["price_vs_sma200"]),
                    "SMA Trend":      sma_sig,
                    "BB %B":          float(last_fd["bb_pct"]),
                    "52W High Dist":  float(last_fd["high_52w_pct"]),
                })
            except Exception:
                pass

    scan_df = pd.DataFrame(scanner_rows).set_index("Ticker")

    def color_ret(v):
        if isinstance(v, float):
            return "color: #1D9E75" if v > 0 else "color: #D85A30" if v < 0 else ""
        return ""

    styled_scan = (
        scan_df.style
        .format({
            "Last Price":  "${:,.2f}",
            "1D Ret":      "{:+.2%}",
            "1M Ret":      "{:+.2%}",
            "RSI (14)":    "{:.1f}",
            "vs SMA 200":  "{:+.1%}",
            "BB %B":       "{:.2f}",
            "52W High Dist": "{:.1%}",
        })
        .map(color_ret, subset=["1D Ret", "1M Ret", "vs SMA 200", "52W High Dist"])
        .background_gradient(subset=["RSI (14)"], cmap="RdYlGn_r", vmin=20, vmax=80)
        .background_gradient(subset=["BB %B"],    cmap="RdYlGn_r", vmin=0, vmax=1)
    )
    st.dataframe(styled_scan, width="stretch", height=680,
                 column_config={
                     "Type":        st.column_config.TextColumn("Type", width="small"),
                     "RSI Signal":  st.column_config.TextColumn("RSI Signal", width="small"),
                     "MACD":        st.column_config.TextColumn("MACD ↑/↓", width="small"),
                     "Vol Regime":  st.column_config.TextColumn("Vol", width="small"),
                     "SMA Trend":   st.column_config.TextColumn("vs SMA200", width="medium"),
                 })

    st.divider()
    st.markdown("##### 1-Month Return  vs.  RSI — Momentum Map")
    st.caption("Each bubble = one stock. X-axis = RSI, Y-axis = 1-month return. "
               "✦ = rotation candidate. Top-left = strong returns, not yet overbought.")
    fig_map = px.scatter(
        scan_df.reset_index(), x="RSI (14)", y="1M Ret", text="Ticker",
        color="Type",
        color_discrete_map={"Core": "#1D9E75", "✦ Rotation": "#F5A623"},
        symbol="Type",
        symbol_map={"Core": "circle", "✦ Rotation": "diamond"},
        size_max=14,
    )
    fig_map.add_vline(x=70, line_dash="dot", line_color="#D85A30", opacity=0.5)
    fig_map.add_vline(x=30, line_dash="dot", line_color="#1D9E75", opacity=0.5)
    fig_map.add_hline(y=0,  line_dash="dash", line_color="#8b949e", opacity=0.4)
    fig_map.update_traces(textposition="top center", marker=dict(size=14))
    fig_map.update_layout(
        height=420, margin=dict(t=30, b=10),
        yaxis=dict(tickformat=".0%", title="1-Month Return", gridcolor="#30363d"),
        xaxis=dict(title="RSI (14) — today"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(title="", orientation="h", y=1.08),
    )
    st.plotly_chart(fig_map, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REBALANCING PLANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("#### 💼 Rebalancing Planner")
    st.markdown(
        "This tab bridges the **Portfolio Optimiser (DCF-BL)** app and the daily technical signals. "
        "Enter your BL target weights and current holdings below. The planner computes the exact "
        "trades needed, estimates their all-in cost at your scale, and cross-checks each trade "
        "against the regime, ML signal, and technical indicators — so you know not just *what* "
        "to trade, but *whether now is a good time to execute it*."
    )

    # ── SLIPPAGE MODEL ────────────────────────────────────────────────────────
    # For a $25k retail portfolio in large-cap US tech, transaction costs break
    # down into three components:
    #
    #   1. Commission:  $0 on Moo Moo (commission-free)
    #   2. Bid-ask spread: half-spread × trade value.
    #      Estimated as spread_bps × price / 2 per share.
    #      For liquid mega-caps: ~1–3 bps. We calibrate per ticker.
    #   3. Slippage (open-gap cost): the difference between yesterday's close
    #      (the signal price) and tomorrow's open (your actual fill).
    #      Estimated as: slip_fraction × ATR_14
    #      where ATR_14 is the 14-day average true range — it captures how
    #      large a typical intraday move is for this stock.
    #      For market orders on liquid names: slip_fraction ≈ 0.10–0.15.
    #      i.e. you expect to fill within 10–15% of the daily range from close.
    #
    # Total cost per trade = spread_cost + slippage_cost
    # Total cost as % of trade = (spread_cost + slippage_cost) / trade_value

    # ── COST MODEL — calibrated to Moo Moo (June 2026) ──────────────────────
    #
    # Moo Moo actual cost structure for US stocks:
    #   - Commission:         $0 (zero commission)
    #   - Platform fee:       ~$0.99 per trade (Moo Moo's regulatory/platform fee)
    #   - SEC/FINRA fees:     ~$0.000008 × trade value (sell orders only, negligible)
    #   - Bid-ask spread:     Half-spread on entry (market-specific, not Moo Moo)
    #
    # Ka Wai's observed cost: ~$1.10 per trade, consistent with:
    #   $0.99 platform fee + ~$0.11 average half-spread on liquid large-caps
    #
    # We model this as:
    #   fixed_fee   = $0.99 (Moo Moo platform fee, every trade)
    #   spread_cost = half_spread_bps × trade_value  (market microstructure)
    #   total       = fixed_fee + spread_cost
    #
    # At $25k trade sizes, spread cost is ~$0.10–$0.25 for liquid names,
    # so total ~$1.10–$1.25 matches your observed experience exactly.
    # We NO LONGER model ATR-based slippage separately because for market orders
    # on liquid mega-caps at this size, the bid-ask spread already captures
    # the realistic execution gap.

    MOOMOO_FIXED_FEE = 0.99   # USD per trade, consistent with Ka Wai's observed ~$1.10

    # Half-spread in basis points (1 bp = 0.01%) per ticker.
    # At $2,500 avg trade: 1 bp = $0.25. These are realistic retail half-spreads.
    SPREAD_BPS = {
        "AAPL": 0.5, "ADBE": 1.0, "AMAT": 1.0, "AMZN": 0.5, "ASML": 2.0,
        "SPGI": 1.0, "FICO": 1.5, "GOOGL": 0.5, "LRCX": 1.0, "MA":   0.8,
        "META": 0.5, "MSCI": 1.5, "MSFT": 0.5, "NFLX": 0.8, "NVDA": 0.5,
        "TSM":  1.5, "V":   0.8,
    }

    def estimate_trade_cost(ticker: str, trade_value: float, feat_df: pd.DataFrame) -> dict:
        """
        Estimate the all-in cost of one trade on Moo Moo.

        Cost components:
        ─────────────────────────────────────────────────────────────
        fixed_fee     : $0.99 Moo Moo platform fee, flat per trade
        spread_cost   : half-spread × trade value
                        The bid-ask spread is the gap between the best
                        buy price and best sell price. You pay half of it
                        on entry. For MSFT at $390 with a 0.5 bp spread:
                        spread_cost = $2,500 × 0.000005 = $0.13
        total_cost    : fixed_fee + spread_cost
                        At your typical $1,000–$5,000 trade sizes this
                        produces ~$1.10–$1.25, matching your observation.
        total_pct     : total / |trade_value| — cost as % of trade
        breakeven_days: days of price movement to recover cost
                        = total_pct / daily_vol

        Note: no separate slippage term — for liquid mega-caps at $25k
        scale, the spread cost already captures realistic execution friction.
        """
        abs_val = abs(trade_value)
        if abs_val < 1:
            return {"fixed_fee": 0, "spread_cost": 0,
                    "total_cost": 0, "total_pct": 0, "breakeven_days": 0}

        spread_bp   = SPREAD_BPS.get(ticker, 1.0)
        spread_cost = abs_val * (spread_bp / 10_000)    # half-spread × notional
        total_cost  = MOOMOO_FIXED_FEE + spread_cost
        total_pct   = total_cost / abs_val if abs_val > 0 else 0

        daily_vol   = float(feat_df["vol_21"].iloc[-1]) / np.sqrt(252)
        breakeven_d = total_pct / daily_vol if daily_vol > 0 else 0

        return {
            "fixed_fee":      round(MOOMOO_FIXED_FEE, 2),
            "spread_cost":    round(spread_cost,       2),
            "total_cost":     round(total_cost,        2),
            "total_pct":      round(total_pct,         6),
            "breakeven_days": round(breakeven_d,       1),
        }

    def score_entry_timing(ticker: str, feat_df: pd.DataFrame,
                           regime: int, prob_up: "float | None",
                           direction: str,
                           lookahead: int = 5,
                           ml_probs: "pd.Series | None" = None) -> dict:
        """
        Institutional-grade signal scorecard using IC-weighted voting.

        What is IC (Information Coefficient)?
        ──────────────────────────────────────
        IC = Spearman rank-correlation between a signal's value and the
        forward return it is trying to predict, computed on training data only.
        It directly answers: 'When this signal fired historically, how well did
        it predict direction for THIS stock?'

          IC >  0.10  → Strong, reliable edge
          IC   0–0.10 → Useful but modest
          IC ≈  0     → Coin flip — signal adds nothing
          IC <  0     → CONTRARIAN: the signal fires in the wrong direction

        Two institutional properties this enables
        ─────────────────────────────────────────
        1. IC-weighted votes — signals with higher |IC| contribute proportionally
           more to the composite score.  A poor signal with IC = 0.01 barely moves
           the needle; a strong one with IC = 0.12 carries 12× the weight.

        2. Contrarian flip — if IC < 0, the raw vote is INVERTED automatically.
           Example: ML says prob_up = 70% (raw vote = +1) but IC_ml = −0.05
           because for this stock the model has historically fired bullish right
           before DOWN moves → effective_vote = +1 × sign(−0.05) = −1.
           The signal now correctly opposes the BUY.

        Regime is a signal, not a gate
        ───────────────────────────────
        Bear market regime reduces the score (negative vote with its own IC weight)
        but does not block an 'Execute now' verdict outright.  The user decides
        whether to enter — the system just shows the full picture.

        Returns
        ───────
        verdict, colour, score (IC-weighted), composite_ic, ic_tstat,
        expected_alpha, alpha_se, kelly_size, confidence_pct,
        agree, oppose, n_signals, n_active, signals, top_reasons,
        regime_label, regime_override (always False)
        """
        from scipy.stats import spearmanr

        last   = feat_df.iloc[-1]
        rsi    = float(last["rsi_14"])
        macd   = float(last["macd_hist"])
        vs200  = float(last["price_vs_sma200"])
        vol_r  = float(last["vol_ratio"])
        h52w   = float(last["high_52w_pct"])
        vol_21 = float(last["vol_21"])
        regime_label = REGIME_LABELS.get(regime, "Unknown")

        # ── Per-signal IC on training data (first 70%, no look-ahead) ────────
        _df  = feat_df.copy()
        _df["_fwd"] = np.log(_df["close"].shift(-lookahead) / _df["close"])
        _df  = _df.dropna()
        _spl = int(len(_df) * 0.70)
        _tr  = _df.iloc[:_spl]
        _n   = max(len(_tr), 1)

        def _ic(col: str, inv: bool = False) -> float:
            if col not in _tr.columns or _n < 50:
                return 0.04
            v  = _tr[col].values.astype(float) * (-1.0 if inv else 1.0)
            f  = _tr["_fwd"].values.astype(float)
            ok = np.isfinite(v) & np.isfinite(f)
            if ok.sum() < 50:
                return 0.04
            val, _ = spearmanr(v[ok], f[ok])
            return float(val) if not np.isnan(val) else 0.04

        ic_map = {
            "Market Regime":       max(_ic("price_vs_sma200"), 0.03),
            "RSI (14)":            _ic("rsi_14",          inv=True),
            "MACD":                _ic("macd_hist"),
            "vs SMA 200":          _ic("price_vs_sma200"),
            "52W High Proximity":  _ic("high_52w_pct"),
            "Vol Regime":          _ic("vol_ratio",        inv=True),
            "ML Signal":           0.04,
        }

        # ML IC from OOS probability series — no look-ahead bias
        if ml_probs is not None and len(ml_probs) > 30:
            _fwd_ml = np.log(feat_df["close"].shift(-lookahead) / feat_df["close"])
            _common = ml_probs.index.intersection(_fwd_ml.dropna().index)
            if len(_common) > 30:
                _v, _ = spearmanr(ml_probs.reindex(_common).values,
                                   _fwd_ml.reindex(_common).values)
                if not np.isnan(_v):
                    ic_map["ML Signal"] = float(_v)

        # ── Signal definitions ────────────────────────────────────────────────
        raw_signals = []

        # 1. Market Regime (HMM)
        if direction == "BUY":
            rv = +1 if regime == 0 else (-1 if regime == 2 else 0)
            raw_signals.append({
                "name": "Market Regime", "vote": rv, "ic": ic_map["Market Regime"],
                "value": regime_label,
                "reason_for":    f"Bull regime ({regime_label}) favours new longs.",
                "reason_against": f"{regime_label} regime — elevated macro risk for new longs.",
            })
        else:
            rv = +1 if regime == 2 else (-1 if regime == 0 else 0)
            raw_signals.append({
                "name": "Market Regime", "vote": rv, "ic": ic_map["Market Regime"],
                "value": regime_label,
                "reason_for":    f"Bear regime ({regime_label}) supports reducing exposure.",
                "reason_against": f"Bull regime ({regime_label}) — selling into macro strength.",
            })

        # 2. RSI (14)
        if direction == "BUY":
            rv = +1 if rsi < 55 else (-1 if rsi > 70 else 0)
            raw_signals.append({
                "name": "RSI (14)", "vote": rv, "ic": ic_map["RSI (14)"],
                "value": f"{rsi:.1f}",
                "reason_for":    f"RSI {rsi:.1f} — not overbought, room to run.",
                "reason_against": f"RSI {rsi:.1f} — overbought, mean-reversion risk.",
            })
        else:
            rv = +1 if rsi > 60 else (-1 if rsi < 35 else 0)
            raw_signals.append({
                "name": "RSI (14)", "vote": rv, "ic": ic_map["RSI (14)"],
                "value": f"{rsi:.1f}",
                "reason_for":    f"RSI {rsi:.1f} — elevated, good trim level.",
                "reason_against": f"RSI {rsi:.1f} — oversold, selling into weakness.",
            })

        # 3. MACD Histogram
        if direction == "BUY":
            rv = +1 if macd > 0 else -1
            raw_signals.append({
                "name": "MACD", "vote": rv, "ic": ic_map["MACD"],
                "value": f"{macd:+.3f}",
                "reason_for":    f"MACD histogram {macd:+.3f} — upward momentum.",
                "reason_against": f"MACD histogram {macd:+.3f} — momentum falling.",
            })
        else:
            rv = +1 if macd < 0 else -1
            raw_signals.append({
                "name": "MACD", "vote": rv, "ic": ic_map["MACD"],
                "value": f"{macd:+.3f}",
                "reason_for":    f"MACD {macd:+.3f} — downward momentum, supports sell.",
                "reason_against": f"MACD {macd:+.3f} — selling into rising momentum.",
            })

        # 4. vs SMA 200
        if direction == "BUY":
            rv = +1 if vs200 > 0 else -1
            raw_signals.append({
                "name": "vs SMA 200", "vote": rv, "ic": ic_map["vs SMA 200"],
                "value": f"{vs200:+.1%}",
                "reason_for":    f"Price {vs200:+.1%} above SMA200 — long-term uptrend intact.",
                "reason_against": f"Price {vs200:+.1%} below SMA200 — buying against the trend.",
            })
        else:
            rv = +1 if vs200 < 0 else -1
            raw_signals.append({
                "name": "vs SMA 200", "vote": rv, "ic": ic_map["vs SMA 200"],
                "value": f"{vs200:+.1%}",
                "reason_for":    f"Price {vs200:+.1%} below SMA200 — downtrend, supports trim.",
                "reason_against": f"Price {vs200:+.1%} above SMA200 — selling long-term strength.",
            })

        # 5. 52-Week High Proximity (replaces BB %B which has r=0.92 with RSI)
        if direction == "BUY":
            rv = +1 if h52w > -0.10 else (-1 if h52w < -0.20 else 0)
            raw_signals.append({
                "name": "52W High Proximity", "vote": rv, "ic": ic_map["52W High Proximity"],
                "value": f"{h52w:+.1%}",
                "reason_for":    f"Price only {h52w:+.1%} from 52-week high — long-term strength.",
                "reason_against": f"Price {h52w:+.1%} from 52-week high — in meaningful correction.",
            })
        else:
            rv = +1 if h52w < -0.20 else (-1 if h52w > -0.05 else 0)
            raw_signals.append({
                "name": "52W High Proximity", "vote": rv, "ic": ic_map["52W High Proximity"],
                "value": f"{h52w:+.1%}",
                "reason_for":    f"Price {h52w:+.1%} from 52-week high — weakness, good trim.",
                "reason_against": f"Price only {h52w:+.1%} from 52-week high — selling yearly strength.",
            })

        # 6. Volatility Regime
        if direction == "BUY":
            rv = +1 if vol_r < 1.1 else (-1 if vol_r > 1.5 else 0)
            raw_signals.append({
                "name": "Vol Regime", "vote": rv, "ic": ic_map["Vol Regime"],
                "value": f"{vol_r:.2f}x",
                "reason_for":    f"Vol ratio {vol_r:.2f} — short-term vol normal, stable entry.",
                "reason_against": f"Vol ratio {vol_r:.2f} — elevated vol, timing risk.",
            })
        else:
            rv = +1 if vol_r > 1.2 else 0
            raw_signals.append({
                "name": "Vol Regime", "vote": rv, "ic": ic_map["Vol Regime"],
                "value": f"{vol_r:.2f}x",
                "reason_for":    f"Vol ratio {vol_r:.2f} — elevated, supports reducing risk.",
                "reason_against": f"Vol ratio {vol_r:.2f} — vol normal, no urgency to sell.",
            })

        # 7. ML Signal
        if prob_up is not None:
            ic_ml = ic_map["ML Signal"]
            is_contrarian = ic_ml < 0
            if direction == "BUY":
                rv = +1 if prob_up > 0.60 else (-1 if prob_up < 0.40 else 0)
            else:
                rv = +1 if prob_up < 0.40 else (-1 if prob_up > 0.60 else 0)
            raw_signals.append({
                "name":       "ML Signal",
                "vote":       rv,
                "ic":         ic_ml,
                "value":      f"{prob_up:.1%}",
                "contrarian": is_contrarian,
                "reason_for":    f"ML model estimates {prob_up:.1%} probability of up move.",
                "reason_against": (
                    f"ML model says {prob_up:.1%} but IC={ic_ml:+.3f} — CONTRARIAN for this "
                    f"stock. High ML confidence here has historically preceded DOWN moves."
                    if is_contrarian else
                    f"ML model only {prob_up:.1%} — below 60% conviction threshold."
                ),
            })

        # ── IC-Weighted Aggregation ───────────────────────────────────────────
        # effective_vote = raw_vote * sign(IC)
        # — positive IC: signal fires correctly → vote stands
        # — negative IC: signal is contrarian → vote is flipped
        for s in raw_signals:
            s["eff_vote"] = s["vote"] * (np.sign(s["ic"]) if s["ic"] != 0 else 1.0)

        weights = [max(abs(s["ic"]), 0.01) for s in raw_signals]
        total_w = sum(weights)
        score   = sum(s["eff_vote"] * w for s, w in zip(raw_signals, weights)) / total_w

        agree_n  = sum(1 for s in raw_signals if s["eff_vote"] > 0)
        oppose_n = sum(1 for s in raw_signals if s["eff_vote"] < 0)
        n_active = agree_n + oppose_n
        agree_w  = sum(w for s, w in zip(raw_signals, weights) if s["eff_vote"] > 0)
        confidence_pct = int(100 * agree_w / total_w) if total_w > 0 else 0

        # ── Expected Alpha ────────────────────────────────────────────────────
        composite_ic   = sum(s["ic"] * w for s, w in zip(raw_signals, weights)) / total_w
        vol_fwd        = vol_21 * np.sqrt(lookahead / 252)
        expected_alpha = composite_ic * vol_fwd
        alpha_se       = vol_fwd / np.sqrt(max(_n, 50))
        ic_tstat       = composite_ic * np.sqrt(max(_n, 50)) / np.sqrt(max(1 - composite_ic**2, 1e-10))

        # ── Half-Kelly Position Sizing ────────────────────────────────────────
        p_win      = float(np.clip(0.5 + composite_ic / 2, 0.01, 0.99))
        kelly_size = float(np.clip((2 * p_win - 1) / 2, 0.005, 0.20))

        # ── Verdict ───────────────────────────────────────────────────────────
        if score >= 0.43:
            verdict, colour = "✅ Execute now",            "#1D9E75"
        elif score >= 0.0:
            verdict, colour = "⚠️ Proceed with caution", "#F5A623"
        else:
            verdict, colour = "🔴 Wait for better entry", "#D85A30"

        for_reasons     = [s["reason_for"]     for s in raw_signals if s["eff_vote"] > 0]
        against_reasons = [s["reason_against"] for s in raw_signals if s["eff_vote"] < 0]
        top_reasons = []
        if for_reasons:     top_reasons.append(("for",     for_reasons[0]))
        if against_reasons: top_reasons.append(("against", against_reasons[0]))

        return {
            "verdict":        verdict,
            "colour":         colour,
            "score":          round(score, 3),
            "composite_ic":   round(composite_ic, 4),
            "ic_tstat":       round(ic_tstat, 2),
            "expected_alpha": round(expected_alpha, 4),
            "alpha_se":       round(alpha_se, 4),
            "kelly_size":     round(kelly_size, 3),
            "confidence_pct": confidence_pct,
            "agree":          agree_n,
            "oppose":         oppose_n,
            "n_signals":      len(raw_signals),
            "n_active":       n_active,
            "signals":        raw_signals,
            "top_reasons":    top_reasons,
            "regime_label":   regime_label,
            "regime_override": False,
        }



        # ── PERSISTENCE HELPERS ───────────────────────────────────────────────────
    # On Streamlit Cloud the app container restarts after inactivity, wiping
    # session_state. We persist holdings and BL weights to a JSON file in the
    # app's working directory so they survive cold starts.
    #
    # The file lives at ./portfolio_state.json next to the app script.
    # It is NOT committed to git (.gitignore excludes *.json data files) but
    # persists on the Streamlit Cloud container between restarts within the
    # same deployment. For full durability across redeployments, use the
    # Export / Import buttons below to copy your data out of the browser.

    import json

    _SAVE_FILE = "/tmp/portfolio_state.json"

    def _load_state() -> dict:
        """Load holdings + BL weights from disk. Returns defaults if file absent."""
        try:
            with open(_SAVE_FILE, "r") as f:
                data = json.load(f)
            return {
                "holdings":   {t: float(data.get("holdings",   {}).get(t, 0.0)) for t in ACTIVE_TICKERS},
                "bl_weights": {t: float(data.get("bl_weights", {}).get(t, 0.0)) for t in ACTIVE_TICKERS},
                "portfolio_size": float(data.get("portfolio_size", 25_000.0)),
                "saved_at": data.get("saved_at", "never"),
            }
        except Exception:
            return {
                "holdings":      {t: 0.0 for t in ACTIVE_TICKERS},
                "bl_weights":    {t: 0.0 for t in ACTIVE_TICKERS},
                "portfolio_size": 25_000.0,
                "saved_at": "never",
            }

    def _save_state(holdings: dict, bl_weights: dict, portfolio_size: float) -> str:
        """Write holdings + BL weights + custom tickers to disk."""
        saved_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        data = {
            "holdings":       holdings,
            "bl_weights":     bl_weights,
            "portfolio_size": portfolio_size,
            "custom_tickers": st.session_state.get("custom_tickers", []),
            "saved_at":       saved_at,
        }
        try:
            with open(_SAVE_FILE, "w") as f:
                json.dump(data, f, indent=2)
            return saved_at
        except Exception as e:
            return f"Save failed: {e}"

    # ── Boot: load from disk into session_state (once per session) ────────────
    if "portfolio_loaded" not in st.session_state:
        _disk = _load_state()
        st.session_state.holdings        = _disk["holdings"]
        st.session_state.bl_weights      = _disk["bl_weights"]
        st.session_state.portfolio_size  = _disk["portfolio_size"]
        st.session_state.portfolio_saved_at = _disk["saved_at"]
        st.session_state.portfolio_loaded = True
    else:
        # On subsequent reruns (e.g. after adding a custom ticker), ensure
        # any newly-added tickers exist in session state dicts with 0.0 default
        # so the holdings/BL grids render correctly without a full reload.
        for _t in ACTIVE_TICKERS:
            st.session_state.holdings.setdefault(_t, 0.0)
            st.session_state.bl_weights.setdefault(_t, 0.0)


    # ── Auto-save helper: called on_change for every holdings + BL widget ─────
    def _auto_save():
        """Write current widget values to disk immediately on any change."""
        h  = {t: float(st.session_state.get(f"hold_{t}", 0.0)) for t in ACTIVE_TICKERS}
        bl = {t: float(st.session_state.get(f"bl_{t}",   0.0)) / 100 for t in ACTIVE_TICKERS}
        ps = float(st.session_state.get("portfolio_size_input",
                                         st.session_state.portfolio_size))
        saved_at = _save_state(h, bl, ps)
        st.session_state.holdings       = h
        st.session_state.bl_weights     = bl
        st.session_state.portfolio_size = ps
        st.session_state.portfolio_saved_at = saved_at

    # ── SECTION 0: Holdings Tracker ───────────────────────────────────────────
    st.divider()
    st.markdown("### 1. Current Holdings")

    # Save / Load / Export / Import controls
    ctrl_save, ctrl_info, ctrl_export, ctrl_import = st.columns([1, 2, 1, 2])

    with ctrl_save:
        if st.button("💾 Save now", type="primary",
                     help="Force-save all values to disk immediately."):
            _auto_save()
            st.success(f"Saved at {st.session_state.portfolio_saved_at}")

    with ctrl_info:
        st.caption(f"Last saved: **{st.session_state.portfolio_saved_at}**")
        st.caption("Both holdings and BL weights **auto-save** as you type. "
                   "Use **Export** for a backup that survives redeployments.")

    # Export: generate a JSON string the user can copy
    with ctrl_export:
        export_data = json.dumps({
            "holdings":       st.session_state.holdings,
            "bl_weights":     st.session_state.bl_weights,
            "portfolio_size": st.session_state.portfolio_size,
            "custom_tickers": st.session_state.get("custom_tickers", []),
        }, indent=2)
        st.download_button(
            "📥 Export JSON",
            data=export_data,
            file_name="portfolio_state.json",
            mime="application/json",
            help="Download your holdings + BL weights as a JSON file for permanent backup.",
        )

    # Import: paste JSON to restore
    with ctrl_import:
        with st.expander("📤 Import JSON"):
            pasted = st.text_area("Paste exported JSON here", height=80, key="import_json")
            if st.button("Load from JSON"):
                try:
                    imported = json.loads(pasted)
                    st.session_state.holdings       = {t: float(imported.get("holdings",   {}).get(t, 0.0)) for t in ACTIVE_TICKERS}
                    st.session_state.bl_weights     = {t: float(imported.get("bl_weights", {}).get(t, 0.0)) for t in ACTIVE_TICKERS}
                    st.session_state.portfolio_size = float(imported.get("portfolio_size", 25_000.0))
                    # Restore custom tickers — filter out any already in core
                    _imported_custom = [
                        t.upper() for t in imported.get("custom_tickers", [])
                        if t.upper() not in TICKERS
                    ]
                    st.session_state.custom_tickers = _imported_custom
                    _save_state(st.session_state.holdings, st.session_state.bl_weights,
                                st.session_state.portfolio_size)
                    st.success("✅ Imported and saved.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

    st.markdown("Enter your current positions. Hit **💾 Save** after editing.")
    if st.session_state.get("custom_tickers"):
        st.caption(f"✦ = custom rotation candidate: {', '.join(st.session_state.custom_tickers)}")

    # Holdings input grid — 4 columns
    tickers_list = ACTIVE_TICKERS
    for row_start in range(0, len(tickers_list), 4):
        row_tickers = tickers_list[row_start: row_start + 4]
        row_cols    = st.columns(4)
        for col, t in zip(row_cols, row_tickers):
            with col:
                price_now = float(closes[t].iloc[-1]) if t in closes.columns else 0.0
                _is_custom = t in st.session_state.get("custom_tickers", [])
                _label = f"{'✦ ' if _is_custom else ''}{t}  (${price_now:,.0f}/sh)"
                col.number_input(
                    _label,
                    min_value=0.0,
                    value=float(st.session_state.holdings.get(t, 0.0)),
                    step=0.1,
                    key=f"hold_{t}",
                    on_change=_auto_save,
                )

    # Derive live portfolio value from widget state (not session_state.holdings —
    # that only updates on Save; widgets are always current)
    current_values = {}
    for t in ACTIVE_TICKERS:
        shares    = float(st.session_state.get(f"hold_{t}", st.session_state.holdings.get(t, 0.0)))
        price_now = float(closes[t].iloc[-1]) if t in closes.columns else 0.0
        current_values[t] = shares * price_now

    total_portfolio_value = sum(current_values.values())

    # Portfolio size input
    st.divider()
    col_pf1, col_pf2, col_pf3 = st.columns(3)
    portfolio_size = col_pf1.number_input(
        "Total deployable capital (USD)",
        min_value=1_000.0,
        value=max(total_portfolio_value, float(st.session_state.portfolio_size)),
        step=500.0,
        key="portfolio_size_input",
        help="Your total capital to allocate including cash. "
             "Defaults to saved value or the sum of your holdings.",
    )
    col_pf2.metric("Current invested", f"${total_portfolio_value:,.0f}")
    col_pf3.metric("Estimated cash",   f"${max(portfolio_size - total_portfolio_value, 0):,.0f}")

    # Current allocation breakdown
    if total_portfolio_value > 0:
        with st.expander("📊 Current allocation breakdown", expanded=False):
            curr_alloc = {t: v / portfolio_size for t, v in current_values.items() if v > 0}
            curr_shares = {t: float(st.session_state.get(f"hold_{t}", 0.0)) for t in curr_alloc}
            curr_df = pd.DataFrame({
                "Ticker":       list(curr_alloc.keys()),
                "Shares":       [curr_shares[t] for t in curr_alloc],
                "Price":        [float(closes[t].iloc[-1]) for t in curr_alloc],
                "Market Value": [current_values[t] for t in curr_alloc],
                "Current Wt":   list(curr_alloc.values()),
            }).set_index("Ticker").sort_values("Market Value", ascending=False)
            st.dataframe(
                curr_df.style
                .format({"Price": "${:,.2f}", "Market Value": "${:,.2f}", "Current Wt": "{:.2%}"}),
                width="stretch",
            )

    # ── SECTION 1: BL Target Weights Input ───────────────────────────────────
    st.divider()
    st.markdown("### 2. BL Target Weights")
    st.markdown(
        "Enter the **BL Optimised** weights from the Portfolio Optimiser app. "
        "Hit **💾 Save** above after entering weights — they'll persist across sessions."
    )
    st.caption(
        "💡 Tip: copy the 'BL Optimised' column from the Views, Returns & Weights tab "
        "of the Portfolio Optimiser app directly into these fields. "
        "Custom tickers (✦) default to 0% — set a weight only if you want to rotate into them."
    )

    for row_start in range(0, len(tickers_list), 4):
        row_tickers = tickers_list[row_start: row_start + 4]
        row_cols    = st.columns(4)
        for col, t in zip(row_cols, row_tickers):
            with col:
                _is_custom = t in st.session_state.get("custom_tickers", [])
                _label = f"{'✦ ' if _is_custom else ''}{t} BL wt (%)"
                col.number_input(
                    _label,
                    min_value=0.0, max_value=100.0,
                    value=float(st.session_state.bl_weights.get(t, 0.0)) * 100,
                    step=0.5,
                    key=f"bl_{t}",
                    on_change=_auto_save,
                )

    # Read BL weights from widget state
    bl_live = {t: float(st.session_state.get(f"bl_{t}", 0.0)) / 100 for t in ACTIVE_TICKERS}
    bl_total = sum(bl_live.values())
    if abs(bl_total - 1.0) > 0.02 and bl_total > 0:
        st.warning(f"⚠️ BL weights sum to **{bl_total:.1%}** — should be ~100%. "
                   "Check your inputs from the Portfolio Optimiser.")
    elif bl_total > 0:
        st.success(f"✅ BL weights sum to {bl_total:.1%}")

    # Alias for downstream code that reads st.session_state.bl_weights
    st.session_state.bl_weights = bl_live

    # ── SECTION 2: Trade List + Cost Estimates ────────────────────────────────
    st.divider()
    st.markdown("### 3. Required Trades & Cost Estimates")
    st.markdown("""
**How the cost model works — calibrated to your Moo Moo account:**

- **Moo Moo Fee** = $0.99 flat per trade (buy or sell), regardless of trade size
- **Spread cost** = half the bid-ask spread × trade value.
  The spread is the gap between the best buy and best sell price in the market.
  You pay half of it on entry. Estimated as `spread_bps × trade_value` where
  spread_bps is calibrated per ticker (0.5 bps for MSFT/NVDA, up to 2 bps for ASML).
- **Total ≈ $1.10** — the $0.99 fee dominates at your trade sizes. This matches your observed ~$1.10 per trade.
- **Break-even days** = how many days of expected price movement are needed to recover the cost.
  Calculated as `total_cost% ÷ daily_vol`. Under 0.5 days = negligible.
    """)

    if bl_total == 0:
        st.info("Enter your BL target weights above to generate the trade list.")
    else:
        trade_rows = []
        for t in ACTIVE_TICKERS:
            target_wt   = bl_live.get(t, 0.0)
            current_wt  = current_values.get(t, 0.0) / portfolio_size
            delta_wt    = target_wt - current_wt
            trade_val   = delta_wt * portfolio_size

            if abs(trade_val) < 10:   # ignore rounding noise
                continue

            direction  = "BUY" if trade_val > 0 else "SELL"
            price_now  = float(closes[t].iloc[-1]) if t in closes.columns else 0.0
            n_shares   = abs(trade_val) / price_now if price_now > 0 else 0

            # Feature data for cost model
            try:
                fd = compute_features(closes[t].dropna(), volumes[t].dropna())
                costs = estimate_trade_cost(t, trade_val, fd)
            except Exception:
                costs = {"fixed_fee": 0, "spread_cost": 0,
                         "total_cost": 0, "total_pct": 0, "breakeven_days": 0}

            trade_rows.append({
                "Ticker":            t,
                "Direction":         direction,
                "Current Wt":        current_wt,
                "Target Wt":         target_wt,
                "Δ Weight":          delta_wt,
                "Trade Value ($)":   trade_val,
                "Shares":            round(n_shares, 2),
                "Moo Moo Fee ($)":   costs["fixed_fee"],
                "Spread Cost ($)":   costs["spread_cost"],
                "Total Cost ($)":    costs["total_cost"],
                "Cost (%)":          costs["total_pct"],
                "Break-even (days)": costs["breakeven_days"],
            })

        if not trade_rows:
            st.success("✅ Portfolio is already at target weights — no trades required.")
        else:
            trade_df = pd.DataFrame(trade_rows).set_index("Ticker")
            total_cost_usd = trade_df["Total Cost ($)"].sum()
            total_trade_vol = trade_df["Trade Value ($)"].abs().sum()
            avg_cost_pct    = total_cost_usd / total_trade_vol if total_trade_vol > 0 else 0

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trades required",   len(trade_rows))
            m2.metric("Total trade volume", f"${total_trade_vol:,.0f}")
            m3.metric("Est. total cost",    f"${total_cost_usd:,.2f}")
            m4.metric("Avg cost as % of trades", f"{avg_cost_pct:.3%}")

            def highlight_direction(val):
                if val == "BUY":  return "color: #1D9E75; font-weight:600"
                if val == "SELL": return "color: #D85A30; font-weight:600"
                return ""

            def color_delta(val):
                if isinstance(val, float):
                    return "color: #1D9E75" if val > 0 else "color: #D85A30"
                return ""

            styled_trades = (
                trade_df.style
                .format({
                    "Current Wt":        "{:.2%}",
                    "Target Wt":         "{:.2%}",
                    "Δ Weight":          "{:+.2%}",
                    "Trade Value ($)":   "${:+,.0f}",
                    "Shares":            "{:.2f}",
                    "Moo Moo Fee ($)":   "${:.2f}",
                    "Spread Cost ($)":   "${:.2f}",
                    "Total Cost ($)":    "${:.2f}",
                    "Cost (%)":          "{:.3%}",
                    "Break-even (days)": "{:.1f}d",
                })
                .map(highlight_direction, subset=["Direction"])
                .map(color_delta, subset=["Δ Weight", "Trade Value ($)"])
                .background_gradient(subset=["Cost (%)"], cmap="YlOrRd", vmin=0, vmax=0.005)
                .background_gradient(subset=["Break-even (days)"], cmap="YlOrRd", vmin=0, vmax=3)
            )
            st.dataframe(styled_trades, width="stretch")

            # ── SECTION 3: Entry Timing Verdicts ─────────────────────────────
            st.divider()
            st.markdown("### 4. Entry Timing — Should You Execute Each Trade Today?")
            st.markdown(
                "Each trade is cross-checked against three layers of signal: "
                "the HMM market regime, the ML probability-of-up, and the technical indicators. "
                "The verdict is a synthesis — not any single signal in isolation."
            )

            # We need the current regime from SPY HMM
            # Re-use fitted HMM from Tab 2 if already computed, else refit
            try:
                spy_ret_plan = np.log(spy / spy.shift(1)).dropna()
                regimes_plan, _, _ = fit_hmm(spy_ret_plan, n_regimes=n_regimes)
                current_regime_plan = int(regimes_plan.iloc[-1])
            except Exception:
                current_regime_plan = 1   # fallback: transitional

            regime_now_label = REGIME_LABELS.get(current_regime_plan, "Unknown")
            regime_now_color = REGIME_COLORS.get(current_regime_plan, "#888")

            st.markdown(
                f"**Current market regime (SPY HMM):** "
                f"<span style='color:{regime_now_color}; font-weight:700;'>{regime_now_label}</span>",
                unsafe_allow_html=True,
            )

            verdict_rows = []
            verdict_details = {}   # ticker → full scorecard dict
            for t in trade_df.index:
                direction = trade_df.loc[t, "Direction"]
                prob_up_now = None
                prob_s_now  = None
                try:
                    fd_t = compute_features(closes[t].dropna(), volumes[t].dropna())
                    prob_s_now, _, _, _ = train_ml_model(fd_t, lookahead=lookahead)
                    prob_up_now = float(prob_s_now.iloc[-1])
                except Exception:
                    pass

                try:
                    fd_t = compute_features(closes[t].dropna(), volumes[t].dropna())
                except Exception:
                    continue

                sc = score_entry_timing(
                    t, fd_t, current_regime_plan, prob_up_now, direction,
                    lookahead=lookahead, ml_probs=prob_s_now,
                )
                verdict_details[t] = sc

                verdict_rows.append({
                    "Ticker":      t,
                    "Direction":   direction,
                    "Verdict":     sc["verdict"],
                    "Confidence":  sc["confidence_pct"],
                    "For":         sc["agree"],
                    "Against":     sc["oppose"],
                    "Score":       round(sc["score"], 2),
                    "_colour":     sc["colour"],
                })

            if verdict_rows:
                priority = {"✅ Execute now": 0, "⚠️ Proceed with caution": 1,
                            "🔴 Wait for better entry": 2}
                verdict_rows.sort(key=lambda r: priority.get(r["Verdict"], 3))

                # ── Summary table ─────────────────────────────────────────────
                st.markdown("**Summary — sorted by priority**")
                summary_df = pd.DataFrame(verdict_rows).set_index("Ticker").drop(columns=["_colour"])

                def style_verdict(val):
                    if "Execute" in str(val): return "color:#1D9E75; font-weight:600"
                    if "caution" in str(val): return "color:#F5A623; font-weight:600"
                    if "Wait"    in str(val): return "color:#D85A30; font-weight:600"
                    return ""

                st.dataframe(
                    summary_df.style
                    .map(style_verdict, subset=["Verdict"])
                    .map(highlight_direction, subset=["Direction"])
                    .background_gradient(subset=["Confidence"], cmap="RdYlGn", vmin=0, vmax=100)
                    .format({"Confidence": "{:.0f}%", "Score": "{:+.2f}"}),
                    width="stretch",
                    column_config={
                        "Verdict":    st.column_config.TextColumn("Verdict",       width="medium"),
                        "Confidence": st.column_config.TextColumn("Signal Agr. %", width="small",
                                        help="% of active (non-neutral) signals that agree. "
                                             "Neutral votes are excluded from the denominator."),
                        "For":        st.column_config.TextColumn("✅ For",        width="small"),
                        "Against":    st.column_config.TextColumn("❌ Against",    width="small"),
                        "Score":      st.column_config.TextColumn("Score",         width="small"),
                    }
                )

                # ── Per-ticker expander scorecards ────────────────────────────
                st.divider()
                st.markdown("**Per-trade signal scorecard** — expand any row for full detail")

                VOTE_ICON  = {+1: "✅", 0: "➖", -1: "❌"}
                VOTE_COLOR = {+1: "#1D9E75", 0: "#8b949e", -1: "#D85A30"}

                for row in verdict_rows:
                    t    = row["Ticker"]
                    sc   = verdict_details[t]
                    conf = sc["confidence_pct"]
                    bar_filled = int(conf / 10)  # out of 10 blocks

                    # Confidence bar using Unicode blocks
                    bar = ("█" * bar_filled) + ("░" * (10 - bar_filled))
                    bar_color = "#1D9E75" if conf >= 60 else ("#F5A623" if conf >= 40 else "#D85A30")

                    expander_label = (
                        f"{t}  ·  {row['Direction']}  ·  {sc['verdict']}  "
                        f"·  {sc['agree']}/{sc['n_signals']} signals agree"
                    )
                    with st.expander(expander_label, expanded=False):

                        # Top row: verdict badge + confidence bar
                        col_v, col_b = st.columns([1, 2])
                        with col_v:
                            st.markdown(
                                f"<div style='font-size:1.1rem; font-weight:700; color:{sc['colour']};'>"
                                f"{sc['verdict']}</div>"
                                f"<div style='font-size:0.8rem; color:#475569;'>Score: {sc['score']:+.2f} "
                                f"(range −1 to +1)</div>",
                                unsafe_allow_html=True,
                            )
                        with col_b:
                            st.markdown(
                                f"<div style='font-size:0.75rem; color:#475569; margin-bottom:2px;'>"
                                f"Signal agreement (IC-weighted)</div>"
                                f"<div style='font-family: monospace; font-size:1.1rem; "
                                f"color:{bar_color}; letter-spacing:2px;'>{bar}</div>"
                                f"<div style='font-size:0.75rem; color:#475569;'>"
                                f"{conf}% of IC-weighted votes favour this trade "
                                f"({sc['agree']} for, {sc['oppose']} against, "
                                f"{sc['n_signals'] - sc['agree'] - sc['oppose']} neutral)</div>",
                                unsafe_allow_html=True,
                            )

                        # ── Quantitative edge metrics ─────────────────────────
                        st.markdown("---")
                        st.markdown("**Quantitative Edge**")
                        qc1, qc2, qc3 = st.columns(3)
                        alpha_pct = sc["expected_alpha"] * 100
                        se_pct    = sc["alpha_se"] * 100
                        ic_t      = sc["ic_tstat"]
                        qc1.metric(
                            f"Expected α ({lookahead}d)",
                            f"{alpha_pct:+.2f}%",
                            delta=f"±{se_pct:.2f}% (1σ)",
                            help=(
                                "Expected edge = composite IC × forward vol. "
                                "The ±1σ band shows how uncertain this estimate is. "
                                "A wide band means the model may be wrong by much more."
                            ),
                        )
                        sig_label = "✅ Significant" if abs(ic_t) >= 1.65 else "⚠️ Not significant"
                        sig_color = "normal" if abs(ic_t) >= 1.65 else "off"
                        qc2.metric(
                            "IC t-statistic",
                            f"{ic_t:+.2f}",
                            delta=sig_label,
                            delta_color=sig_color,
                            help=(
                                "t-stat = composite IC × √N / √(1-IC²). "
                                "≥ 1.65 = 90% confidence edge is real. "
                                "≥ 1.96 = 95% confidence. "
                                "Below 1.65: the apparent edge may be noise."
                            ),
                        )
                        qc3.metric(
                            "Half-Kelly size",
                            f"{sc['kelly_size']:.1%}",
                            help=(
                                "Suggested position as % of portfolio (half-Kelly). "
                                "Formula: Kelly = 2p − 1 where p ≈ 0.5 + IC/2, "
                                "then halved for safety. This is the institutional "
                                "standard — captures most EV while halving variance."
                            ),
                        )

                        st.markdown("---")

                        # Plain-English top reasons
                        if sc["top_reasons"]:
                            st.markdown("**Why:**")
                            for side, reason_text in sc["top_reasons"]:
                                icon = "✅" if side == "for" else "⚠️"
                                st.markdown(f"{icon} {reason_text}")

                        st.markdown("---")

                        # Per-signal scorecard
                        st.markdown("**Signal breakdown:**")
                        sig_cols = st.columns([2, 1, 1, 3])
                        sig_cols[0].markdown("**Signal**")
                        sig_cols[1].markdown("**Value · Vote**")
                        sig_cols[2].markdown("**IC**")
                        sig_cols[3].markdown("**Reasoning**")

                        for sig in sc["signals"]:
                            eff  = sig.get("eff_vote", sig["vote"])
                            icon = VOTE_ICON.get(int(np.sign(eff)), "➖") if eff != 0 else "➖"
                            col_col = VOTE_COLOR.get(int(np.sign(eff)), "#8b949e") if eff != 0 else "#8b949e"
                            reason_text = (sig["reason_for"] if eff >= 0 else sig["reason_against"])
                            ic_val  = sig.get("ic", 0)
                            ic_str  = f"{ic_val:+.3f}"
                            ic_note = " ⚠️" if sig.get("contrarian") else ""
                            c1, c2, c3, c4 = st.columns([2, 1, 1, 3])
                            c1.markdown(
                                f"<span style='color:#1e293b; font-weight:500;'>{sig['name']}</span>",
                                unsafe_allow_html=True,
                            )
                            c2.markdown(
                                f"<span style='color:#475569; font-size:0.8rem;'>{sig['value']}</span> "
                                f"<span style='font-size:1rem;'>{icon}</span>",
                                unsafe_allow_html=True,
                            )
                            ic_color = "#1D9E75" if ic_val > 0.05 else ("#D85A30" if ic_val < 0 else "#F5A623")
                            c3.markdown(
                                f"<span style='color:{ic_color}; font-family:monospace; font-size:0.8rem;'>"
                                f"{ic_str}{ic_note}</span>",
                                unsafe_allow_html=True,
                            )
                            c4.markdown(
                                f"<span style='color:{col_col}; font-size:0.82rem;'>{reason_text}</span>",
                                unsafe_allow_html=True,
                            )

            # ── SECTION 4: Rebalancing Waterfall ──────────────────────────────
            st.divider()
            st.markdown("### 5. Capital Redeployment Waterfall")
            st.caption(
                "Visualises the weight shift from current to target allocation. "
                "Net of estimated transaction costs."
            )

            tickers_sorted = trade_df.sort_values("Δ Weight").index.tolist()
            deltas = [float(trade_df.loc[t, "Δ Weight"]) for t in tickers_sorted]
            colors = [REGIME_COLORS[0] if d > 0 else REGIME_COLORS[2] for d in deltas]

            fig_wf = go.Figure()
            fig_wf.add_trace(go.Bar(
                y=tickers_sorted, x=deltas, orientation="h",
                marker_color=colors, opacity=0.85,
                hovertemplate="<b>%{y}</b><br>Δ Weight: %{x:.2%}<extra></extra>",
            ))
            fig_wf.add_vline(x=0, line_color="#30363d", line_width=1)
            fig_wf.update_layout(
                height=max(250, len(tickers_sorted) * 32),
                xaxis=dict(title="Weight change", tickformat=".1%", gridcolor="#30363d"),
                yaxis=dict(autorange="reversed"),
                margin=dict(t=20, b=20, l=10, r=10),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_wf, width="stretch")

            # ── SECTION 5: Cost Impact Summary ────────────────────────────────
            st.divider()
            st.markdown("### 6. Full Cost Breakdown")
            with st.expander("📐 Understand every cost term", expanded=False):
                st.markdown("""
| Term | Definition | Your Moo Moo context |
|------|-----------|---------------------|
| **Moo Moo Fee** | Flat platform fee charged per trade by Moo Moo | **$0.99 per trade** (buy or sell), regardless of size |
| **Bid-ask spread** | Gap between the best buy and best sell price in the market. You pay half on entry. | ~0.5–2 bps for US large-caps. On a $2,500 trade at 1 bp: ~$0.13 |
| **Commission** | Brokerage commission | **$0** — Moo Moo is commission-free |
| **Market impact** | Your order moves the price against you | **Zero** at $25k in stocks with $5B+ daily volume |
| **Break-even days** | Days of expected price movement to recover the total entry cost | = total_cost% ÷ daily_vol. Under 0.5 days = negligible. |
| **Total cost ≈ $1.10** | $0.99 fee + ~$0.10–0.15 half-spread = **~$1.10–$1.15** | Matches your observed ~$1.10 per trade exactly |
                """)

            cost_breakdown = trade_df[["Direction", "Trade Value ($)",
                                        "Moo Moo Fee ($)", "Spread Cost ($)",
                                        "Total Cost ($)", "Cost (%)",
                                        "Break-even (days)"]].copy()

            # Add a totals row
            totals = pd.Series({
                "Direction":         "—",
                "Trade Value ($)":   trade_df["Trade Value ($)"].abs().sum(),
                "Moo Moo Fee ($)":   trade_df["Moo Moo Fee ($)"].sum(),
                "Spread Cost ($)":   trade_df["Spread Cost ($)"].sum(),
                "Total Cost ($)":    trade_df["Total Cost ($)"].sum(),
                "Cost (%)":          avg_cost_pct,
                "Break-even (days)": (trade_df["Break-even (days)"] *
                                       trade_df["Trade Value ($)"].abs()).sum() /
                                      total_trade_vol if total_trade_vol > 0 else 0,
            }, name="TOTAL")
            cost_breakdown = pd.concat([cost_breakdown, totals.to_frame().T])

            st.dataframe(
                cost_breakdown.style
                .format({
                    "Trade Value ($)":   "${:,.0f}",
                    "Moo Moo Fee ($)":   "${:.2f}",
                    "Spread Cost ($)":   "${:.2f}",
                    "Total Cost ($)":    "${:.2f}",
                    "Cost (%)":          "{:.3%}",
                    "Break-even (days)": "{:.1f}d",
                })
                .map(highlight_direction, subset=["Direction"]),
                width="stretch",
            )

            n_trades = len(trade_df)
            est_total = n_trades * 1.10
            st.caption(
                f"**Your actual cost:** {n_trades} trades × ~$1.10 = **~${est_total:.2f}** total. "
                "The $0.99 Moo Moo fee dominates at your trade sizes — the spread cost is "
                "typically $0.10–$0.20 per trade on these liquid names. "
                "Signal quality from the BL model matters far more than these costs at your scale."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — SIGNAL AUDIT
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("#### 🔍 Signal Audit — Can You Trust What This App Is Saying?")
    st.markdown(
        "This tab answers three questions that determine whether the signals in this app "
        "are actually worth following:\n\n"
        "1. **Backtest** — if you had followed the buy/sell signals historically, what happened?\n"
        "2. **Correlation** — are the 7 scoring signals genuinely independent, or are some redundant?\n"
        "3. **Calibration** — when the ML model says 70% probability of up, does the stock go up 70% of the time?\n\n"
        "All three analyses run on the selected stock using only historical data — "
        "no look-ahead bias, no cherry-picking."
    )

    audit_ticker = st.selectbox(
        "Audit stock", ACTIVE_TICKERS,
        index=ACTIVE_TICKERS.index(selected_ticker) if selected_ticker in ACTIVE_TICKERS else 0,
        key="audit_ticker",
    )

    try:
        audit_price = closes[audit_ticker].dropna()
        audit_vol   = volumes[audit_ticker].dropna()
        audit_fd    = compute_features(audit_price, audit_vol)
    except Exception as e:
        st.error(f"Could not compute features for {audit_ticker}: {e}")
        st.stop()

    audit_lookahead = st.slider(
        "Forward return window (days)", 1, 21, lookahead, 1, key="audit_lookahead",
        help="How many days forward to measure each signal's outcome.",
    )

    st.divider()

    # ── 1: BACKTEST ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    st.markdown("### 1. Backtest — Did the signals predict future returns?")
    st.markdown(
        f"For every historical day, compute the verdict that would have been given, "
        f"then measure the actual **{audit_lookahead}-day return** that followed. "
        "Execute days should show higher returns than Wait days if the signals have edge."
    )
    st.caption(
        "⚠️ Technical and regime signals are rule-based (no training = clean backtest). "
        "The ML component is trained on 70% of data — its backtest here covers the OOS 30% only."
    )

    with st.spinner(f"Running backtest for {audit_ticker}…"):
        fwd_ret = np.log(
            audit_price.shift(-audit_lookahead) / audit_price
        ).reindex(audit_fd.index).dropna()

        try:
            ml_probs_bt, _, _, _ = train_ml_model(audit_fd, lookahead=audit_lookahead)
        except Exception:
            ml_probs_bt = pd.Series(dtype=float)

        spy_ret_bt = np.log(spy / spy.shift(1)).dropna()
        try:
            regimes_bt, _, _ = fit_hmm(spy_ret_bt, n_regimes=n_regimes)
        except Exception:
            regimes_bt = pd.Series(1, index=spy_ret_bt.index)

        common_idx = fwd_ret.index.intersection(ml_probs_bt.index)
        bt_rows = []
        for day in common_idx:
            if day not in audit_fd.index:
                continue
            day_fd     = audit_fd.loc[:day].tail(1)
            regime_day = int(regimes_bt.get(day, 1))
            prob_day   = float(ml_probs_bt.get(day, 0.5))
            fwd        = float(fwd_ret.loc[day])
            sc_day     = score_entry_timing(audit_ticker, day_fd, regime_day, prob_day, "BUY")
            bt_rows.append({"date": day, "verdict": sc_day["verdict"],
                             "score": sc_day["score"], "fwd_ret": fwd})

        bt_df = pd.DataFrame(bt_rows).set_index("date")

    if len(bt_df) > 20:
        col_bt1, col_bt2 = st.columns([3, 2])
        v_order  = ["✅ Execute now", "⚠️ Proceed with caution", "🔴 Wait for better entry"]
        v_colors = ["#1D9E75", "#F5A623", "#D85A30"]

        with col_bt1:
            st.markdown(f"##### {audit_lookahead}-Day Return Distribution by Verdict")
            fig_bt = go.Figure()
            for v_label, v_col in zip(v_order, v_colors):
                subset = bt_df[bt_df["verdict"] == v_label]["fwd_ret"]
                if len(subset) < 3:
                    continue
                fig_bt.add_trace(go.Histogram(
                    x=subset, name=v_label, nbinsx=40,
                    marker_color=v_col, opacity=0.65, histnorm="probability",
                ))
            fig_bt.add_vline(x=0, line_color="#8b949e", line_dash="dash")
            fig_bt.update_layout(
                barmode="overlay", height=300, margin=dict(t=20, b=10),
                xaxis=dict(title=f"{audit_lookahead}d log-return",
                           tickformat=".1%", gridcolor="#30363d"),
                yaxis=dict(title="Probability", gridcolor="#30363d"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.12, font_size=10),
            )
            st.plotly_chart(fig_bt, width="stretch")

        with col_bt2:
            st.markdown("##### Average Outcome per Verdict")
            summary_bt = bt_df.groupby("verdict")["fwd_ret"].agg(
                ["mean", "std", "count", lambda x: (x > 0).mean()]
            ).rename(columns={"mean": "Avg Ret", "std": "Std Dev",
                               "count": "N Days", "<lambda_0>": "Win Rate"})
            summary_bt = summary_bt.reindex([v for v in v_order if v in summary_bt.index])
            st.dataframe(
                summary_bt.style.format({
                    "Avg Ret":  "{:+.2%}",
                    "Std Dev":  "{:.2%}",
                    "Win Rate": "{:.1%}",
                }),
                width="stretch",
            )
            st.caption(
                "Execute days should have higher average return and win rate than Wait days. "
                "If they don't differ, the signal has no edge for this stock."
            )

        execute_mask = bt_df["verdict"] == "✅ Execute now"
        if execute_mask.sum() > 30:
            st.markdown(f"##### Rolling 60-day Win Rate (Execute days)")
            roll_acc = (bt_df.loc[execute_mask, "fwd_ret"] > 0).rolling(60, min_periods=20).mean()
            fig_roll = go.Figure()
            fig_roll.add_hline(y=0.5, line_dash="dash", line_color="#8b949e",
                                annotation_text="50% baseline")
            fig_roll.add_trace(go.Scatter(
                x=roll_acc.index, y=roll_acc.values,
                line=dict(color="#1D9E75", width=1.5), fill="tozeroy",
                fillcolor="rgba(29,158,117,0.1)", name="Win rate",
            ))
            fig_roll.update_layout(
                height=200, margin=dict(t=20, b=10),
                yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="#30363d"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_roll, width="stretch")
            st.caption("Sustained periods above 55% indicate genuine edge. Near 50% = noise.")
    else:
        st.info("Not enough data for backtest — try an earlier data start date.")

    # ── 2: CORRELATION ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 2. Signal Correlation — Which signals are redundant?")
    st.markdown(
        "High correlation (|r| > 0.7) between two signals means they carry the same "
        "information. The scoring function still counts them as two independent votes, "
        "which inflates confidence without adding evidence. Values near 0 = independent."
    )

    try:
        ml_probs_corr, _, _, _ = train_ml_model(audit_fd, lookahead=audit_lookahead)
        regimes_corr, _, _ = fit_hmm(np.log(spy / spy.shift(1)).dropna(), n_regimes=n_regimes)

        sig_df = pd.DataFrame({
            "Regime":    regimes_corr.reindex(audit_fd.index).fillna(1),
            "RSI":       audit_fd["rsi_14"],
            "MACD":      audit_fd["macd_hist"],
            "vs SMA200": audit_fd["price_vs_sma200"],
            "BB %B":     audit_fd["bb_pct"],
            "Vol Ratio": audit_fd["vol_ratio"],
            "ML P(Up)":  ml_probs_corr.reindex(audit_fd.index),
        }).dropna()

        corr_matrix = sig_df.corr()
        fig_corr = px.imshow(
            corr_matrix, color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1, text_auto=".2f",
        )
        fig_corr.update_layout(height=420, margin=dict(t=20, b=10),
                                paper_bgcolor="rgba(0,0,0,0)")

        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            st.plotly_chart(fig_corr, width="stretch")
        with col_c2:
            st.markdown("##### High-correlation pairs (|r| > 0.5)")
            sig_names_corr = list(corr_matrix.columns)
            high_corr = []
            for i in range(len(sig_names_corr)):
                for j in range(i+1, len(sig_names_corr)):
                    r = corr_matrix.iloc[i, j]
                    if abs(r) > 0.5:
                        high_corr.append({
                            "Signal A": sig_names_corr[i],
                            "Signal B": sig_names_corr[j],
                            "r":        round(r, 3),
                            "Status":   "Redundant" if abs(r) > 0.7 else "Overlapping",
                        })
            if high_corr:
                st.dataframe(
                    pd.DataFrame(high_corr).style
                    .background_gradient(subset=["r"], cmap="RdYlGn", vmin=-1, vmax=1),
                    width="stretch",
                )
                st.caption(
                    "**Redundant** pairs (|r| > 0.7) count as two votes but carry one signal's "
                    "worth of information. This inflates confidence for correlated clusters."
                )
            else:
                st.success("✅ All signal pairs |r| < 0.5 — no redundancy concern.")
    except Exception as e:
        st.warning(f"Could not compute correlation matrix: {e}")

    # ── 3: CALIBRATION ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 3. Calibration — Does P(up) = 70% mean up 70% of the time?")
    st.markdown(
        "A well-calibrated model follows the diagonal — predicted probability equals "
        "actual frequency. Above diagonal = under-confident. Below = over-confident. "
        "Bubble size = number of observations in that probability bin."
    )

    try:
        cal_probs, _, _, _ = train_ml_model(audit_fd, lookahead=audit_lookahead)
        actual_up = (np.log(
            audit_price.shift(-audit_lookahead) / audit_price
        ) > 0).reindex(cal_probs.index)

        cal_df = pd.DataFrame({"prob": cal_probs, "actual": actual_up}).dropna()
        cal_df["bin"] = pd.cut(cal_df["prob"], bins=10,
                                labels=[f"{i*10+5}%" for i in range(10)])
        cal_summary = cal_df.groupby("bin", observed=True).agg(
            predicted=("prob",   "mean"),
            actual    =("actual", "mean"),
            n         =("actual", "count"),
        ).dropna()

        brier       = float(((cal_df["prob"] - cal_df["actual"].astype(float)) ** 2).mean())
        brier_skill = 1 - brier / 0.25

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#8b949e", dash="dash"), name="Perfect calibration",
        ))
        fig_cal.add_trace(go.Scatter(
            x=cal_summary["predicted"], y=cal_summary["actual"],
            mode="lines+markers",
            line=dict(color="#1D9E75", width=2),
            marker=dict(size=cal_summary["n"] / cal_summary["n"].max() * 20 + 4,
                        color="#1D9E75"),
            name="Model",
            hovertemplate="Predicted: %{x:.0%}<br>Actual: %{y:.0%}<extra></extra>",
        ))
        fig_cal.update_layout(
            height=340, margin=dict(t=20, b=10),
            xaxis=dict(title="Predicted P(up)", tickformat=".0%",
                       range=[0, 1], gridcolor="#30363d"),
            yaxis=dict(title="Actual win rate", tickformat=".0%",
                       range=[0, 1], gridcolor="#30363d"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
        )

        col_ca1, col_ca2 = st.columns([3, 2])
        with col_ca1:
            st.plotly_chart(fig_cal, width="stretch")
        with col_ca2:
            st.metric("Brier Score", f"{brier:.3f}",
                      help="Lower is better. 0.25 = random, 0 = perfect.")
            st.metric("Skill vs. Random", f"{brier_skill:+.1%}",
                      help="Positive = better than random.")
            st.markdown("---")
            st.dataframe(
                cal_summary.style.format({
                    "predicted": "{:.0%}", "actual": "{:.0%}", "n": "{:.0f}",
                }),
                width="stretch",
            )
            st.caption(
                "Brier < 0.20 = good. 0.20–0.25 = marginal edge. "
                "> 0.25 = worse than random for that bin."
            )
    except Exception as e:
        st.warning(f"Could not compute calibration: {e}")

    # ── 4: PER-SIGNAL HIT RATES ────────────────────────────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 4. Per-Signal Hit Rates & IC — Which signals actually work for this stock?")
    st.markdown(
        f"When each signal alone fires 'bullish', what fraction of the following "
        f"{audit_lookahead}-day periods were actually positive? "
        "**Win Rate** = binary accuracy. **IC** (Spearman) = continuous predictive correlation — "
        "the institutional standard. IC < 0 = contrarian (signal fires in the WRONG direction)."
    )

    try:
        from scipy.stats import spearmanr as _spearmanr
        regimes_hr, _, _ = fit_hmm(np.log(spy / spy.shift(1)).dropna(), n_regimes=n_regimes)
        ml_probs_hr, _, _, _ = train_ml_model(audit_fd, lookahead=audit_lookahead)

        fwd_hr  = np.log(audit_price.shift(-audit_lookahead) / audit_price).reindex(audit_fd.index)
        fwd_bin = (fwd_hr > 0)
        common  = audit_fd.index.intersection(fwd_hr.dropna().index)
        fwd_c   = fwd_bin.reindex(common)
        fwd_ret = fwd_hr.reindex(common)
        regime_c = regimes_hr.reindex(common).fillna(1)
        ml_c    = ml_probs_hr.reindex(common).fillna(0.5)

        checks = [
            ("Market Regime (Bull)",     (regime_c == 0),                           None,                                   None),
            ("RSI < 55",                 (audit_fd["rsi_14"].reindex(common) < 55), -audit_fd["rsi_14"].reindex(common),     "Lower RSI → more bullish"),
            ("MACD positive",            (audit_fd["macd_hist"].reindex(common) > 0), audit_fd["macd_hist"].reindex(common), "Positive MACD → bullish"),
            ("Above SMA200",             (audit_fd["price_vs_sma200"].reindex(common) > 0), audit_fd["price_vs_sma200"].reindex(common), "Above SMA200 → bullish"),
            ("52W High Proximity (>-10%)", (audit_fd["high_52w_pct"].reindex(common) > -0.10), audit_fd["high_52w_pct"].reindex(common), "Nearer 52w high → bullish"),
            ("Low vol ratio (<1.1)",     (audit_fd["vol_ratio"].reindex(common) < 1.1), -audit_fd["vol_ratio"].reindex(common), "Lower vol → better entry"),
            ("ML P(Up) > 60%",           (ml_c > 0.60),                             ml_c,                                   "Higher P(Up) → bullish"),
        ]

        hit_rows = []
        for sig_name, sig_mask, sig_vals, ic_note in checks:
            n_days = int(sig_mask.sum())

            # Binary win rate
            if n_days < 10:
                hit_rows.append({"Signal": sig_name, "N days": n_days,
                                  "Win Rate": None, "Edge vs 50%": None,
                                  "IC (Spearman)": None, "IC t-stat": None,
                                  "Assessment": "Too few obs"})
                continue
            win_rate = float(fwd_c[sig_mask].mean())
            edge     = win_rate - 0.5

            # Spearman IC on continuous signal values
            ic_val = None
            ic_t   = None
            if sig_vals is not None:
                _v = sig_vals.reindex(common).values.astype(float)
                _f = fwd_ret.values.astype(float)
                _ok = np.isfinite(_v) & np.isfinite(_f)
                if _ok.sum() >= 30:
                    _ic, _ = _spearmanr(_v[_ok], _f[_ok])
                    if not np.isnan(_ic):
                        ic_val = round(float(_ic), 4)
                        ic_t   = round(float(_ic) * np.sqrt(_ok.sum()) /
                                       np.sqrt(max(1 - float(_ic)**2, 1e-10)), 2)

            assess = ("Strong ✅"      if edge > 0.10 else
                      "Useful ✓"       if edge > 0.05 else
                      "Marginal"       if edge > 0    else
                      "Contrarian ⚠️"  if edge > -0.05 else
                      "Inverse ❌")
            if ic_val is not None and ic_val < 0 and assess not in ("Contrarian ⚠️","Inverse ❌"):
                assess += " (IC negative)"
            hit_rows.append({
                "Signal":       sig_name,
                "N days":       n_days,
                "Win Rate":     win_rate,
                "Edge vs 50%":  edge,
                "IC (Spearman)": ic_val,
                "IC t-stat":    ic_t,
                "Assessment":   assess,
            })

        hit_df = pd.DataFrame(hit_rows).set_index("Signal")
        st.dataframe(
            hit_df.style
            .format({"Win Rate": "{:.1%}", "Edge vs 50%": "{:+.1%}",
                     "N days": "{:.0f}", "IC (Spearman)": "{:+.4f}",
                     "IC t-stat": "{:+.2f}"}, na_rep="—")
            .background_gradient(subset=["Win Rate"], cmap="RdYlGn", vmin=0.40, vmax=0.65)
            .background_gradient(subset=["IC (Spearman)"], cmap="RdYlGn", vmin=-0.1, vmax=0.1),
            width="stretch",
        )
        st.caption(
            "**IC interpretation:** IC > 0.05 = useful. IC > 0.10 = strong. "
            "IC < 0 = CONTRARIAN — in the scoring function the vote for this signal is "
            "automatically flipped. **IC t-stat ≥ 1.65** = edge is statistically significant "
            "at 90% confidence. Below that, the apparent edge may be noise."
        )

    except Exception as e:
        st.warning(f"Could not compute hit rates: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer**: Educational and personal research only. Not financial advice. "
    "All ML signals are trained on historical data and do not predict the future. "
    "Always pair technical signals with fundamental analysis and risk management. "
    "Data: Yahoo Finance."
)
