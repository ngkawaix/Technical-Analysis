# 📊 Ka Wai's Quantitative Portfolio Suite

A two-app system for systematic portfolio management combining fundamental analysis,
machine learning, and daily technical signals.

---

## Architecture

```
portfolio-suite/
│
├── technical_analysis_app.py   ← THIS REPO  (ML Technical Analysis)
│   Tabs: Overview · Technicals · Regime Detection ·
│         ML Signals · LSTM Forecast · Universe Scanner · Rebalancing Planner
│
└── app.py                      ← SEPARATE REPO  (Portfolio Optimiser DCF-BL)
    Tabs: Introduction · DCF Models · Confidence ·
          Views & Weights · Simulation & Stress Tests · Strategy Comparison
```

The two apps are **deliberately separate** but designed to be used together:

| App | Time horizon | Question it answers |
|-----|-------------|-------------------|
| **Portfolio Optimiser (DCF-BL)** | Quarterly | What *should* my allocation be? |
| **Technical Analysis (ML)**       | Daily      | *When* should I execute that allocation, and at what cost? |

The **Rebalancing Planner** tab in the technical app is the bridge: paste your BL
target weights from the Portfolio Optimiser, enter your current holdings, and get
a prioritised trade list with timing verdicts.

---

## Deployment — Technical Analysis App

### Option A: Streamlit Community Cloud (recommended, free)

**Step 1 — Create a GitHub repo**

```bash
git init portfolio-technical-analysis
cd portfolio-technical-analysis
```

**Step 2 — Copy files into the repo**

Your repo should look exactly like this:

```
portfolio-technical-analysis/
├── technical_analysis_app.py   ← the main app (rename to app.py if you prefer)
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── .gitignore
└── README.md
```

**Step 3 — Push to GitHub**

```bash
git add .
git commit -m "Initial deployment"
git remote add origin https://github.com/YOUR_USERNAME/portfolio-technical-analysis.git
git push -u origin main
```

**Step 4 — Deploy on Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub account and select the repo
4. Set **Main file path** to `technical_analysis_app.py`
5. Click **Deploy**

Streamlit Cloud will install `requirements.txt` automatically. First deployment
takes 3–8 minutes (TensorFlow is large). Subsequent deployments are faster.

---

### Option B: Run locally

```bash
# 1. Clone your repo
git clone https://github.com/YOUR_USERNAME/portfolio-technical-analysis.git
cd portfolio-technical-analysis

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run technical_analysis_app.py
```

The app opens at `http://localhost:8501`.

---

## Deployment — Portfolio Optimiser (DCF-BL App)

### Critical: the edhec_risk_kit_final.py module

The BL app imports a local module from the EDHEC Advanced Portfolio Construction
course:

```python
import edhec_risk_kit_final as erk
```

This file is **not on PyPI** — it came with your course materials. You must include
it in the same directory as `app.py` in your repo.

```
portfolio-bl-app/
├── app.py                      ← the BL Portfolio Optimiser
├── edhec_risk_kit_final.py     ← REQUIRED — copy from your course files
├── requirements_bl_app.txt     ← rename to requirements.txt in this repo
├── .streamlit/
│   └── config.toml
├── .gitignore
└── README.md
```

> ⚠️ **Check your EDHEC course portal or local Jupyter environment for
> `edhec_risk_kit_final.py`.** It is typically in the same folder as your
> course notebooks. Without it, the BL app will fail to import on startup.

**Rename and deploy the same way as Option A above**, using `requirements_bl_app.txt`
as your `requirements.txt`.

---

## Dependency notes

### TensorFlow and cold-start time

TensorFlow is ~600 MB. On Streamlit Community Cloud's free tier, this means:

- **First cold start**: 4–8 minutes (downloading + installing TF)
- **Warm starts** (app already running): near-instant
- **After inactivity** (~15 min idle on free tier): cold start again

**To reduce cold-start time**, replace `tensorflow` with `tensorflow-cpu` in
`requirements.txt`. The Python API is identical — only GPU support is dropped,
which Streamlit Cloud doesn't have anyway:

```
# In requirements.txt, change:
tensorflow>=2.16.0,<3.0.0
# to:
tensorflow-cpu>=2.16.0,<3.0.0
```

### Yahoo Finance rate limits

Both apps pull live data from Yahoo Finance via `yfinance`. On Streamlit Community
Cloud, the server IP is shared across many users, which can occasionally trigger
Yahoo Finance rate limits. The apps handle this with:
- `@st.cache_data(ttl=3600)` — data is re-fetched at most once per hour
- Retry logic with exponential backoff (in the BL app)

If you see `Empty DataFrame` errors, wait 60 seconds and refresh.

### Python version

Streamlit Community Cloud defaults to **Python 3.11**. This is the tested version.
The apps use Python 3.10+ syntax (`float | None` union types in type hints).
Do not deploy on Python 3.9 or earlier.

---

## How the apps connect — daily workflow

```
Morning routine (~5 minutes)
─────────────────────────────────────────────────────────────────────

1. Open Portfolio Optimiser (DCF-BL)
   → Check if any price targets need updating (sidebar)
   → Note BL Optimised weights from "Views, Returns & Weights" tab
   → Are you materially off-target from current holdings?

2. Open Technical Analysis (ML)
   → "Overview" tab: what's the daily signal for your largest positions?
   → "Regime Detection" tab: are we in Bull / Transitional / Bear?
   → "Rebalancing Planner" tab:
       a. Confirm current holdings (persistent from last session)
       b. Paste BL target weights
       c. Read the trade list + timing verdicts
       d. Execute "✅ Execute now" trades on Moo Moo
       e. Set a reminder for "🔴 Wait for better entry" trades
```

---

## File reference

| File | Purpose |
|------|---------|
| `technical_analysis_app.py` | Main app — 6 tabs of ML technical analysis + rebalancing planner |
| `requirements.txt` | Python dependencies for the technical analysis app |
| `requirements_bl_app.txt` | Python dependencies for the BL portfolio app |
| `.streamlit/config.toml` | Streamlit server config + dark theme |
| `.gitignore` | Excludes caches, venvs, secrets, model artefacts |
| `README.md` | This file |

---

## Disclaimer

This project is for educational and personal research purposes only. Nothing in
these apps constitutes financial advice. All ML signals are trained on historical
data and do not predict the future. Always pair technical signals with fundamental
analysis and your own risk management framework. Data sourced from Yahoo Finance
and FRED.
