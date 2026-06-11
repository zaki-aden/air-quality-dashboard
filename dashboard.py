"""
Air Quality Monitoring Dashboard
NO2 Prediction for Istanbul using Machine Learning and Satellite Data
Districts: Kartal, Kagithane, Uskudar
Data Source: Sentinel-5P via Google Earth Engine
Model: Random Forest -- One-Hot Encoding (best_rf_model_ohe.pkl)
Author: Graduation Project 2024-25
Usage: streamlit run dashboard.py
"""

import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Istanbul Air Quality Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark theme override ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Force sidebar to match dark theme */
[data-testid="stSidebar"] {
    background-color: #0a1220 !important;
    border-right: 1px solid #1a2e45 !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #0a1220 !important;
}

/* Radio buttons in sidebar */
[data-testid="stSidebar"] .stRadio label {
    color: #7aaad4 !important;
    font-size: 12px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #ffffff !important;
}

/* Main background */
.stApp {
    background-color: #0e0e11 !important;
}

/* Remove default streamlit padding weirdness */
[data-testid="stSidebar"] .block-container {
    padding-top: 0 !important;
}

/* Scrollbar dark */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a1220; }
::-webkit-scrollbar-thumb { background: #1a3a5c; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2a5a8c; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "dark_blue": "#0D1B2A",
    "mid_blue": "#1B3A5C",
    "sky_blue": "#2196F3",
    "light_blue": "#64B5F6",
    "green": "#4CAF50",
    "soft_gray": "#B0BEC5",
    "white": "#F5F9FF",
    "kartal": "#2196F3",
    "kagithane": "#4CAF50",
    "uskudar": "#FF9800",
}

st.markdown(f"""
<style>
.stApp {{ background-color: {COLORS['dark_blue']}; color: {COLORS['white']}; }}
[data-testid="stSidebar"] {{ background-color: {COLORS['mid_blue']}; }}
[data-testid="stSidebar"] * {{ color: {COLORS['white']} !important; }}
[data-testid="stMetric"] {{
background: linear-gradient(135deg, {COLORS['mid_blue']}, {COLORS['dark_blue']});
border: 1px solid {COLORS['sky_blue']}44; border-radius: 12px; padding: 16px !important;
}}
[data-testid="stMetricLabel"] {{ color: {COLORS['soft_gray']} !important; font-size: 13px !important; }}
[data-testid="stMetricValue"] {{ color: {COLORS['white']} !important; font-size: 28px !important; }}
h1, h2, h3 {{ color: {COLORS['white']} !important; }}
hr {{ border-color: {COLORS['mid_blue']}; }}
.info-card {{
background: linear-gradient(135deg, {COLORS['mid_blue']}cc, {COLORS['dark_blue']}cc);
border-left: 4px solid {COLORS['sky_blue']}; border-radius: 8px;
padding: 16px 20px; margin-bottom: 16px;
}}
.placeholder-card {{
background: {COLORS['mid_blue']}55; border: 2px dashed {COLORS['soft_gray']}66;
border-radius: 12px; padding: 32px; text-align: center; color: {COLORS['soft_gray']};
}}
.prediction-card {{
background: linear-gradient(135deg, {COLORS['sky_blue']}22, {COLORS['green']}22);
border: 2px solid {COLORS['sky_blue']}; border-radius: 16px; padding: 24px; text-align: center;
}}
.accuracy-card {{
background: {COLORS['mid_blue']}; border: 1px solid {COLORS['soft_gray']}44;
border-radius: 10px; padding: 18px; text-align: center;
}}
.who-banner {{
border-radius: 8px; padding: 14px 18px; margin: 10px 0;
}}
</style>
""", unsafe_allow_html=True)

DISTRICTS = ["Kartal", "Kağıthane", "Üsküdar"]
SEASONS = ["DJF / Winter", "MAM / Spring", "JJA / Summer", "SON / Autumn"]
SEASON_MAP = {"DJF / Winter": "DJF", "MAM / Spring": "MAM",
              "JJA / Summer": "JJA", "SON / Autumn": "SON"}
SEASON_NAME = {"DJF": "Winter", "MAM": "Spring", "JJA": "Summer", "SON": "Autumn"}

# Seasonal mean values from real dataset (mol/m2 * 1e5)
SEASONAL_MEANS = {
    "Kağıthane": {"DJF": 11.51e-5, "MAM": 9.61e-5, "JJA": 8.52e-5, "SON": 8.78e-5},
    "Üsküdar":   {"DJF": 11.87e-5, "MAM": 11.67e-5,"JJA": 10.06e-5,"SON": 10.91e-5},
    "Kartal":    {"DJF": 11.62e-5, "MAM": 11.72e-5, "JJA": 10.50e-5,"SON": 11.39e-5},
}

DISTRICT_COORDS = {
    "Kartal":    {"lat": 40.8889, "lon": 29.1872},
    "Kağıthane": {"lat": 41.0842, "lon": 28.9856},
    "Üsküdar":   {"lat": 41.0231, "lon": 29.0151},
}
DISTRICT_COLORS = {
    "Kartal":    COLORS["kartal"],
    "Kağıthane": COLORS["kagithane"],
    "Üsküdar":   COLORS["uskudar"],
}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(27,58,92,0.13)",
    font=dict(color=COLORS["white"], family="sans-serif"),
    xaxis=dict(gridcolor="#1B3A5C", linecolor="rgba(33,150,243,0.27)"),
    yaxis=dict(gridcolor="#1B3A5C", linecolor="rgba(33,150,243,0.27)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["white"])),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ── WHO THRESHOLD SYSTEM (FIXED: added ug/m3 equivalents + sensitive group clarification) ──
# WHO 2021 NO2 annual guideline: 10 ug/m3 = 0.0053 umol/m2 column density equivalent
# WHO 2021 NO2 24-hour guideline: 25 ug/m3 = ~0.013 umol/m2
# The dashboard thresholds below are based on tropospheric column density in umol/m2
# converted proportionally from the WHO surface concentration guidelines

WHO_LEVELS = [
    {"tier": "Good",                        "max": 54.0,          "color": "#4CAF50", "bg": "#1a3a1a",
     "note": "Below WHO annual guideline (10 µg/m³)",
     "ug_range": "0.0–10.8 µg/m³ equivalent",
     "advice": "Air quality is satisfactory. No health risk for any population group."},
    {"tier": "Moderate",                    "max": 90.0,          "color": "#FFC107", "bg": "#3a2e00",
     "note": "Approaching WHO 24-hour guideline (25 µg/m³)",
     "ug_range": "10.8–18.0 µg/m³ equivalent",
     "advice": "Acceptable. Unusually sensitive individuals may consider limiting prolonged outdoor exposure."},
    {"tier": "Unhealthy — Sensitive Groups","max": 120.0,         "color": "#FF9800", "bg": "#3a2000",
     "note": "Exceeds WHO 24-hour guideline — sensitive groups at risk",
     "ug_range": "18.0–24.0 µg/m³ equivalent",
     "advice": "Children, elderly and people with respiratory or cardiovascular conditions should reduce outdoor activity. General public not likely affected."},
    {"tier": "Unhealthy",                   "max": 160.0,         "color": "#F44336", "bg": "#3a0a0a",
     "note": "Significantly exceeds WHO guideline — general public at risk",
     "ug_range": "24.0–32.0 µg/m³ equivalent",
     "advice": "Everyone may begin to experience health effects. Sensitive groups should avoid outdoor activity."},
    {"tier": "Very Unhealthy (Sensitive Groups: Avoid All Outdoor Activity)",
                                            "max": float("inf"), "color": "#9C27B0", "bg": "#2a0a3a",
     "note": "Far exceeds WHO guideline — health emergency for sensitive groups",
     "ug_range": ">32.0 µg/m³ equivalent",
     "advice": "Sensitive groups (children, elderly, respiratory patients) should avoid ALL outdoor activity. General public should avoid prolonged exertion outdoors."},
]

WHO_ICONS = {
    "Good": "✅",
    "Moderate": "🟡",
    "Unhealthy — Sensitive Groups": "🟠",
    "Unhealthy": "🔴",
    "Very Unhealthy (Sensitive Groups: Avoid All Outdoor Activity)": "🟣",
}

def who_level(val_umol: float) -> dict:
    for lvl in WHO_LEVELS:
        if val_umol <= lvl["max"]:
            return lvl
    return WHO_LEVELS[-1]

def who_banner(district: str, val_umol: float):
    lvl = who_level(val_umol)
    icon = WHO_ICONS.get(lvl["tier"], "⚪")
    st.markdown(f"""
<div class='who-banner' style='background:{lvl["bg"]}; border-left:5px solid {lvl["color"]};'>
<div style='display:flex; align-items:center; gap:10px; margin-bottom:6px;'>
<span style='font-size:20px;'>{icon}</span>
<span style='font-weight:700; font-size:15px; color:{lvl["color"]};'>
        {district} — {lvl["tier"]}
</span>
<span style='font-size:11px; background:{lvl["color"]}33; color:{lvl["color"]};
padding:2px 10px; border-radius:12px; margin-left:auto;'>
        {val_umol:.5f} µmol/m²
</span>
</div>
<div style='font-size:12px; color:{COLORS["soft_gray"]}; margin-bottom:2px;'>
<b>Equivalent surface concentration:</b> {lvl["ug_range"]}
</div>
<div style='font-size:13px; color:{COLORS["soft_gray"]}; margin-bottom:3px;'>
<b>WHO note:</b> {lvl["note"]}
</div>
<div style='font-size:13px; color:{COLORS["soft_gray"]};'>
<b>Health advice:</b> {lvl["advice"]}
</div>
</div>
""", unsafe_allow_html=True)

def who_sidebar_legend():
    st.markdown(f"<div style='font-size:13px; font-weight:700; color:{COLORS['sky_blue']}; margin-bottom:8px;'>WHO NO₂ Thresholds (2021)</div>", unsafe_allow_html=True)
    rows = ""
    prev = 0.0
    for lvl in WHO_LEVELS:
        upper = lvl["max"]
        r = f"{prev:.3f}–{upper:.3f} µmol/m²" if upper != float("inf") else f">{prev:.3f} µmol/m²"
        ug = lvl.get("ug_range", "")
        rows += f"""<tr>
<td style='padding:4px 6px; color:{COLORS['soft_gray']}; font-size:11px;'>
<span style='display:inline-block;width:10px;height:10px;background:{lvl['color']};border-radius:50%;vertical-align:middle;margin-right:5px;'></span>
{lvl['tier']}
</td>
<td style='padding:4px 6px; font-size:10px; color:{COLORS['soft_gray']};'>{r}</td>
</tr>
<tr><td colspan='2' style='padding:0 6px 6px 26px; font-size:10px; color:{COLORS['soft_gray']}88;'>{ug}</td></tr>"""
        if upper != float("inf"):
            prev = upper
    st.markdown(f"""
<table style='width:100%;border-collapse:collapse;'>{rows}</table>
<div style='font-size:10px;color:{COLORS['soft_gray']};margin-top:6px;'>
WHO 2021 guidelines: Annual mean 10 µg/m³ · 24-h mean 25 µg/m³<br>
<i>Note: µmol/m² values are tropospheric column density equivalents from Sentinel-5P TROPOMI.
Surface µg/m³ conversions are approximate.</i>
</div>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        saved = joblib.load("best_rf_model_ohe.pkl")
        if hasattr(saved, 'predict'):
            return saved, None, True
        return saved["model"], saved.get("preprocessor"), True
    except FileNotFoundError:
        return None, None, False
    except Exception as e:
        st.warning(f"Model load error: {e}")
        return None, None, False

rf_model, preprocessor, model_loaded = load_model()

WAVELET_LOOKUP = {
    ("Kartal",    "DJF"): {"d1": 0.0, "d2": 0.0, "d3": -2.823e-07},
    ("Kartal",    "JJA"): {"d1": 0.0, "d2": 0.0, "d3": -1.415e-06},
    ("Kartal",    "MAM"): {"d1": 0.0, "d2": 0.0, "d3":  2.823e-07},
    ("Kartal",    "SON"): {"d1": 0.0, "d2": 0.0, "d3":  1.415e-06},
    ("Kağıthane", "DJF"): {"d1": 0.0, "d2": 0.0, "d3": 0.0},
    ("Kağıthane", "JJA"): {"d1": 0.0, "d2": 0.0, "d3": 0.0},
    ("Kağıthane", "MAM"): {"d1": 0.0, "d2": 0.0, "d3": 0.0},
    ("Kağıthane", "SON"): {"d1": 0.0, "d2": 0.0, "d3": 0.0},
    ("Üsküdar",   "DJF"): {"d1": 0.0, "d2": 0.0, "d3": 0.0},
    ("Üsküdar",   "JJA"): {"d1": 0.0, "d2": 0.0, "d3": 0.0},
    ("Üsküdar",   "MAM"): {"d1": 0.0, "d2": 0.0, "d3": 0.0},
    ("Üsküdar",   "SON"): {"d1": 0.0, "d2": 0.0, "d3": 0.0},
}

def run_prediction(district: str, season_code: str) -> float:
    if model_loaded:
        coords = DISTRICT_COORDS[district]
        wavelet = WAVELET_LOOKUP.get((district, season_code), {"d1": 0.0, "d2": 0.0, "d3": 0.0})
        region_ohe = [1.0 if district == "Kartal" else 0.0,
                      1.0 if district == "Kağıthane" else 0.0,
                      1.0 if district == "Üsküdar" else 0.0]
        season_ohe = [1.0 if season_code == "DJF" else 0.0,
                      1.0 if season_code == "JJA" else 0.0,
                      1.0 if season_code == "MAM" else 0.0,
                      1.0 if season_code == "SON" else 0.0]
        numeric_base = [float(coords["lat"]), float(coords["lon"])]
        numeric_wave = [float(wavelet["d1"]), float(wavelet["d2"]), float(wavelet["d3"])]
        if hasattr(rf_model, "named_steps"):
            rf_step = rf_model.named_steps["rf"]
        else:
            rf_step = rf_model
        n_feat = rf_step.n_features_in_
        if n_feat == 12:
            features = region_ohe + season_ohe + numeric_base + numeric_wave
        else:
            features = region_ohe + season_ohe + numeric_base
        X = np.array([features], dtype=np.float64)
        return float(rf_step.predict(X)[0])
    else:
        base = {"Kartal": 0.000110, "Kağıthane": 0.000125, "Üsküdar": 0.000105}
        season_factor = {"DJF": 1.20, "MAM": 1.05, "JJA": 0.85, "SON": 1.00}
        return base[district] * season_factor[season_code]

@st.cache_data
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv("NO2_Istanbul_Seasonal.csv")
        if "lat" not in df.columns or df["lat"].isnull().all():
            def extract_coords(geo_str):
                try:
                    geo = json.loads(geo_str)
                    return geo["coordinates"][1], geo["coordinates"][0]
                except Exception:
                    return None, None
            df[["lat", "lon"]] = df[".geo"].apply(lambda x: pd.Series(extract_coords(x)))
        if "date" not in df.columns:
            season_dates = {"DJF": "2024-01-15", "MAM": "2024-04-15",
                            "JJA": "2024-07-15", "SON": "2024-10-15"}
            df["date"] = pd.to_datetime(df["season"].map(season_dates))
        df["season_name"] = df["season"].map(SEASON_NAME)
        # Standardise Fall -> Autumn
        df["season_name"] = df["season_name"].replace({"Fall": "Autumn"})
        return df
    except FileNotFoundError:
        np.random.seed(42)
        rows = []
        base = {"Kartal": 0.000110, "Kağıthane": 0.000125, "Üsküdar": 0.000105}
        for season, date_str, factor in [
            ("DJF", "2024-01-15", 1.20), ("MAM", "2024-04-15", 1.05),
            ("JJA", "2024-07-15", 0.85), ("SON", "2024-10-15", 1.00),
        ]:
            for district, base_val in base.items():
                for _ in range(25):
                    noise = np.random.normal(0, base_val * 0.10)
                    rows.append({
                        "date": pd.to_datetime(date_str),
                        "region_name": district, "season": season,
                        "season_name": SEASON_NAME[season],
                        "NO2_column_number_density": max(0, base_val * factor + noise),
                        "lat": DISTRICT_COORDS[district]["lat"],
                        "lon": DISTRICT_COORDS[district]["lon"],
                    })
        return pd.DataFrame(rows)

df_all = load_data()

def no2_to_umol(val: float) -> float:
    return val * 1e6

def get_latest_no2(df: pd.DataFrame) -> dict:
    latest = df.groupby("region_name")["NO2_column_number_density"].mean()
    return latest.to_dict()

def get_current_season() -> str:
    m = datetime.now().month
    if m in [12, 1, 2]: return "DJF"
    elif m in [3, 4, 5]: return "MAM"
    elif m in [6, 7, 8]: return "JJA"
    else: return "SON"

def bar_chart_seasonal(df: pd.DataFrame, district: str) -> go.Figure:
    d = (df[df["region_name"] == district]
         .groupby("season")["NO2_column_number_density"]
         .mean().reset_index())
    season_order = ["DJF", "MAM", "JJA", "SON"]
    d["season"] = pd.Categorical(d["season"], categories=season_order, ordered=True)
    d = d.sort_values("season")
    fig = go.Figure(go.Bar(
        x=d["season"],
        y=d["NO2_column_number_density"] * 1e6,
        marker_color=[COLORS["sky_blue"], COLORS["green"], COLORS["uskudar"], COLORS["soft_gray"]],
        hovertemplate="Season: %{x}<br>Mean NO2: %{y:.4f} µmol/m²<extra></extra>",
    ))
    fig.update_layout(title=f"Seasonal Average NO2 — {district}",
                      yaxis_title="NO2 (µmol/m²)", **PLOTLY_LAYOUT)
    return fig

def line_chart_no2(df: pd.DataFrame, title: str = "NO2 by Season") -> go.Figure:
    fig = go.Figure()
    season_order = ["DJF", "MAM", "JJA", "SON"]
    for district in DISTRICTS:
        d = (df[df["region_name"] == district]
             .groupby("season")["NO2_column_number_density"]
             .mean().reindex(season_order).reset_index())
        fig.add_trace(go.Scatter(
            x=d["season"], y=d["NO2_column_number_density"] * 1e6,
            name=district, mode="lines+markers",
            line=dict(color=DISTRICT_COLORS[district], width=2),
            hovertemplate="Season: %{x}<br>NO2: %{y:.4f} µmol/m²<extra>" + district + "</extra>",
        ))
    fig.update_layout(title=title, yaxis_title="NO2 (µmol/m²)",
                      xaxis_title="Season", **PLOTLY_LAYOUT)
    return fig

# ── SIDEBAR ──
with st.sidebar:
    st.markdown(f"""
<style>
@keyframes spin-earth {{
    0%   {{ transform: rotate(0deg); }}
    100% {{ transform: rotate(360deg); }}
}}
.earth-spin {{
    display: inline-block;
    animation: spin-earth 8s linear infinite;
    font-size: 36px;
    line-height: 1;
}}
</style>
<div style='text-align:center; padding:16px 0 12px 0; border-bottom:1px solid #1a2e45; margin-bottom:12px;'>
<div class='earth-spin'>🌍</div>
<div style='font-size:12px; font-weight:700; color:{COLORS["sky_blue"]};
letter-spacing:2px; line-height:1.4; margin-top:6px;'>AIR QUALITY<br>MONITORING</div>
<div style='font-size:10px; color:#2a5a7a; margin-top:3px;'>Istanbul · Sentinel-5P</div>
</div>
""", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=["🏠 Home", "📍 District Analysis", "🤖 ML Predictions",
                 "📊 Analytics & Seasonal", "🌡️ Seasonal Interpretation", "🏭 PM10 & PM2.5 (Future)"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1a2e45; margin:12px 0;'>", unsafe_allow_html=True)

    if model_loaded:
        st.markdown("""
<div style='background:#0a2a1a; border:1px solid #1a5a2a; border-radius:8px;
padding:8px 10px; display:flex; align-items:center; gap:6px; margin-bottom:12px;'>
<span style='font-size:13px;'>✅</span>
<span style='font-size:10px; color:#4aaa6a; font-weight:600;'>Model loaded (OHE)</span>
</div>""", unsafe_allow_html=True)
    else:
        st.info("ℹ️ Demo mode")

    st.markdown("""
<div style='font-size:12px; font-weight:700; color:#4a9fd4; letter-spacing:1px;
margin-bottom:12px; text-transform:uppercase;'>WHO NO₂ Levels (µmol/m²)</div>
<div style='display:flex; flex-direction:column; gap:0px;'>

<div style='display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid #1a2e3a;'>
<div style='width:10px;height:10px;border-radius:50%;background:#4CAF50;flex-shrink:0;'></div>
<div style='font-size:12px;color:#4CAF50;flex:1;font-weight:500;'>Good</div>
<div style='font-size:11px;color:#4a7aaa;'>&lt;54</div></div>
<div style='font-size:11px;color:#2a5a2a;padding:2px 0 6px 18px;'>0–10.8 µg/m³</div>

<div style='display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid #1a2e3a;'>
<div style='width:10px;height:10px;border-radius:50%;background:#FFC107;flex-shrink:0;'></div>
<div style='font-size:12px;color:#FFC107;flex:1;font-weight:500;'>Moderate</div>
<div style='font-size:11px;color:#4a7aaa;'>54–90</div></div>
<div style='font-size:11px;color:#4a4a00;padding:2px 0 6px 18px;'>10.8–18 µg/m³</div>

<div style='display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid #1a2e3a;'>
<div style='width:10px;height:10px;border-radius:50%;background:#FF9800;flex-shrink:0;'></div>
<div style='font-size:12px;color:#FF9800;flex:1;font-weight:500;'>Sensitive Groups</div>
<div style='font-size:11px;color:#4a7aaa;'>90–120</div></div>
<div style='font-size:11px;color:#4a3000;padding:2px 0 6px 18px;'>18–24 µg/m³</div>

<div style='display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid #1a2e3a;'>
<div style='width:10px;height:10px;border-radius:50%;background:#F44336;flex-shrink:0;'></div>
<div style='font-size:12px;color:#F44336;flex:1;font-weight:500;'>Unhealthy</div>
<div style='font-size:11px;color:#4a7aaa;'>120–160</div></div>
<div style='font-size:11px;color:#4a1010;padding:2px 0 6px 18px;'>24–32 µg/m³</div>

<div style='display:flex;align-items:center;gap:8px;padding:7px 0;'>
<div style='width:10px;height:10px;border-radius:50%;background:#9C27B0;flex-shrink:0;'></div>
<div style='font-size:12px;color:#9C27B0;flex:1;font-weight:500;'>Very Unhealthy</div>
<div style='font-size:11px;color:#4a7aaa;'>&gt;160</div></div>
<div style='font-size:11px;color:#3a1a4a;padding:2px 0 4px 18px;'>&gt;32 µg/m³ · Sensitive avoid outdoor</div>

</div>
<hr style='border-color:#1a2e3a; margin:14px 0 10px 0;'>
<div style='font-size:11px;color:#2a5a7a;margin-bottom:4px;'>🛰️ Sentinel-5P / GEE</div>
<div style='font-size:11px;color:#2a5a7a;margin-bottom:4px;'>🤖 Random Forest (.pkl)</div>
<div style='font-size:11px;color:#2a5a7a;'>🎓 Graduation Project 2024-25</div>
""", unsafe_allow_html=True)

# ── PAGE 1: HOME ──
if page == "🏠 Home":
    st.markdown(f"""
<h1 style='font-size:32px; font-weight:800; margin-bottom:4px;'>
🌍 Air Quality Monitoring Dashboard
</h1>
<div style='color:{COLORS["soft_gray"]}; font-size:15px; margin-bottom:24px;'>
NO2 Prediction for Istanbul Districts using Machine Learning & Satellite Data (Sentinel-5P)
</div>
""", unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        selected_district_home = st.selectbox("Select District", ["All"] + DISTRICTS, key="home_dist")
    with col_f2:
        selected_season_home = st.selectbox("Select Season", ["All"] + list(SEASON_NAME.values()), key="home_season")

    df_home = df_all.copy()
    if selected_district_home != "All":
        df_home = df_home[df_home["region_name"] == selected_district_home]
    if selected_season_home != "All":
        rev = {v: k for k, v in SEASON_NAME.items()}
        df_home = df_home[df_home["season"] == rev[selected_season_home]]

    st.markdown("### 📡 Current NO₂ Levels")
    latest = get_latest_no2(df_all)
    current_season = get_current_season()

    c1, c2, c3 = st.columns(3)
    for col, district in zip([c1, c2, c3], DISTRICTS):
        val = latest.get(district, 0)
        val_umol = no2_to_umol(val)
        lvl = who_level(val_umol)
        seasonal_mean = SEASONAL_MEANS.get(district, {}).get(current_season, val)
        seasonal_mean_umol = seasonal_mean * 1e6
        col.markdown(f"""
<div style='background:linear-gradient(135deg,{COLORS["mid_blue"]},{COLORS["dark_blue"]});
border:1px solid {COLORS["mid_blue"]}; border-top:2px solid {lvl["color"]};
border-radius:14px; overflow:hidden;'>
<div style='padding:16px 16px 12px 16px;'>
<div style='font-size:11px; color:{COLORS["soft_gray"]}88; margin-bottom:6px; letter-spacing:0.5px;'>{district}</div>
<div style='font-size:28px; font-weight:800; color:{COLORS["sky_blue"]}; line-height:1;'>
        {val_umol:.5f}</div>
<div style='font-size:11px; color:{COLORS["soft_gray"]}55; margin-top:2px;'>µmol/m²</div>
<div style='font-size:10px; color:{COLORS["soft_gray"]}44; margin-top:4px;'>
Seasonal mean ({SEASON_NAME[current_season]}): {seasonal_mean_umol:.5f} µmol/m²
</div>
</div>
<div style='background:{lvl["color"]}18; border-top:1px solid {lvl["color"]}33;
padding:6px 16px; display:flex; align-items:center; justify-content:space-between;'>
<div style='display:flex; align-items:center; gap:5px; overflow:hidden;'>
<span style='width:5px; height:5px; border-radius:50%; background:{lvl["color"]}; display:inline-block; flex-shrink:0;'></span>
<span style='font-size:9px; color:{lvl["color"]}; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{lvl["tier"]}</span>
</div>
<span style='font-size:8px; color:{COLORS["soft_gray"]}44; white-space:nowrap;'>{lvl["ug_range"]}</span>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 NO₂ Trend by Season")
    fig_line = line_chart_no2(df_home, title="NO2 by Season — All Districts")
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown(f"""
<div class='info-card'>
<b>About this project:</b> This dashboard visualizes NO2 pollution levels for three Istanbul
districts (Kartal, Kağıthane, Üsküdar) using satellite imagery from Sentinel-5P. A trained
Random Forest model (R²=0.9762) provides NO2 predictions. Data is retrieved via Google Earth Engine.
</div>
""", unsafe_allow_html=True)

# ── PAGE 2: DISTRICT ANALYSIS ──
elif page == "📍 District Analysis":
    st.markdown("<h1>📍 District Analysis</h1>", unsafe_allow_html=True)

    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        sel_district = st.selectbox("Select District", DISTRICTS, key="dist_select")
    with col_d2:
        st.markdown("<div style='margin-top:28px;'>", unsafe_allow_html=True)
        compare_mode = st.toggle("Compare All Districts", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    df_dist = df_all[df_all["region_name"] == sel_district]

    st.markdown("### 🗺️ District Location Map")
    map_districts = DISTRICTS if compare_mode else [sel_district]
    latest = get_latest_no2(df_all)
    map_df = pd.DataFrame([
        {**DISTRICT_COORDS[d], "district": d,
         "NO2": round(no2_to_umol(latest.get(d, 0)), 5)}
        for d in map_districts
    ])
    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", hover_name="district",
        hover_data={"NO2": True, "lat": False, "lon": False},
        color="district", color_discrete_map={d: DISTRICT_COLORS[d] for d in DISTRICTS},
        size=[20] * len(map_df), zoom=10, height=320,
    )
    fig_map.update_layout(
        mapbox_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    val_umol = no2_to_umol(latest.get(sel_district, 0))
    who_banner(sel_district, val_umol)

    st.markdown("### 📊 Detailed Charts")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        df_plot = df_all if compare_mode else df_dist
        fig_l = line_chart_no2(df_plot, title=f"NO2 by Season — {sel_district if not compare_mode else 'All Districts'}")
        st.plotly_chart(fig_l, use_container_width=True)
    with col_c2:
        fig_b = bar_chart_seasonal(df_all, sel_district)
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("### 🗓️ Seasonal Statistics Table")
    season_stats = (
        df_dist.groupby("season")["NO2_column_number_density"]
        .agg(["mean", "min", "max", "std"])
        .rename(columns={"mean": "Mean", "min": "Min", "max": "Max", "std": "Std Dev"})
        .reset_index().rename(columns={"season": "Season"})
    )
    season_stats[["Mean", "Min", "Max", "Std Dev"]] = (
        season_stats[["Mean", "Min", "Max", "Std Dev"]] * 1e6
    ).round(5)
    st.dataframe(season_stats.style.background_gradient(subset=["Mean"], cmap="Blues"),
                 use_container_width=True)

# ── PAGE 3: ML PREDICTIONS ──
elif page == "🤖 ML Predictions":
    st.markdown("<h1>🤖 Machine Learning Predictions</h1>", unsafe_allow_html=True)

    if model_loaded:
        st.success("✅ Random Forest model loaded (OHE + Wavelet features d1/d2/d3 | R²=0.9762, MAE=1.27×10⁻⁶, RMSE=1.85×10⁻⁶ mol/m²)")
    else:
        st.info("ℹ️ Running in demo mode. Place best_rf_model_ohe.pkl in the app directory to enable live predictions.")

    col_p1, col_p2, col_p3 = st.columns([1.5, 1.5, 1])
    with col_p1:
        pred_district = st.selectbox("Select District", DISTRICTS, key="pred_dist")
    with col_p2:
        future_date = st.date_input(
            "Select Future Date",
            value=datetime.today() + timedelta(days=30),
            min_value=datetime.today().date(), key="pred_date",
        )
    with col_p3:
        st.markdown("<div style='margin-top:28px;'>", unsafe_allow_html=True)
        col_b1, col_b2 = st.columns(2)
        predict_clicked = col_b1.button("🔮 Predict", use_container_width=True)
        reset_clicked = col_b2.button("🔄 Reset", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if reset_clicked:
        st.rerun()

    if predict_clicked:
        month = future_date.month
        if month in [12, 1, 2]: season_code = "DJF"
        elif month in [3, 4, 5]: season_code = "MAM"
        elif month in [6, 7, 8]: season_code = "JJA"
        else: season_code = "SON"

        prediction = run_prediction(pred_district, season_code)
        pred_umol = no2_to_umol(prediction)
        lvl = who_level(pred_umol)

        st.markdown("### 🎯 Prediction Result")
        st.markdown(f"""
<div class='prediction-card'>
<div style='font-size:13px; color:{COLORS["soft_gray"]}; margin-bottom:8px;'>
        {pred_district} • {future_date} • Season: {season_code} ({SEASON_NAME[season_code]})
</div>
<div style='font-size:48px; font-weight:900; color:{COLORS["sky_blue"]};'>
        {pred_umol:.5f}
</div>
<div style='font-size:16px; color:{COLORS["soft_gray"]};'>µmol/m² NO₂</div>
<div style='margin-top:12px; font-size:18px; font-weight:700; color:{lvl["color"]};'>
Air Quality: {lvl["tier"]}
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        who_banner(pred_district, pred_umol)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 📐 Model Accuracy Metrics")
        m1, m2, m3, m4 = st.columns(4)
        for col, label, value, unit in [
            (m1, "R²", 0.9762, "Overall"),
            (m2, "MAE", 1.27, "×10⁻⁶ mol/m²"),
            (m3, "RMSE", 1.85, "×10⁻⁶ mol/m²"),
            (m4, "Features", 7, "incl. d1/d2/d3"),
        ]:
            col.markdown(f"""
<div class='accuracy-card'>
<div style='font-size:13px; color:{COLORS["soft_gray"]};'>{label}</div>
<div style='font-size:26px; font-weight:800; color:{COLORS["sky_blue"]};'>{value}</div>
<div style='font-size:12px; color:{COLORS["soft_gray"]};'>{unit}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 R² Score by Season (Test Set)")
        seasonal_r2 = {"DJF (Winter)": 0.6676, "MAM (Spring)": 0.9763,
                       "JJA (Summer)": 0.9605, "SON (Autumn)": 0.9885}
        seasonal_colors = ["#534AB7", "#1D9E75", "#EF9F27", "#D85A30"]
        fig_r2 = go.Figure(go.Bar(
            x=list(seasonal_r2.keys()), y=list(seasonal_r2.values()),
            marker_color=seasonal_colors,
            text=[f"{v:.4f}" for v in seasonal_r2.values()], textposition="outside",
            hovertemplate="Season: %{x}<br>R²: %{y:.4f}<extra></extra>",
        ))
        fig_r2.add_hline(y=0.95, line_dash="dash", line_color="#B0BEC5",
                         annotation_text="R²=0.95 target", annotation_position="top right")
        fig_r2.update_layout(
            title="Per-Season R² — Wavelet-Enhanced Model",
            yaxis_title="R²", yaxis_range=[0, 1.05], **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_r2, use_container_width=True)

        results_df = df_all[df_all["region_name"] == pred_district][["region_name", "season"]].copy()
        results_df["actual_NO2_mol_m2"] = df_all[df_all["region_name"] == pred_district]["NO2_column_number_density"].values
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="⬇️ Download Results as CSV", data=csv_bytes,
                           file_name=f"NO2_predictions_{pred_district}_{future_date}.csv",
                           mime="text/csv")
    else:
        st.markdown(f"""
<div class='info-card' style='margin-top:20px;'>
Select a district and date above, then click <b>Predict</b> to run the model.
</div>
""", unsafe_allow_html=True)

# ── PAGE 4: ANALYTICS ──
elif page == "📊 Analytics & Seasonal":
    st.markdown("<h1>📊 Analytics & Seasonal Trends</h1>", unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([2, 1])
    with col_a1:
        sel_season_label = st.selectbox("Select Season", SEASONS, key="analytics_season")
    sel_season_code = SEASON_MAP[sel_season_label]
    df_season = df_all[df_all["season"] == sel_season_code]

    st.markdown("### 🌦️ Seasonal Comparison — All Districts")
    season_comp = (df_all.groupby(["season", "region_name"])["NO2_column_number_density"]
                   .mean().reset_index())
    season_comp["NO2_umol"] = season_comp["NO2_column_number_density"] * 1e6
    fig_bar = px.bar(
        season_comp, x="season", y="NO2_umol", color="region_name",
        barmode="group", color_discrete_map=DISTRICT_COLORS,
        labels={"NO2_umol": "Mean NO2 (µmol/m²)", "season": "Season", "region_name": "District"},
        category_orders={"season": ["DJF", "MAM", "JJA", "SON"]},
    )
    fig_bar.update_layout(title="Mean NO2 by Season & District", **PLOTLY_LAYOUT)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(f"### 🍩 Pollution Distribution — {sel_season_label}")
    pie_data = df_season.groupby("region_name")["NO2_column_number_density"].mean().reset_index()
    fig_pie = px.pie(pie_data, values="NO2_column_number_density", names="region_name",
                     color="region_name", color_discrete_map=DISTRICT_COLORS, hole=0.45)
    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["white"]),
                          legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=30, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(f"### 📋 Statistical Summary — {sel_season_label}")
    stats_table = (df_season.groupby("region_name")["NO2_column_number_density"]
                   .agg(Mean="mean", Min="min", Max="max", Std=lambda x: x.std())
                   .reset_index().rename(columns={"region_name": "District"}))
    stats_table[["Mean", "Min", "Max", "Std"]] = (stats_table[["Mean", "Min", "Max", "Std"]] * 1e6).round(5)
    st.dataframe(stats_table.style.background_gradient(subset=["Mean"], cmap="Blues"),
                 use_container_width=True, hide_index=True)

# ── PAGE 5: SEASONAL INTERPRETATION (NEW) ──
elif page == "🌡️ Seasonal Interpretation":
    st.markdown("<h1>🌡️ Seasonal Interpretation & Critical Days</h1>", unsafe_allow_html=True)
    st.markdown(f"""
<div class='info-card'>
<b>What this page shows:</b> Based on the WHO 2021 NO2 guidelines, we estimate how many days
per season the NO2 concentrations in each district are likely to exceed the thresholds that are
considered dangerous for <b>sensitive population groups</b> — including children under 14,
elderly people over 65, pregnant women, and individuals with respiratory or cardiovascular conditions.
</div>
""", unsafe_allow_html=True)

    st.markdown("### 👥 District Population Overview")

    pop_data = {
        "Kağıthane": {"total": 450000, "sensitive": 126000, "general": 324000},
        "Üsküdar":   {"total": 530000, "sensitive": 150000, "general": 380000},
        "Kartal":    {"total": 470000, "sensitive": 132000, "general": 338000},
    }

    pc1, pc2, pc3 = st.columns(3)
    for col, district in zip([pc1, pc2, pc3], DISTRICTS):
        p = pop_data[district]
        col.markdown(f"""
<div style='background:linear-gradient(135deg,{COLORS["mid_blue"]},{COLORS["dark_blue"]});
border:2px solid {COLORS["sky_blue"]}55; border-radius:14px; padding:14px; text-align:center;'>
<div style='font-size:13px; color:{COLORS["soft_gray"]}; margin-bottom:4px;'>{district}</div>
<div style='font-size:20px; font-weight:800; color:{COLORS["sky_blue"]};'>
{{p["total"]:,}}</div>
<div style='font-size:11px; color:{COLORS["soft_gray"]}; margin-bottom:6px;'>Total residents</div>
<div style='font-size:13px; color:#F44336; font-weight:700;'>{{p["sensitive"]:,}} sensitive</div>
<div style='font-size:11px; color:{COLORS["soft_gray"]}; margin-bottom:4px;'>children, elderly, respiratory patients</div>
<div style='font-size:13px; color:#4CAF50; font-weight:700;'>{{p["general"]:,}} general</div>
<div style='font-size:11px; color:{COLORS["soft_gray"]};'>rest of population</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📅 Estimated Critical Days per Season")

    critical_days = {
        "Kağıthane": {"DJF": 72, "MAM": 58, "JJA": 41, "SON": 55},
        "Üsküdar":   {"DJF": 68, "MAM": 61, "JJA": 44, "SON": 59},
        "Kartal":    {"DJF": 65, "MAM": 59, "JJA": 43, "SON": 57},
    }
    critical_days_general = {
        "Kağıthane": {"DJF": 48, "MAM": 35, "JJA": 22, "SON": 31},
        "Üsküdar":   {"DJF": 44, "MAM": 38, "JJA": 25, "SON": 34},
        "Kartal":    {"DJF": 42, "MAM": 36, "JJA": 24, "SON": 33},
    }
    season_days = {"DJF": 90, "MAM": 92, "JJA": 92, "SON": 91}
    season_labels = {"DJF": "Winter (DJF)", "MAM": "Spring (MAM)",
                     "JJA": "Summer (JJA)", "SON": "Autumn (SON)"}

    sel_season_interp = st.selectbox("Select Season", list(season_labels.values()), key="interp_season")
    sel_code_interp = {v: k for k, v in season_labels.items()}[sel_season_interp]
    total_days = season_days[sel_code_interp]

    st.markdown(f"#### {sel_season_interp} — {total_days} total days in season")

    cols = st.columns(3)
    for col, district in zip(cols, DISTRICTS):
        crit_s = critical_days[district][sel_code_interp]
        crit_g = critical_days_general[district][sel_code_interp]
        pct_s = (crit_s / total_days) * 100
        pct_g = (crit_g / total_days) * 100
        pop_s = pop_data[district]["sensitive"]
        pop_g = pop_data[district]["general"]
        col.markdown(f"""
<div style='background:linear-gradient(135deg,{COLORS["mid_blue"]},{COLORS["dark_blue"]});
border:2px solid #F4433688; border-radius:14px; padding:18px; text-align:center;'>
<div style='font-size:13px; color:{COLORS["soft_gray"]}; margin-bottom:8px; font-weight:700;'>{district}</div>
<div style='background:#F4433622; border-radius:8px; padding:8px; margin-bottom:8px;'>
<div style='font-size:11px; color:#F44336; font-weight:700;'>🔴 Sensitive Groups ({pop_s:,} people)</div>
<div style='font-size:28px; font-weight:800; color:#F44336;'>{crit_s} days</div>
<div style='font-size:11px; color:{COLORS["soft_gray"]};'>({pct_s:.0f}% of season at risk)</div>
</div>
<div style='background:#FF980022; border-radius:8px; padding:8px;'>
<div style='font-size:11px; color:#FF9800; font-weight:700;'>🟠 General Population ({pop_g:,} people)</div>
<div style='font-size:28px; font-weight:800; color:#FF9800;'>{crit_g} days</div>
<div style='font-size:11px; color:{COLORS["soft_gray"]};'>({pct_g:.0f}% of season at risk)</div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Critical Days Comparison — All Seasons")

    rows_data = []
    for district in DISTRICTS:
        for season_code, label in season_labels.items():
            rows_data.append({
                "District": district, "Season": label, "Group": "Sensitive",
                "Critical Days": critical_days[district][season_code],
                "People Affected": pop_data[district]["sensitive"],
                "% of Season": round((critical_days[district][season_code] / season_days[season_code]) * 100, 1),
            })
            rows_data.append({
                "District": district, "Season": label, "Group": "General Population",
                "Critical Days": critical_days_general[district][season_code],
                "People Affected": pop_data[district]["general"],
                "% of Season": round((critical_days_general[district][season_code] / season_days[season_code]) * 100, 1),
            })
    df_crit = pd.DataFrame(rows_data)

    tab1, tab2 = st.tabs(["🔴 Sensitive Groups", "🟠 General Population"])
    with tab1:
        df_s = df_crit[df_crit["Group"] == "Sensitive"]
        fig_s = px.bar(df_s, x="Season", y="Critical Days", color="District",
                       barmode="group", color_discrete_map=DISTRICT_COLORS, text="Critical Days",
                       labels={"Critical Days": "Critical Days (Sensitive Groups)"})
        fig_s.update_layout(title=f"Critical Days — Sensitive Groups (children, elderly, respiratory patients)",
                            **PLOTLY_LAYOUT)
        st.plotly_chart(fig_s, use_container_width=True)
        st.dataframe(df_s[["District","Season","Critical Days","People Affected","% of Season"]]
                     .style.background_gradient(subset=["Critical Days","% of Season"], cmap="Reds"),
                     use_container_width=True, hide_index=True)
    with tab2:
        df_g = df_crit[df_crit["Group"] == "General Population"]
        fig_g = px.bar(df_g, x="Season", y="Critical Days", color="District",
                       barmode="group", color_discrete_map=DISTRICT_COLORS, text="Critical Days",
                       labels={"Critical Days": "Critical Days (General Population)"})
        fig_g.update_layout(title="Critical Days — General Population",
                            **PLOTLY_LAYOUT)
        st.plotly_chart(fig_g, use_container_width=True)
        st.dataframe(df_g[["District","Season","Critical Days","People Affected","% of Season"]]
                     .style.background_gradient(subset=["Critical Days","% of Season"], cmap="Oranges"),
                     use_container_width=True, hide_index=True)

    st.markdown("### 📊 Combined Population Impact Summary")
    # Total people affected on worst day (winter)
    total_sensitive = sum(pop_data[d]["sensitive"] for d in DISTRICTS)
    total_general = sum(pop_data[d]["general"] for d in DISTRICTS)
    col_t1, col_t2 = st.columns(2)
    col_t1.markdown(f"""
<div style='background:#F4433622; border:2px solid #F44336; border-radius:12px; padding:16px; text-align:center;'>
<div style='font-size:12px; color:{COLORS["soft_gray"]}; margin-bottom:4px;'>Sensitive people across all 3 districts</div>
<div style='font-size:32px; font-weight:800; color:#F44336;'>{total_sensitive:,}</div>
<div style='font-size:11px; color:{COLORS["soft_gray"]};'>at risk on a typical winter day</div>
</div>
""", unsafe_allow_html=True)
    col_t2.markdown(f"""
<div style='background:#FF980022; border:2px solid #FF9800; border-radius:12px; padding:16px; text-align:center;'>
<div style='font-size:12px; color:{COLORS["soft_gray"]}; margin-bottom:4px;'>General population across all 3 districts</div>
<div style='font-size:32px; font-weight:800; color:#FF9800;'>{total_general:,}</div>
<div style='font-size:11px; color:{COLORS["soft_gray"]};'>exposed to unhealthy air on a typical winter day</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class='info-card' style='margin-top:16px;'>
<b>⚠️ Important note:</b> These are estimated values based on seasonal mean NO2 concentrations
from the Sentinel-5P dataset and WHO 2021 guidelines. Actual daily critical days may vary depending
on specific meteorological conditions, wind patterns, and emission events. The values represent
approximate exceedance frequency calculated from seasonal NO2 distribution patterns.
<br><br>
<b>Sensitive groups include:</b> Children under 14, elderly individuals over 65, pregnant women,
and persons with pre-existing respiratory conditions (asthma, COPD) or cardiovascular disease.
</div>
""", unsafe_allow_html=True)

# ── PAGE 6: PM FUTURE ──
elif page == "🏭 PM10 & PM2.5 (Future)":
    st.markdown("<h1>🏭 PM10 & PM2.5 — Future Integration Plan</h1>", unsafe_allow_html=True)
    st.markdown(f"""
<div class='info-card'>
<b>⚠️ Why PM10 and PM2.5 are not in the current dataset</b><br><br>
Sentinel-5P TROPOMI measures column-integrated atmospheric gases (NO2, SO2, CO, O3, CH4)
at ~3.5 km resolution. Particulate matter (PM10 and PM2.5) <b>cannot be directly retrieved</b>
from TROPOMI because it does not measure aerosol optical properties at the wavelengths needed
for surface PM estimation. Alternative satellite-based approaches are required.
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🛰️ Proposed Data Sources")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"""
<div class='info-card'>
<b>🛰️ MODIS Terra/Aqua — Aerosol Optical Depth (AOD)</b><br><br>
<b>GEE dataset:</b> MODIS/061/MOD08_D3<br>
<b>Resolution:</b> 1° × 1° (~111 km) daily<br>
<b>Key variable:</b> Aerosol_Optical_Depth_Land_Mean<br><br>
AOD measures how much sunlight aerosol particles prevent from reaching the ground.
It can be converted to surface PM2.5 using empirical regression models.
</div>
""", unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
<div class='info-card'>
<b>🌍 MERRA-2 — Meteorological Reanalysis</b><br><br>
<b>GEE dataset:</b> NASA/GSFC/MERRA/flx/2<br>
<b>Resolution:</b> 0.5° × 0.625° (~50 km) hourly<br>
<b>Key variables:</b> DUSMASS25, OCSMASS, BCSMASS, SSSMASS25<br><br>
MERRA-2 provides modelled PM component mass concentrations at surface level.
</div>
""", unsafe_allow_html=True)

    st.markdown("### 📏 WHO PM Thresholds (2021 Guidelines)")
    st.markdown(f"""
<div class='info-card' style='font-size:12px;'>
<b>Note on these values:</b> The thresholds below are WHO 2021 annual and 24-hour mean guidelines
for particulate matter. These apply to the <b>general population</b>. For sensitive groups
(children, elderly, respiratory patients), even lower concentrations may cause health effects.
The Interim Target 1 values represent transitional targets for countries that cannot immediately
meet the WHO guideline.
</div>
""", unsafe_allow_html=True)
    pm_thresh = pd.DataFrame({
        "Pollutant": ["PM2.5", "PM2.5", "PM10", "PM10"],
        "Averaging Period": ["Annual mean", "24-hour mean", "Annual mean", "24-hour mean"],
        "WHO Guideline (General Population)": ["5 µg/m³", "15 µg/m³", "15 µg/m³", "45 µg/m³"],
        "Interim Target 1": ["35 µg/m³", "75 µg/m³", "70 µg/m³", "150 µg/m³"],
        "Applies to Sensitive Groups?": ["Yes, stricter limits recommended", "Yes, stricter limits recommended",
                                          "Yes, stricter limits recommended", "Yes, stricter limits recommended"],
    })
    st.dataframe(pm_thresh, use_container_width=True, hide_index=True)

    st.markdown("### 📍 Current Status")
    c1, c2, c3 = st.columns(3)
    for col, label in zip([c1, c2, c3], DISTRICTS):
        col.markdown(f"""
<div class='placeholder-card'>
<div style='font-size:28px;'>🏭</div>
<div style='font-weight:700; margin:8px 0 4px;'>{label}</div>
<div style='font-size:12px;'>PM10 — data extraction pending</div>
<div style='font-size:12px;'>PM2.5 — data extraction pending</div>
<div style='font-size:11px; margin-top:8px; color:{COLORS["soft_gray"]};'>
Source: MERRA-2 / MODIS AOD</div>
</div>
""", unsafe_allow_html=True)
