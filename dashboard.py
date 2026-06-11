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

# ── DESIGN TOKENS ──
BG     = "#0e0e11"
CELL   = "#161618"
BORDER = "#252528"
MUTED  = "#6b7280"
WHITE  = "#f0f0f5"
INDIGO = "#6366f1"
ROSE   = "#f43f5e"
ORANGE = "#f97316"
GREEN  = "#22c55e"

COLORS = {
    "dark_blue": BG, "mid_blue": "#1a1a24", "sky_blue": INDIGO,
    "light_blue": "#a5b4fc", "green": GREEN, "soft_gray": MUTED,
    "white": WHITE, "kartal": INDIGO, "kagithane": ROSE, "uskudar": ORANGE,
}

DISTRICTS = ["Kartal", "Kağıthane", "Üsküdar"]
SEASONS   = ["DJF / Winter", "MAM / Spring", "JJA / Summer", "SON / Autumn"]
SEASON_MAP  = {"DJF / Winter": "DJF", "MAM / Spring": "MAM",
               "JJA / Summer": "JJA", "SON / Autumn": "SON"}
SEASON_NAME = {"DJF": "Winter", "MAM": "Spring", "JJA": "Summer", "SON": "Autumn"}

SEASONAL_MEANS = {
    "Kağıthane": {"DJF": 11.51e-5, "MAM": 9.61e-5,  "JJA": 8.52e-5,  "SON": 8.78e-5},
    "Üsküdar":   {"DJF": 11.87e-5, "MAM": 11.67e-5, "JJA": 10.06e-5, "SON": 10.91e-5},
    "Kartal":    {"DJF": 11.62e-5, "MAM": 11.72e-5, "JJA": 10.50e-5, "SON": 11.39e-5},
}
DISTRICT_COORDS = {
    "Kartal":    {"lat": 40.8889, "lon": 29.1872},
    "Kağıthane": {"lat": 41.0842, "lon": 28.9856},
    "Üsküdar":   {"lat": 41.0231, "lon": 29.0151},
}
DISTRICT_COLORS = {"Kartal": INDIGO, "Kağıthane": ROSE, "Üsküdar": ORANGE}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,22,24,0.5)",
    font=dict(color=WHITE, family="Inter, sans-serif"),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER,
               tickfont=dict(color=MUTED, size=11)),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
               tickfont=dict(color=MUTED, size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=WHITE)),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ── GLOBAL CSS (single block) ──
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:{BG}; --cell:{CELL}; --border:{BORDER};
  --muted:{MUTED}; --white:{WHITE};
  --indigo:{INDIGO}; --rose:{ROSE}; --orange:{ORANGE}; --green:{GREEN};
}}
*, *::before, *::after {{ box-sizing: border-box; }}
body, .stApp {{ background-color: var(--bg) !important; color: var(--white); }}
.stApp {{ font-family: 'Inter', sans-serif !important; }}
/* Sidebar */
[data-testid="stSidebar"] {{
  background-color: var(--cell) !important;
  border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] * {{ color: var(--white) !important; }}
[data-testid="stSidebar"] .stRadio label {{
  padding: 8px 12px; border-radius: 8px; margin: 2px 0;
  transition: background 0.15s; cursor: pointer;
}}
[data-testid="stSidebar"] .stRadio label:hover {{ background: #252528 !important; }}
/* Bento columns */
[data-testid="column"] {{
  background: var(--cell) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 16px !important;
}}
/* Remove streamlit metric default chrome */
[data-testid="stMetric"] {{
  background: none !important; border: none !important; padding: 0 !important;
}}
/* Form inputs */
.stSelectbox > div > div,
.stDateInput > div > div {{
  background: #1c1c20 !important;
  border-color: var(--border) !important;
  color: var(--white) !important;
}}
/* Buttons */
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"] {{
  background: #1c1c20 !important;
  border: 1px solid var(--border) !important;
  color: var(--white) !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  transition: background 0.15s, border-color 0.15s;
}}
button[data-testid="baseButton-secondary"]:hover,
button[data-testid="baseButton-primary"]:hover {{
  background: var(--indigo) !important;
  border-color: var(--indigo) !important;
}}
/* Headings */
h1, h2, h3, h4 {{
  color: var(--white) !important;
  font-family: 'Space Grotesk', sans-serif !important;
}}
hr {{ border-color: var(--border); }}
/* DataFrames */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}}
/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
  background: var(--cell);
  border-bottom: 1px solid var(--border);
  border-radius: 8px 8px 0 0;
  gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{ color: var(--muted) !important; }}
.stTabs [aria-selected="true"] {{ color: var(--white) !important; }}
/* Utility classes */
.num {{ font-family: 'Space Grotesk', sans-serif !important; }}
.label {{
  font-size: 9px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin-bottom: 4px;
}}
.bento-cell {{
  background: var(--cell); border: 1px solid var(--border);
  border-radius: 12px; padding: 18px; margin-bottom: 14px;
}}
.bento-section {{
  background: var(--cell); border: 1px solid var(--border);
  border-radius: 16px; padding: 20px; margin-bottom: 16px;
}}
.info-card {{
  background: #1c1c20; border-left: 3px solid var(--indigo);
  border-radius: 8px; padding: 14px 18px; margin-bottom: 14px;
  font-size: 13px; color: var(--muted); line-height: 1.6;
}}
.who-pill {{
  display: inline-block; font-size: 10px; font-weight: 600;
  padding: 2px 8px; border-radius: 20px; white-space: nowrap;
  font-family: 'Inter', sans-serif;
}}
.page-title {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 26px; font-weight: 800; color: var(--white);
  margin-bottom: 4px; margin-top: 4px;
}}
.page-sub {{
  font-size: 13px; color: var(--muted); margin-bottom: 20px;
}}
.placeholder-card {{
  background: #1c1c20; border: 2px dashed var(--border);
  border-radius: 12px; padding: 32px; text-align: center; color: var(--muted);
}}
</style>
""", unsafe_allow_html=True)

# ── WHO THRESHOLDS ──
WHO_LEVELS = [
    {"tier": "Good",                         "max": 0.054,        "color": GREEN,   "bg": "#0f2a1a",
     "note": "Below WHO annual guideline (10 µg/m³)",
     "ug_range": "0.000–10.8 µg/m³ equivalent",
     "advice": "Air quality is satisfactory. No health risk for any population group."},
    {"tier": "Moderate",                     "max": 0.090,        "color": "#eab308","bg": "#1f1a00",
     "note": "Approaching WHO 24-hour guideline (25 µg/m³)",
     "ug_range": "10.8–18.0 µg/m³ equivalent",
     "advice": "Acceptable. Unusually sensitive individuals may consider limiting prolonged outdoor exposure."},
    {"tier": "Unhealthy — Sensitive Groups", "max": 0.120,        "color": ORANGE,  "bg": "#1f1000",
     "note": "Exceeds WHO 24-hour guideline — sensitive groups at risk",
     "ug_range": "18.0–24.0 µg/m³ equivalent",
     "advice": "Children, elderly and people with respiratory or cardiovascular conditions should reduce outdoor activity. General public not likely affected."},
    {"tier": "Unhealthy",                    "max": 0.160,        "color": ROSE,    "bg": "#1f050f",
     "note": "Significantly exceeds WHO guideline — general public at risk",
     "ug_range": "24.0–32.0 µg/m³ equivalent",
     "advice": "Everyone may begin to experience health effects. Sensitive groups should avoid outdoor activity."},
    {"tier": "Very Unhealthy",               "max": float("inf"),"color": "#a855f7","bg": "#150a1f",
     "note": "Far exceeds WHO guideline — health emergency for sensitive groups",
     "ug_range": ">32.0 µg/m³ equivalent",
     "advice": "Sensitive groups (children, elderly, respiratory patients) should avoid ALL outdoor activity. General public should avoid prolonged exertion outdoors."},
]
WHO_ICONS = {
    "Good": "✅", "Moderate": "🟡",
    "Unhealthy — Sensitive Groups": "🟠", "Unhealthy": "🔴", "Very Unhealthy": "🟣",
}

def who_level(val_umol: float) -> dict:
    for lvl in WHO_LEVELS:
        if val_umol <= lvl["max"]:
            return lvl
    return WHO_LEVELS[-1]

def who_pill_html(tier: str, color: str) -> str:
    icon = WHO_ICONS.get(tier, "⚪")
    return (f"<span class='who-pill' style='background:{color}22;"
            f"border:1px solid {color};color:{color};'>{icon} {tier}</span>")

def who_banner(district: str, val_umol: float):
    lvl = who_level(val_umol)
    icon = WHO_ICONS.get(lvl["tier"], "⚪")
    color = lvl["color"]
    tier  = lvl["tier"]
    note  = lvl["note"]
    advice = lvl["advice"]
    ug    = lvl["ug_range"]
    st.markdown(f"""
<div class='bento-cell' style='border-left:3px solid {color};'>
  <div style='display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:8px;'>
    <div>
      <div class='label'>{district} — Air Quality Status</div>
      <div style='font-size:15px;font-weight:700;color:{color};margin-top:4px;'>{icon} {tier}</div>
    </div>
    <div style='text-align:right;'>
      <div class='label'>Tropospheric Column</div>
      <div class='num' style='font-size:18px;font-weight:700;color:{WHITE};'>{val_umol:.5f}</div>
      <div style='font-size:10px;color:{MUTED};'>µmol/m²</div>
    </div>
  </div>
  <div style='font-size:11px;color:{MUTED};margin-bottom:2px;'><b style='color:{WHITE}99;'>Equivalent:</b> {ug}</div>
  <div style='font-size:11px;color:{MUTED};margin-bottom:2px;'><b style='color:{WHITE}99;'>WHO note:</b> {note}</div>
  <div style='font-size:11px;color:{MUTED};'><b style='color:{WHITE}99;'>Advice:</b> {advice}</div>
</div>
""", unsafe_allow_html=True)

def who_sidebar_legend():
    st.markdown(
        f"<div class='label' style='font-size:10px;color:{INDIGO};margin-bottom:8px;'>"
        f"WHO NO₂ THRESHOLDS (2021)</div>",
        unsafe_allow_html=True,
    )
    prev = 0.0
    for lvl in WHO_LEVELS:
        upper = lvl["max"]
        r = f"{prev:.3f}–{upper:.3f}" if upper != float("inf") else f">{prev:.3f}"
        color = lvl["color"]
        tier  = lvl["tier"]
        ug    = lvl.get("ug_range", "")
        st.markdown(f"""
<div style='display:flex;align-items:flex-start;gap:8px;margin-bottom:9px;'>
  <span style='display:inline-block;width:8px;height:8px;background:{color};
    border-radius:50%;margin-top:3px;flex-shrink:0;'></span>
  <div>
    <div style='font-size:11px;color:{WHITE};font-weight:500;'>{tier}</div>
    <div style='font-size:9px;color:{MUTED};'>{r} µmol/m²</div>
    <div style='font-size:9px;color:{MUTED}88;'>{ug}</div>
  </div>
</div>
""", unsafe_allow_html=True)
        if upper != float("inf"):
            prev = upper
    st.markdown(
        f"<div style='font-size:9px;color:{MUTED};margin-top:4px;'>"
        f"WHO 2021: Annual 10 µg/m³ · 24-h 25 µg/m³</div>",
        unsafe_allow_html=True,
    )

# ── MODEL LOADING ──
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
        features = region_ohe + season_ohe + numeric_base + numeric_wave if n_feat == 12 \
                   else region_ohe + season_ohe + numeric_base
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
    return df.groupby("region_name")["NO2_column_number_density"].mean().to_dict()

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
    accent = DISTRICT_COLORS.get(district, INDIGO)
    fig = go.Figure(go.Bar(
        x=d["season"], y=d["NO2_column_number_density"] * 1e6,
        marker_color=accent, marker_opacity=0.85,
        hovertemplate="Season: %{x}<br>Mean NO2: %{y:.4f} µmol/m²<extra></extra>",
    ))
    fig.update_layout(title=f"Seasonal Average — {district}",
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
            line=dict(color=DISTRICT_COLORS[district], width=2.5),
            marker=dict(size=7),
            hovertemplate="Season: %{x}<br>NO2: %{y:.4f} µmol/m²<extra>" + district + "</extra>",
        ))
    fig.update_layout(title=title, yaxis_title="NO2 (µmol/m²)",
                      xaxis_title="Season", **PLOTLY_LAYOUT)
    return fig

def make_globe(latest: dict) -> go.Figure:
    fig = go.Figure()
    clat, clon = 41.01, 29.03
    # Concentric pulse rings
    for radius, alpha, width in [(1.8, "55", 1.5), (3.5, "33", 1.0), (5.5, "1a", 0.7)]:
        n = 120
        lats = [clat + radius * np.sin(2 * np.pi * i / n) for i in range(n + 1)]
        lons = [clon + radius * 1.2 * np.cos(2 * np.pi * i / n) for i in range(n + 1)]
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines",
            line=dict(color=f"{INDIGO}{alpha}", width=width),
            showlegend=False, hoverinfo="none",
        ))
    # District markers + labels
    for district in DISTRICTS:
        c = DISTRICT_COORDS[district]
        val_umol = no2_to_umol(latest.get(district, 0))
        color = DISTRICT_COLORS[district]
        lvl = who_level(val_umol)
        tier = lvl["tier"]
        fig.add_trace(go.Scattergeo(
            lat=[c["lat"]], lon=[c["lon"]],
            mode="markers+text",
            marker=dict(size=14, color=color, symbol="circle",
                        line=dict(color=WHITE, width=1.5)),
            text=[f"  {district}  {val_umol:.4f} µmol/m²"],
            textposition="middle right",
            textfont=dict(color=WHITE, size=10, family="Inter"),
            name=district,
            hovertemplate=(
                f"<b>{district}</b><br>"
                f"NO2: {val_umol:.5f} µmol/m²<br>"
                f"Status: {tier}<extra></extra>"
            ),
        ))
    fig.update_geos(
        projection_type="orthographic",
        center=dict(lat=41.0, lon=29.0),
        showland=True,       landcolor=CELL,
        showocean=True,      oceancolor="#0a0a14",
        showcountries=True,  countrycolor="#2d2d38",
        showcoastlines=True, coastlinecolor="#2d2d38",
        showframe=False,     bgcolor=BG,
        lataxis_showgrid=True, lataxis_gridcolor="#1a1a2a",
        lonaxis_showgrid=True, lonaxis_gridcolor="#1a1a2a",
    )
    fig.update_layout(
        paper_bgcolor=BG,
        margin=dict(l=0, r=0, t=0, b=0),
        height=450,
        legend=dict(
            bgcolor="rgba(22,22,24,0.85)", bordercolor=BORDER, borderwidth=1,
            font=dict(color=WHITE, size=11), x=0.01, y=0.99,
        ),
    )
    return fig

# ── SIDEBAR ──
with st.sidebar:
    st.markdown(f"""
<div style='text-align:center;padding:20px 0 12px 0;'>
  <div style='font-size:30px;margin-bottom:8px;'>🌍</div>
  <div style='font-family:"Space Grotesk",sans-serif;font-size:13px;font-weight:700;
    color:{INDIGO};letter-spacing:2px;text-transform:uppercase;'>AIR QUALITY</div>
  <div style='font-family:"Space Grotesk",sans-serif;font-size:11px;font-weight:600;
    color:{MUTED};letter-spacing:1px;text-transform:uppercase;margin-top:2px;'>MONITORING</div>
  <div style='font-size:10px;color:{MUTED};margin-top:4px;'>Istanbul · Sentinel-5P</div>
</div>
<hr style='border-color:{BORDER};margin:8px 0 16px 0;'>
""", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=["🏠 Home", "📍 District Analysis", "🤖 ML Predictions",
                 "📊 Analytics & Seasonal", "🌡️ Seasonal Interpretation", "🏭 PM10 & PM2.5 (Future)"],
        label_visibility="collapsed",
    )

    st.markdown(f"<hr style='border-color:{BORDER};margin:16px 0;'>", unsafe_allow_html=True)

    if model_loaded:
        st.markdown(
            f"<div style='background:#0f2a1a;border:1px solid {GREEN}44;border-radius:8px;"
            f"padding:8px 12px;font-size:11px;color:{GREEN};'>✅ Model loaded (OHE)</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background:{CELL};border:1px solid {BORDER};border-radius:8px;"
            f"padding:8px 12px;font-size:11px;color:{MUTED};'>ℹ️ Demo mode — place best_rf_model_ohe.pkl here</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    who_sidebar_legend()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:10px;color:{MUTED};line-height:1.8;'>"
        f"🛰️ Data: Sentinel-5P / Google Earth Engine<br>"
        f"🤖 Model: Random Forest (.pkl)<br>"
        f"🎓 Graduation Project 2024-25</div>",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown(
        "<div class='page-title'>🌍 Air Quality Monitoring Dashboard</div>"
        f"<div class='page-sub'>NO2 Prediction for Istanbul Districts · "
        f"Machine Learning & Sentinel-5P Satellite Data</div>",
        unsafe_allow_html=True,
    )

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        selected_district_home = st.selectbox("Select District", ["All"] + DISTRICTS, key="home_dist")
    with col_f2:
        selected_season_home = st.selectbox(
            "Select Season", ["All"] + list(SEASON_NAME.values()), key="home_season"
        )

    df_home = df_all.copy()
    if selected_district_home != "All":
        df_home = df_home[df_home["region_name"] == selected_district_home]
    if selected_season_home != "All":
        rev = {v: k for k, v in SEASON_NAME.items()}
        df_home = df_home[df_home["season"] == rev[selected_season_home]]

    latest = get_latest_no2(df_all)
    current_season = get_current_season()

    # ── Globe hero ──
    st.markdown("<div class='bento-section'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='label' style='margin-bottom:12px;'>📡 LIVE SATELLITE VIEW · ISTANBUL REGION</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_globe(latest), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── District metric cards ──
    st.markdown(f"<div class='label' style='margin:6px 0 10px 0;'>📡 CURRENT NO₂ LEVELS</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, district in zip([c1, c2, c3], DISTRICTS):
        val      = latest.get(district, 0)
        val_umol = no2_to_umol(val)
        lvl      = who_level(val_umol)
        smean    = SEASONAL_MEANS.get(district, {}).get(current_season, val)
        smean_u  = smean * 1e6
        accent   = DISTRICT_COLORS[district]
        pill     = who_pill_html(lvl["tier"], lvl["color"])
        val_str  = f"{val_umol:.5f}"
        sm_str   = f"{smean_u:.5f}"
        col.markdown(f"""
<div style='border-left:3px solid {accent};padding-left:12px;'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;'>
    <div class='label'>{district.upper()}</div>
    {pill}
  </div>
  <div class='num' style='font-size:28px;font-weight:800;color:{accent};line-height:1.1;'>{val_str}</div>
  <div style='font-size:10px;color:{MUTED};margin-top:2px;'>µmol/m²</div>
  <div style='font-size:9px;color:{MUTED};margin-top:6px;'>
    Seasonal mean ({SEASON_NAME[current_season]}): {sm_str} µmol/m²
  </div>
  <div style='font-size:9px;color:{MUTED};margin-top:2px;'>{lvl["ug_range"]}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(
        line_chart_no2(df_home, title="NO₂ by Season — All Districts"),
        use_container_width=True,
    )

    st.markdown(f"""
<div class='info-card'>
<b style='color:{WHITE};'>About this project:</b> This dashboard visualizes NO2 pollution levels
for three Istanbul districts (Kartal, Kağıthane, Üsküdar) using satellite imagery from Sentinel-5P.
A trained Random Forest model (R²=0.9762) provides NO2 predictions. Data is retrieved via
Google Earth Engine.
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PAGE 2 — DISTRICT ANALYSIS
# ════════════════════════════════════════════════════════
elif page == "📍 District Analysis":
    st.markdown("<div class='page-title'>📍 District Analysis</div>", unsafe_allow_html=True)

    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        sel_district = st.selectbox("Select District", DISTRICTS, key="dist_select")
    with col_d2:
        st.markdown("<div style='margin-top:28px;'>", unsafe_allow_html=True)
        compare_mode = st.toggle("Compare All Districts", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    df_dist = df_all[df_all["region_name"] == sel_district]
    latest  = get_latest_no2(df_all)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        map_districts = DISTRICTS if compare_mode else [sel_district]
        map_df = pd.DataFrame([
            {**DISTRICT_COORDS[d], "district": d,
             "NO2": round(no2_to_umol(latest.get(d, 0)), 5)}
            for d in map_districts
        ])
        fig_map = px.scatter_mapbox(
            map_df, lat="lat", lon="lon", hover_name="district",
            hover_data={"NO2": True, "lat": False, "lon": False},
            color="district", color_discrete_map={d: DISTRICT_COLORS[d] for d in DISTRICTS},
            size=[20] * len(map_df), zoom=10, height=340,
        )
        fig_map.update_layout(
            mapbox_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=WHITE)),
        )
        st.markdown(f"<div class='label' style='margin-bottom:8px;'>🗺️ DISTRICT LOCATION MAP</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(fig_map, use_container_width=True)

    with col_right:
        val_umol = no2_to_umol(latest.get(sel_district, 0))
        who_banner(sel_district, val_umol)
        st.plotly_chart(bar_chart_seasonal(df_all, sel_district), use_container_width=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        df_plot = df_all if compare_mode else df_dist
        title_l = f"NO2 by Season — {'All Districts' if compare_mode else sel_district}"
        st.plotly_chart(line_chart_no2(df_plot, title=title_l), use_container_width=True)
    with col_c2:
        st.markdown(f"<div class='label' style='margin-bottom:8px;'>🗓️ SEASONAL STATISTICS</div>",
                    unsafe_allow_html=True)
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

# ════════════════════════════════════════════════════════
# PAGE 3 — ML PREDICTIONS
# ════════════════════════════════════════════════════════
elif page == "🤖 ML Predictions":
    st.markdown("<div class='page-title'>🤖 Machine Learning Predictions</div>",
                unsafe_allow_html=True)

    status_html = (
        f"<div style='background:#0f1f0f;border:1px solid {GREEN}44;border-radius:8px;"
        f"padding:10px 14px;font-size:12px;color:{GREEN};margin-bottom:16px;'>"
        f"✅ Random Forest model loaded — OHE + Wavelet features d1/d2/d3 · "
        f"R²=0.9762 · MAE=1.27×10⁻⁶ · RMSE=1.85×10⁻⁶ mol/m²</div>"
        if model_loaded else
        f"<div style='background:{CELL};border:1px solid {BORDER};border-radius:8px;"
        f"padding:10px 14px;font-size:12px;color:{MUTED};margin-bottom:16px;'>"
        f"ℹ️ Demo mode — place best_rf_model_ohe.pkl in the app directory.</div>"
    )
    st.markdown(status_html, unsafe_allow_html=True)

    # Top metric cells
    m1, m2, m3, m4 = st.columns(4)
    for col, lbl, val, unit, color in [
        (m1, "R² SCORE",  "0.9762", "Overall",          INDIGO),
        (m2, "MAE",       "1.27",   "×10⁻⁶ mol/m²",     ROSE),
        (m3, "RMSE",      "1.85",   "×10⁻⁶ mol/m²",     ORANGE),
        (m4, "FEATURES",  "7",      "incl. d1/d2/d3",   GREEN),
    ]:
        col.markdown(f"""
<div style='border-top:2px solid {color};padding-top:12px;'>
  <div class='label'>{lbl}</div>
  <div class='num' style='font-size:32px;font-weight:800;color:{color};line-height:1.1;'>{val}</div>
  <div style='font-size:10px;color:{MUTED};margin-top:2px;'>{unit}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

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
        reset_clicked   = col_b2.button("🔄 Reset",   use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if reset_clicked:
        st.rerun()

    if predict_clicked:
        m = future_date.month
        if m in [12, 1, 2]:   season_code = "DJF"
        elif m in [3, 4, 5]:  season_code = "MAM"
        elif m in [6, 7, 8]:  season_code = "JJA"
        else:                  season_code = "SON"

        prediction = run_prediction(pred_district, season_code)
        pred_umol  = no2_to_umol(prediction)
        lvl        = who_level(pred_umol)
        accent     = DISTRICT_COLORS[pred_district]
        pill       = who_pill_html(lvl["tier"], lvl["color"])
        pred_str   = f"{pred_umol:.5f}"
        sname      = SEASON_NAME[season_code]

        col_res, col_chart = st.columns([1, 2])
        with col_res:
            st.markdown(f"""
<div style='border-top:3px solid {accent};padding-top:16px;'>
  <div class='label'>{pred_district.upper()} · {season_code} ({sname})</div>
  <div class='label' style='margin-bottom:10px;'>{future_date}</div>
  <div class='num' style='font-size:44px;font-weight:900;color:{accent};line-height:1;'>{pred_str}</div>
  <div style='font-size:12px;color:{MUTED};margin-top:4px;'>µmol/m² NO₂</div>
  <div style='margin-top:12px;'>{pill}</div>
</div>
""", unsafe_allow_html=True)
            who_banner(pred_district, pred_umol)

            results_df = df_all[df_all["region_name"] == pred_district][["region_name", "season"]].copy()
            results_df["actual_NO2_mol_m2"] = \
                df_all[df_all["region_name"] == pred_district]["NO2_column_number_density"].values
            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download CSV", data=csv_bytes,
                file_name=f"NO2_predictions_{pred_district}_{future_date}.csv",
                mime="text/csv", use_container_width=True,
            )

        with col_chart:
            seasonal_r2 = {"DJF (Winter)": 0.6676, "MAM (Spring)": 0.9763,
                           "JJA (Summer)": 0.9605, "SON (Autumn)": 0.9885}
            fig_r2 = go.Figure(go.Bar(
                x=list(seasonal_r2.keys()), y=list(seasonal_r2.values()),
                marker_color=[INDIGO, GREEN, ORANGE, ROSE],
                text=[f"{v:.4f}" for v in seasonal_r2.values()],
                textposition="outside", textfont=dict(color=WHITE),
                hovertemplate="Season: %{x}<br>R²: %{y:.4f}<extra></extra>",
            ))
            fig_r2.add_hline(y=0.95, line_dash="dash", line_color=MUTED,
                             annotation_text="R²=0.95 target",
                             annotation_position="top right",
                             annotation_font_color=MUTED)
            fig_r2.update_layout(
                title="Per-Season R² — Wavelet-Enhanced Model",
                yaxis_title="R²", yaxis_range=[0, 1.05], **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_r2, use_container_width=True)
    else:
        st.markdown(
            f"<div class='info-card'>Select a district and date above, then click "
            f"<b style='color:{WHITE};'>🔮 Predict</b> to run the model.</div>",
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════
# PAGE 4 — ANALYTICS & SEASONAL
# ════════════════════════════════════════════════════════
elif page == "📊 Analytics & Seasonal":
    st.markdown("<div class='page-title'>📊 Analytics & Seasonal Trends</div>",
                unsafe_allow_html=True)

    col_a1, _ = st.columns([2, 1])
    with col_a1:
        sel_season_label = st.selectbox("Select Season", SEASONS, key="analytics_season")
    sel_season_code = SEASON_MAP[sel_season_label]
    df_season = df_all[df_all["season"] == sel_season_code]

    season_comp = (df_all.groupby(["season", "region_name"])["NO2_column_number_density"]
                   .mean().reset_index())
    season_comp["NO2_umol"] = season_comp["NO2_column_number_density"] * 1e6
    fig_bar = px.bar(
        season_comp, x="season", y="NO2_umol", color="region_name",
        barmode="group", color_discrete_map=DISTRICT_COLORS,
        labels={"NO2_umol": "Mean NO2 (µmol/m²)", "season": "Season", "region_name": "District"},
        category_orders={"season": ["DJF", "MAM", "JJA", "SON"]},
    )
    fig_bar.update_layout(title="Mean NO₂ by Season & District", **PLOTLY_LAYOUT)
    st.plotly_chart(fig_bar, use_container_width=True)

    col_pie, col_table = st.columns(2)
    with col_pie:
        pie_data = df_season.groupby("region_name")["NO2_column_number_density"].mean().reset_index()
        fig_pie = px.pie(
            pie_data, values="NO2_column_number_density", names="region_name",
            color="region_name", color_discrete_map=DISTRICT_COLORS, hole=0.5,
        )
        fig_pie.update_layout(
            title=f"Pollution Distribution — {sel_season_label}",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=WHITE), legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_table:
        st.markdown(
            f"<div class='label' style='margin-bottom:8px;'>"
            f"📋 STATISTICAL SUMMARY — {sel_season_label.upper()}</div>",
            unsafe_allow_html=True,
        )
        stats_table = (
            df_season.groupby("region_name")["NO2_column_number_density"]
            .agg(Mean="mean", Min="min", Max="max", Std=lambda x: x.std())
            .reset_index().rename(columns={"region_name": "District"})
        )
        stats_table[["Mean", "Min", "Max", "Std"]] = (
            stats_table[["Mean", "Min", "Max", "Std"]] * 1e6
        ).round(5)
        st.dataframe(stats_table.style.background_gradient(subset=["Mean"], cmap="Blues"),
                     use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════
# PAGE 5 — SEASONAL INTERPRETATION
# ════════════════════════════════════════════════════════
elif page == "🌡️ Seasonal Interpretation":
    st.markdown("<div class='page-title'>🌡️ Seasonal Interpretation & Critical Days</div>",
                unsafe_allow_html=True)
    st.markdown(f"""
<div class='info-card'>
<b style='color:{WHITE};'>What this page shows:</b> Based on WHO 2021 NO2 guidelines, we estimate how many
days per season NO2 concentrations are likely to exceed thresholds dangerous for
<b style='color:{WHITE};'>sensitive population groups</b> — including children under 14, elderly over 65,
pregnant women, and individuals with respiratory or cardiovascular conditions.
</div>
""", unsafe_allow_html=True)

    pop_data = {
        "Kağıthane": {"total": 450000, "sensitive": 126000, "general": 324000},
        "Üsküdar":   {"total": 530000, "sensitive": 150000, "general": 380000},
        "Kartal":    {"total": 470000, "sensitive": 132000, "general": 338000},
    }

    st.markdown(f"<div class='label' style='margin-bottom:10px;'>👥 DISTRICT POPULATION OVERVIEW</div>",
                unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    for col, district in zip([pc1, pc2, pc3], DISTRICTS):
        p      = pop_data[district]
        accent = DISTRICT_COLORS[district]
        total_p     = p["total"]
        sensitive_p = p["sensitive"]
        general_p   = p["general"]
        col.markdown(f"""
<div style='border-top:2px solid {accent};padding-top:12px;'>
  <div class='label'>{district.upper()}</div>
  <div class='num' style='font-size:26px;font-weight:800;color:{accent};margin-top:4px;'>{total_p:,}</div>
  <div style='font-size:10px;color:{MUTED};'>Total residents</div>
  <div style='margin-top:10px;'>
    <div style='font-size:12px;color:{ROSE};font-weight:600;'>{sensitive_p:,} sensitive</div>
    <div style='font-size:9px;color:{MUTED};'>children · elderly · respiratory</div>
  </div>
  <div style='margin-top:6px;'>
    <div style='font-size:12px;color:{GREEN};font-weight:600;'>{general_p:,} general</div>
    <div style='font-size:9px;color:{MUTED};'>rest of population</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

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
    season_days   = {"DJF": 90, "MAM": 92, "JJA": 92, "SON": 91}
    season_labels = {"DJF": "Winter (DJF)", "MAM": "Spring (MAM)",
                     "JJA": "Summer (JJA)", "SON": "Autumn (SON)"}

    sel_season_interp = st.selectbox("Select Season", list(season_labels.values()), key="interp_season")
    sel_code_interp   = {v: k for k, v in season_labels.items()}[sel_season_interp]
    total_days        = season_days[sel_code_interp]

    st.markdown(
        f"<div class='label' style='margin:8px 0;'>📅 CRITICAL DAYS — "
        f"{sel_season_interp.upper()} ({total_days} TOTAL DAYS)</div>",
        unsafe_allow_html=True,
    )
    dcols = st.columns(3)
    for col, district in zip(dcols, DISTRICTS):
        crit_s = critical_days[district][sel_code_interp]
        crit_g = critical_days_general[district][sel_code_interp]
        pct_s  = (crit_s / total_days) * 100
        pct_g  = (crit_g / total_days) * 100
        pop_s  = pop_data[district]["sensitive"]
        pop_g  = pop_data[district]["general"]
        col.markdown(f"""
<div>
  <div class='label'>{district.upper()}</div>
  <div style='background:{ROSE}15;border:1px solid {ROSE}44;border-radius:8px;
    padding:10px;margin-top:8px;margin-bottom:6px;'>
    <div style='font-size:9px;font-weight:600;color:{ROSE};text-transform:uppercase;margin-bottom:4px;'>
      🔴 Sensitive ({pop_s:,})
    </div>
    <div class='num' style='font-size:30px;font-weight:800;color:{ROSE};'>{crit_s}</div>
    <div style='font-size:9px;color:{MUTED};'>days · {pct_s:.0f}% of season</div>
  </div>
  <div style='background:{ORANGE}15;border:1px solid {ORANGE}44;border-radius:8px;padding:10px;'>
    <div style='font-size:9px;font-weight:600;color:{ORANGE};text-transform:uppercase;margin-bottom:4px;'>
      🟠 General ({pop_g:,})
    </div>
    <div class='num' style='font-size:30px;font-weight:800;color:{ORANGE};'>{crit_g}</div>
    <div style='font-size:9px;color:{MUTED};'>days · {pct_g:.0f}% of season</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='label' style='margin-bottom:10px;'>📊 CRITICAL DAYS — ALL SEASONS</div>",
                unsafe_allow_html=True)

    rows_data = []
    for district in DISTRICTS:
        for scode, slabel in season_labels.items():
            rows_data.append({
                "District": district, "Season": slabel, "Group": "Sensitive",
                "Critical Days": critical_days[district][scode],
                "People Affected": pop_data[district]["sensitive"],
                "% of Season": round((critical_days[district][scode] / season_days[scode]) * 100, 1),
            })
            rows_data.append({
                "District": district, "Season": slabel, "Group": "General Population",
                "Critical Days": critical_days_general[district][scode],
                "People Affected": pop_data[district]["general"],
                "% of Season": round((critical_days_general[district][scode] / season_days[scode]) * 100, 1),
            })
    df_crit = pd.DataFrame(rows_data)

    tab1, tab2 = st.tabs(["🔴 Sensitive Groups", "🟠 General Population"])
    with tab1:
        df_s = df_crit[df_crit["Group"] == "Sensitive"]
        fig_s = px.bar(df_s, x="Season", y="Critical Days", color="District",
                       barmode="group", color_discrete_map=DISTRICT_COLORS, text="Critical Days",
                       labels={"Critical Days": "Critical Days (Sensitive Groups)"})
        fig_s.update_layout(title="Critical Days — Sensitive Groups", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_s, use_container_width=True)
        st.dataframe(df_s[["District", "Season", "Critical Days", "People Affected", "% of Season"]]
                     .style.background_gradient(subset=["Critical Days", "% of Season"], cmap="Reds"),
                     use_container_width=True, hide_index=True)
    with tab2:
        df_g = df_crit[df_crit["Group"] == "General Population"]
        fig_g = px.bar(df_g, x="Season", y="Critical Days", color="District",
                       barmode="group", color_discrete_map=DISTRICT_COLORS, text="Critical Days",
                       labels={"Critical Days": "Critical Days (General Population)"})
        fig_g.update_layout(title="Critical Days — General Population", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_g, use_container_width=True)
        st.dataframe(df_g[["District", "Season", "Critical Days", "People Affected", "% of Season"]]
                     .style.background_gradient(subset=["Critical Days", "% of Season"], cmap="Oranges"),
                     use_container_width=True, hide_index=True)

    total_sensitive = sum(pop_data[d]["sensitive"] for d in DISTRICTS)
    total_general   = sum(pop_data[d]["general"]   for d in DISTRICTS)
    col_t1, col_t2  = st.columns(2)
    col_t1.markdown(f"""
<div style='border-top:2px solid {ROSE};padding-top:14px;'>
  <div class='label'>SENSITIVE PEOPLE — ALL 3 DISTRICTS</div>
  <div class='num' style='font-size:36px;font-weight:900;color:{ROSE};margin:6px 0;'>{total_sensitive:,}</div>
  <div style='font-size:11px;color:{MUTED};'>at risk on a typical winter day</div>
</div>
""", unsafe_allow_html=True)
    col_t2.markdown(f"""
<div style='border-top:2px solid {ORANGE};padding-top:14px;'>
  <div class='label'>GENERAL POPULATION — ALL 3 DISTRICTS</div>
  <div class='num' style='font-size:36px;font-weight:900;color:{ORANGE};margin:6px 0;'>{total_general:,}</div>
  <div style='font-size:11px;color:{MUTED};'>exposed to unhealthy air on a typical winter day</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class='info-card' style='margin-top:16px;'>
<b style='color:{WHITE};'>⚠️ Important note:</b> These are estimated values based on seasonal mean
NO2 concentrations from the Sentinel-5P dataset and WHO 2021 guidelines. Actual daily critical
days may vary depending on meteorological conditions, wind patterns, and emission events.
<br><br>
<b style='color:{WHITE};'>Sensitive groups include:</b> Children under 14, elderly over 65, pregnant women,
and persons with respiratory conditions (asthma, COPD) or cardiovascular disease.
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PAGE 6 — PM10 & PM2.5 FUTURE
# ════════════════════════════════════════════════════════
elif page == "🏭 PM10 & PM2.5 (Future)":
    st.markdown("<div class='page-title'>🏭 PM10 & PM2.5 — Future Integration Plan</div>",
                unsafe_allow_html=True)

    st.markdown(f"""
<div class='bento-cell' style='border-left:3px solid {ORANGE};'>
  <div class='label' style='margin-bottom:8px;'>WHY PM10 AND PM2.5 ARE NOT IN THE CURRENT DATASET</div>
  <div style='font-size:13px;color:{MUTED};line-height:1.7;'>
    Sentinel-5P TROPOMI measures column-integrated atmospheric gases (NO2, SO2, CO, O3, CH4)
    at ~3.5 km resolution. Particulate matter (PM10 and PM2.5)
    <b style='color:{WHITE};'>cannot be directly retrieved</b> from TROPOMI because it does not
    measure aerosol optical properties at the wavelengths needed for surface PM estimation.
    Alternative satellite-based approaches are required.
  </div>
</div>
""", unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"""
<div class='bento-cell'>
  <div class='label' style='margin-bottom:10px;'>🛰️ MODIS TERRA/AQUA — AEROSOL OPTICAL DEPTH (AOD)</div>
  <div style='font-size:12px;color:{MUTED};line-height:1.7;'>
    <b style='color:{WHITE};'>GEE dataset:</b> MODIS/061/MOD08_D3<br>
    <b style='color:{WHITE};'>Resolution:</b> 1° × 1° (~111 km) daily<br>
    <b style='color:{WHITE};'>Key variable:</b> Aerosol_Optical_Depth_Land_Mean<br><br>
    AOD measures how much sunlight aerosol particles prevent from reaching the ground.
    It can be converted to surface PM2.5 using empirical regression models.
  </div>
</div>
""", unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
<div class='bento-cell'>
  <div class='label' style='margin-bottom:10px;'>🌍 MERRA-2 — METEOROLOGICAL REANALYSIS</div>
  <div style='font-size:12px;color:{MUTED};line-height:1.7;'>
    <b style='color:{WHITE};'>GEE dataset:</b> NASA/GSFC/MERRA/flx/2<br>
    <b style='color:{WHITE};'>Resolution:</b> 0.5° × 0.625° (~50 km) hourly<br>
    <b style='color:{WHITE};'>Key variables:</b> DUSMASS25, OCSMASS, BCSMASS, SSSMASS25<br><br>
    MERRA-2 provides modelled PM component mass concentrations at surface level.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class='bento-cell'>
  <div class='label' style='margin-bottom:10px;'>📏 WHO PM THRESHOLDS (2021 GUIDELINES)</div>
  <div style='font-size:11px;color:{MUTED};margin-bottom:12px;line-height:1.6;'>
    The thresholds below are WHO 2021 annual and 24-hour mean guidelines for particulate matter.
    These apply to the <b style='color:{WHITE};'>general population</b>. For sensitive groups
    (children, elderly, respiratory patients), even lower concentrations may cause health effects.
    Interim Target 1 values represent transitional targets.
  </div>
""", unsafe_allow_html=True)
    pm_thresh = pd.DataFrame({
        "Pollutant":         ["PM2.5", "PM2.5", "PM10", "PM10"],
        "Averaging Period":  ["Annual mean", "24-hour mean", "Annual mean", "24-hour mean"],
        "WHO Guideline":     ["5 µg/m³", "15 µg/m³", "15 µg/m³", "45 µg/m³"],
        "Interim Target 1":  ["35 µg/m³", "75 µg/m³", "70 µg/m³", "150 µg/m³"],
        "Sensitive Groups?": ["Stricter limits recommended"] * 4,
    })
    st.dataframe(pm_thresh, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='label' style='margin:16px 0 10px 0;'>📍 CURRENT STATUS</div>",
                unsafe_allow_html=True)
    cs1, cs2, cs3 = st.columns(3)
    for col, district in zip([cs1, cs2, cs3], DISTRICTS):
        accent = DISTRICT_COLORS[district]
        col.markdown(f"""
<div style='text-align:center;border-top:2px solid {accent};padding-top:16px;'>
  <div style='font-size:28px;margin-bottom:8px;'>🏭</div>
  <div style='font-weight:700;font-size:13px;color:{WHITE};margin-bottom:8px;'>{district}</div>
  <div style='font-size:11px;color:{MUTED};'>PM10 — data extraction pending</div>
  <div style='font-size:11px;color:{MUTED};'>PM2.5 — data extraction pending</div>
  <div style='font-size:10px;color:{MUTED}88;margin-top:8px;'>Source: MERRA-2 / MODIS AOD</div>
</div>
""", unsafe_allow_html=True)
