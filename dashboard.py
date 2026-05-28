"""
Air Quality Monitoring Dashboard
NO₂ Prediction for Istanbul using Machine Learning and Satellite Data
Districts: Kartal, Kağıthane, Üsküdar
Data Source: Sentinel-5P via Google Earth Engine
Model: Random Forest — One-Hot Encoding (best_rf_model_ohe.pkl)
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

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Istanbul Air Quality Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# COLOR THEME
# ─────────────────────────────────────────────
COLORS = {
    "dark_blue":  "#0D1B2A",
    "mid_blue":   "#1B3A5C",
    "sky_blue":   "#2196F3",
    "light_blue": "#64B5F6",
    "green":      "#4CAF50",
    "soft_gray":  "#B0BEC5",
    "white":      "#F5F9FF",
    "kartal":     "#2196F3",
    "kagithane":  "#4CAF50",
    "uskudar":    "#FF9800",
}

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DISTRICTS = ["Kartal", "Kağıthane", "Üsküdar"]
SEASONS    = ["DJF / Winter", "MAM / Spring", "JJA / Summer", "SON / Autumn"]
SEASON_MAP = {"DJF / Winter": "DJF", "MAM / Spring": "MAM",
              "JJA / Summer": "JJA", "SON / Autumn": "SON"}
SEASON_NAME = {"DJF": "Winter", "MAM": "Spring", "JJA": "Summer", "SON": "Autumn"}

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

# ─────────────────────────────────────────────
# WHO THRESHOLD SYSTEM
# ─────────────────────────────────────────────
WHO_LEVELS = [
    {"tier": "Good",                       "max": 0.054, "color": "#4CAF50", "bg": "#1a3a1a",
     "note": "Below WHO annual guideline",
     "advice": "Air quality is satisfactory. No health risk."},
    {"tier": "Moderate",                   "max": 0.090, "color": "#FFC107", "bg": "#3a2e00",
     "note": "Approaching WHO 24-hour guideline",
     "advice": "Acceptable. Unusually sensitive individuals should limit prolonged outdoor exposure."},
    {"tier": "Unhealthy — Sensitive Groups","max": 0.120, "color": "#FF9800", "bg": "#3a2000",
     "note": "Exceeds WHO 24-hour guideline (25 µg/m³ equivalent)",
     "advice": "Children, elderly and people with respiratory conditions should reduce outdoor activity."},
    {"tier": "Unhealthy",                  "max": 0.160, "color": "#F44336", "bg": "#3a0a0a",
     "note": "Significantly exceeds WHO guideline",
     "advice": "Everyone may begin to experience health effects. Sensitive groups should avoid outdoor activity."},
    {"tier": "Very Unhealthy",             "max": float("inf"), "color": "#9C27B0", "bg": "#2a0a3a",
     "note": "Far exceeds WHO guideline — health emergency level",
     "advice": "Health warnings issued. Everyone should avoid prolonged outdoor exertion."},
]

WHO_ICONS = {"Good": "✅", "Moderate": "🟡",
             "Unhealthy — Sensitive Groups": "🟠", "Unhealthy": "🔴", "Very Unhealthy": "🟣"}

def who_level(val_umol: float) -> dict:
    for lvl in WHO_LEVELS:
        if val_umol <= lvl["max"]:
            return lvl
    return WHO_LEVELS[-1]

def who_banner(district: str, val_umol: float):
    lvl  = who_level(val_umol)
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
      <div style='font-size:13px; color:{COLORS["soft_gray"]}; margin-bottom:3px;'>
        <b>WHO note:</b> {lvl["note"]}
      </div>
      <div style='font-size:13px; color:{COLORS["soft_gray"]};'>
        <b>Advice:</b> {lvl["advice"]}
      </div>
    </div>
    """, unsafe_allow_html=True)

def who_sidebar_legend():
    st.markdown(f"<div style='font-size:13px; font-weight:700; color:{COLORS['sky_blue']}; margin-bottom:8px;'>WHO NO₂ Thresholds (2021)</div>", unsafe_allow_html=True)
    rows = ""
    prev = 0.0
    for lvl in WHO_LEVELS:
        upper = lvl["max"]
        r = f"{prev:.3f}–{upper:.3f}" if upper != float("inf") else f">{prev:.3f}"
        rows += f"<tr><td style='padding:3px 6px; color:{COLORS['soft_gray']}; font-size:12px;'><span style='display:inline-block;width:10px;height:10px;background:{lvl['color']};border-radius:50%;vertical-align:middle;margin-right:5px;'></span>{lvl['tier']}</td><td style='padding:3px 6px; font-size:11px; color:{COLORS['soft_gray']};'>{r}</td></tr>"
        if upper != float("inf"):
            prev = upper
    st.markdown(f"<table style='width:100%;border-collapse:collapse;'>{rows}</table><div style='font-size:10px;color:{COLORS['soft_gray']};margin-top:4px;'>Annual 10 µg/m³ · 24-h 25 µg/m³</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        saved = joblib.load("best_rf_model_ohe.pkl")
        # Mustafa's new model is a sklearn Pipeline directly
        if hasattr(saved, 'predict'):
            return saved, None, True
        # fallback: old format with dict
        return saved["model"], saved.get("preprocessor"), True
    except FileNotFoundError:
        return None, None, False
    except Exception as e:
        st.warning(f"Model load error: {e}")
        return None, None, False

rf_model, preprocessor, model_loaded = load_model()

# ── Wavelet lookup table (mean d1,d2,d3 per district/season) ──
# Used to pass correct wavelet features when predicting
WAVELET_LOOKUP = {
    ("Kartal",    "DJF"): {"d1": 0.0, "d2": 0.0, "d3": -2.823e-07},
    ("Kartal",    "JJA"): {"d1": 0.0, "d2": 0.0, "d3": -1.415e-06},
    ("Kartal",    "MAM"): {"d1": 0.0, "d2": 0.0, "d3":  2.823e-07},
    ("Kartal",    "SON"): {"d1": 0.0, "d2": 0.0, "d3":  1.415e-06},
    ("Kağıthane", "DJF"): {"d1": 0.0, "d2": 0.0, "d3":  0.0},
    ("Kağıthane", "JJA"): {"d1": 0.0, "d2": 0.0, "d3":  0.0},
    ("Kağıthane", "MAM"): {"d1": 0.0, "d2": 0.0, "d3":  0.0},
    ("Kağıthane", "SON"): {"d1": 0.0, "d2": 0.0, "d3":  0.0},
    ("Üsküdar",   "DJF"): {"d1": 0.0, "d2": 0.0, "d3":  0.0},
    ("Üsküdar",   "JJA"): {"d1": 0.0, "d2": 0.0, "d3":  0.0},
    ("Üsküdar",   "MAM"): {"d1": 0.0, "d2": 0.0, "d3":  0.0},
    ("Üsküdar",   "SON"): {"d1": 0.0, "d2": 0.0, "d3":  0.0},
}

def run_prediction(district: str, season_code: str) -> float:
    """Return predicted NO₂ in mol/m² using the real model or a demo fallback."""
    if model_loaded:
        coords  = DISTRICT_COORDS[district]
        wavelet = WAVELET_LOOKUP.get((district, season_code), {"d1": 0.0, "d2": 0.0, "d3": 0.0})
        row = pd.DataFrame([{
            "region_name": district,
            "season":      season_code,
            "lat":         coords["lat"],
            "lon":         coords["lon"],
            "d1":          wavelet["d1"],
            "d2":          wavelet["d2"],
            "d3":          wavelet["d3"],
        }])
        # Mustafa's model is a Pipeline — call predict directly
        return float(rf_model.predict(row)[0])
    else:
        base          = {"Kartal": 0.000110, "Kağıthane": 0.000125, "Üsküdar": 0.000105}
        season_factor = {"DJF": 1.20, "MAM": 1.05, "JJA": 0.85, "SON": 1.00}
        return base[district] * season_factor[season_code]

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load real Sentinel-5P dataset.
    Falls back to synthetic data if CSV not found.
    """
    try:
        df = pd.read_csv("NO2_Istanbul_Seasonal.csv")

        # Extract lat/lon from .geo if needed
        if "lat" not in df.columns or df["lat"].isnull().all():
            def extract_coords(geo_str):
                try:
                    geo = json.loads(geo_str)
                    return geo["coordinates"][1], geo["coordinates"][0]
                except Exception:
                    return None, None
            df[["lat", "lon"]] = df[".geo"].apply(
                lambda x: pd.Series(extract_coords(x))
            )

        # Add synthetic date for trend charts (seasonal data has no date column)
        if "date" not in df.columns:
            season_dates = {"DJF": "2024-01-15", "MAM": "2024-04-15",
                            "JJA": "2024-07-15", "SON": "2024-10-15"}
            df["date"] = pd.to_datetime(df["season"].map(season_dates))

        df["season_name"] = df["season"].map(SEASON_NAME)
        return df

    except FileNotFoundError:
        # Synthetic fallback
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
                        "date":                      pd.to_datetime(date_str),
                        "region_name":               district,
                        "season":                    season,
                        "season_name":               SEASON_NAME[season],
                        "NO2_column_number_density": max(0, base_val * factor + noise),
                        "lat":  DISTRICT_COORDS[district]["lat"],
                        "lon":  DISTRICT_COORDS[district]["lon"],
                    })
        return pd.DataFrame(rows)

df_all = load_data()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def no2_to_umol(val: float) -> float:
    return val * 1e6

def get_latest_no2(df: pd.DataFrame) -> dict:
    latest = df.groupby("region_name")["NO2_column_number_density"].mean()
    return latest.to_dict()

# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
def bar_chart_seasonal(df: pd.DataFrame, district: str) -> go.Figure:
    # BUG FIX: added () after .mean and .reset_index()
    d = (df[df["region_name"] == district]
         .groupby("season")["NO2_column_number_density"]
         .mean()
         .reset_index())
    season_order = ["DJF", "MAM", "JJA", "SON"]
    d["season"] = pd.Categorical(d["season"], categories=season_order, ordered=True)
    d = d.sort_values("season")
    fig = go.Figure(go.Bar(
        x=d["season"],
        y=d["NO2_column_number_density"] * 1e6,
        marker_color=[COLORS["sky_blue"], COLORS["green"], COLORS["uskudar"], COLORS["soft_gray"]],
        hovertemplate="Season: %{x}<br>Mean NO₂: %{y:.4f} µmol/m²<extra></extra>",
    ))
    fig.update_layout(title=f"Seasonal Average NO₂ — {district}",
                      yaxis_title="NO₂ (µmol/m²)", **PLOTLY_LAYOUT)
    return fig

def line_chart_no2(df: pd.DataFrame, title: str = "NO₂ by Season") -> go.Figure:
    fig = go.Figure()
    season_order = ["DJF", "MAM", "JJA", "SON"]
    for district in DISTRICTS:
        d = (df[df["region_name"] == district]
             .groupby("season")["NO2_column_number_density"]
             .mean().reindex(season_order).reset_index())
        fig.add_trace(go.Scatter(
            x=d["season"],
            y=d["NO2_column_number_density"] * 1e6,
            name=district, mode="lines+markers",
            line=dict(color=DISTRICT_COLORS[district], width=2),
            hovertemplate="Season: %{x}<br>NO₂: %{y:.4f} µmol/m²<extra>" + district + "</extra>",
        ))
    fig.update_layout(title=title, yaxis_title="NO₂ (µmol/m²)",
                      xaxis_title="Season", **PLOTLY_LAYOUT)
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:16px 0 8px 0;'>
      <div style='font-size:36px;'>🌍</div>
      <div style='font-size:14px; font-weight:700; color:{COLORS["sky_blue"]};
                  letter-spacing:1px; margin-top:4px;'>AIR QUALITY<br>MONITORING</div>
      <div style='font-size:11px; color:{COLORS["soft_gray"]}; margin-top:2px;'>
        Istanbul • Sentinel-5P</div>
    </div>
    <hr style='border-color:{COLORS["sky_blue"]}44; margin:8px 0 16px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=["🏠  Home", "📍  District Analysis", "🤖  ML Predictions",
                 "📊  Analytics & Seasonal", "🏭  PM10 & PM2.5 (Future)"],
        label_visibility="collapsed",
    )

    st.markdown(f"<hr style='border-color:{COLORS['mid_blue']}; margin:16px 0;'>", unsafe_allow_html=True)

    # Model status badge
    if model_loaded:
        st.success("✅ Model loaded (OHE)")
    else:
        st.info("ℹ️ Demo mode — place best_rf_model_ohe.pkl here to enable live predictions")

    st.markdown("<br>", unsafe_allow_html=True)
    who_sidebar_legend()
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("🛰️ Data: Sentinel-5P / Google Earth Engine")
    st.caption("🤖 Model: Random Forest (.pkl)")
    st.caption("🎓 Graduation Project 2024-25")

# ─────────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────────
if page == "🏠  Home":
    st.markdown(f"""
    <h1 style='font-size:32px; font-weight:800; margin-bottom:4px;'>
      🌍 Air Quality Monitoring Dashboard
    </h1>
    <div style='color:{COLORS["soft_gray"]}; font-size:15px; margin-bottom:24px;'>
      NO₂ Prediction for Istanbul Districts using Machine Learning & Satellite Data (Sentinel-5P)
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

    # WHO district summary cards
    st.markdown("### 📡 Current NO₂ Levels")
    latest = get_latest_no2(df_all)
    c1, c2, c3 = st.columns(3)
    for col, district in zip([c1, c2, c3], DISTRICTS):
        val      = latest.get(district, 0)
        val_umol = no2_to_umol(val)
        lvl      = who_level(val_umol)
        col.markdown(f"""
        <div style='background:linear-gradient(135deg,{COLORS["mid_blue"]},{COLORS["dark_blue"]});
          border:2px solid {lvl["color"]}88; border-radius:14px; padding:18px; text-align:center;'>
          <div style='font-size:13px; color:{COLORS["soft_gray"]}; margin-bottom:4px;'>{district}</div>
          <div style='font-size:26px; font-weight:800; color:{COLORS["sky_blue"]};'>
            {val_umol:.5f}</div>
          <div style='font-size:12px; color:{COLORS["soft_gray"]}; margin-bottom:10px;'>µmol/m²</div>
          <div style='display:inline-block; background:{lvl["color"]}; color:#fff;
            font-size:12px; font-weight:700; padding:4px 12px; border-radius:20px;'>
            {lvl["tier"]}</div>
          <div style='font-size:10px; color:{COLORS["soft_gray"]}; margin-top:6px;'>
            {lvl["note"]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 NO₂ Trend by Season")
    fig_line = line_chart_no2(df_home, title="NO₂ by Season — All Districts")
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown(f"""
    <div class='info-card'>
      <b>About this project:</b> This dashboard visualizes NO₂ pollution levels for three Istanbul
      districts (Kartal, Kağıthane, Üsküdar) using satellite imagery from Sentinel-5P. A trained
      Random Forest model (R=0.98, R²=0.96) provides NO₂ predictions.
      Data is retrieved via Google Earth Engine.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 2 — DISTRICT ANALYSIS
# ─────────────────────────────────────────────
elif page == "📍  District Analysis":
    st.markdown("<h1>📍 District Analysis</h1>", unsafe_allow_html=True)

    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        sel_district = st.selectbox("Select District", DISTRICTS, key="dist_select")
    with col_d2:
        st.markdown("<div style='margin-top:28px;'>", unsafe_allow_html=True)
        compare_mode = st.toggle("Compare All Districts", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    df_dist = df_all[df_all["region_name"] == sel_district]

    # Map
    st.markdown("### 🗺️ District Location Map")
    map_districts = DISTRICTS if compare_mode else [sel_district]
    latest        = get_latest_no2(df_all)
    map_df = pd.DataFrame([
        {**DISTRICT_COORDS[d], "district": d,
         "NO2": round(no2_to_umol(latest.get(d, 0)), 5)}
        for d in map_districts
    ])
    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", hover_name="district",
        hover_data={"NO2": True, "lat": False, "lon": False},
        color="district",
        color_discrete_map={d: DISTRICT_COLORS[d] for d in DISTRICTS},
        size=[20] * len(map_df), zoom=10, height=320,
    )
    fig_map.update_layout(
        mapbox_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # WHO banner for selected district
    val_umol = no2_to_umol(latest.get(sel_district, 0))
    who_banner(sel_district, val_umol)

    # Charts
    st.markdown("### 📊 Detailed Charts")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        df_plot = df_all if compare_mode else df_dist
        fig_l   = line_chart_no2(df_plot, title=f"NO₂ by Season — {sel_district if not compare_mode else 'All Districts'}")
        st.plotly_chart(fig_l, use_container_width=True)
    with col_c2:
        fig_b = bar_chart_seasonal(df_all, sel_district)
        st.plotly_chart(fig_b, use_container_width=True)

    # Seasonal stats table
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

# ─────────────────────────────────────────────
# PAGE 3 — ML PREDICTIONS
# ─────────────────────────────────────────────
elif page == "🤖  ML Predictions":
    st.markdown("<h1>🤖 Machine Learning Predictions</h1>", unsafe_allow_html=True)

    if model_loaded:
        st.success("✅ Random Forest model loaded (OHE + Wavelet features d1/d2/d3 | R²=0.9762, MAE=1.27×10⁻⁶, RMSE=1.85×10⁻⁶ mol/m²)")
    else:
        st.info("ℹ️ Running in demo mode. Place `best_rf_model_ohe.pkl` in the app directory to enable live predictions.")

    # Controls
    col_p1, col_p2, col_p3 = st.columns([1.5, 1.5, 1])
    with col_p1:
        pred_district = st.selectbox("Select District", DISTRICTS, key="pred_dist")
    with col_p2:
        future_date = st.date_input(
            "Select Future Date",
            value=datetime.today() + timedelta(days=30),
            min_value=datetime.today().date(),
            key="pred_date",
        )
    with col_p3:
        st.markdown("<div style='margin-top:28px;'>", unsafe_allow_html=True)
        col_b1, col_b2 = st.columns(2)
        predict_clicked = col_b1.button("🔮 Predict", use_container_width=True)
        reset_clicked   = col_b2.button("🔄 Reset",   use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if reset_clicked:
        st.rerun()

    # BUG FIX: removed `or True` — block only runs when button is clicked
    if predict_clicked:
        month = future_date.month
        if month in [12, 1, 2]:   season_code = "DJF"
        elif month in [3, 4, 5]:  season_code = "MAM"
        elif month in [6, 7, 8]:  season_code = "JJA"
        else:                      season_code = "SON"

        # BUG FIX: real model prediction instead of hardcoded 0.000065
        prediction = run_prediction(pred_district, season_code)
        pred_umol  = no2_to_umol(prediction)
        lvl        = who_level(pred_umol)

        # Prediction card
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

        # WHO warning banner
        who_banner(pred_district, pred_umol)
        st.markdown("<br>", unsafe_allow_html=True)

        # Actual vs predicted chart (using real model on all seasons for this district)
        st.markdown("### 📈 Actual vs Predicted NO₂")
        df_dist_pred = df_all[df_all["region_name"] == pred_district].copy()

        if model_loaded:
            X = preprocessor.transform(
                df_dist_pred[["region_name", "season", "lat", "lon"]]
            )
            pred_vals = rf_model.predict(X)
        else:
            base          = {"Kartal": 0.000110, "Kağıthane": 0.000125, "Üsküdar": 0.000105}
            season_factor = {"DJF": 1.20, "MAM": 1.05, "JJA": 0.85, "SON": 1.00}
            pred_vals = np.array([
                base[pred_district] * season_factor[s]
                for s in df_dist_pred["season"]
            ])

        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=df_dist_pred["season"],
            y=df_dist_pred["NO2_column_number_density"] * 1e6,
            name="Actual", mode="markers+lines",
            line=dict(color=COLORS["sky_blue"], width=2),
        ))
        fig_avp.add_trace(go.Scatter(
            x=df_dist_pred["season"],
            y=pred_vals * 1e6,
            name="Predicted (RF)", mode="markers+lines",
            line=dict(color=COLORS["green"], width=2, dash="dash"),
        ))
        fig_avp.update_layout(
            title=f"Actual vs Predicted NO₂ — {pred_district}",
            yaxis_title="NO₂ (µmol/m²)", xaxis_title="Season", **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_avp, use_container_width=True)

        # Accuracy metrics (updated with wavelet model metrics)
        st.markdown("### 📐 Model Accuracy Metrics")
        m1, m2, m3, m4 = st.columns(4)
        for col, label, value, unit in [
            (m1, "R²",   0.9762, "Overall"),
            (m2, "MAE",  1.27,   "×10⁻⁶ mol/m²"),
            (m3, "RMSE", 1.85,   "×10⁻⁶ mol/m²"),
            (m4, "Features", 7,  "incl. d1/d2/d3"),
        ]:
            col.markdown(f"""
            <div class='accuracy-card'>
              <div style='font-size:13px; color:{COLORS["soft_gray"]};'>{label}</div>
              <div style='font-size:26px; font-weight:800; color:{COLORS["sky_blue"]};'>
                {value}
              </div>
              <div style='font-size:12px; color:{COLORS["soft_gray"]};'>{unit}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Seasonal R² breakdown chart
        st.markdown("### 📊 R² Score by Season (Test Set)")
        seasonal_r2 = {
            "DJF (Winter)": 0.6676,
            "MAM (Spring)": 0.9763,
            "JJA (Summer)": 0.9605,
            "SON (Autumn)": 0.9885,
        }
        seasonal_colors = ["#534AB7", "#1D9E75", "#EF9F27", "#D85A30"]
        fig_r2 = go.Figure(go.Bar(
            x=list(seasonal_r2.keys()),
            y=list(seasonal_r2.values()),
            marker_color=seasonal_colors,
            text=[f"{v:.4f}" for v in seasonal_r2.values()],
            textposition="outside",
            hovertemplate="Season: %{x}<br>R²: %{y:.4f}<extra></extra>",
        ))
        fig_r2.add_hline(y=0.95, line_dash="dash", line_color="#B0BEC5",
                         annotation_text="R²=0.95 target", annotation_position="top right")
        fig_r2.update_layout(
            title="Per-Season R² — Wavelet-Enhanced Model (n=300, max_depth=15)",
            yaxis_title="R²", yaxis_range=[0, 1.05],
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_r2, use_container_width=True)
        st.markdown(f"""
        <div class='info-card' style='font-size:12px;'>
          <b>Note:</b> DJF (Winter) R²=0.667 is lower than other seasons but improved by +0.06 vs baseline.
          JJA (Summer) regressed slightly (0.985→0.960) — wavelet features hurt Summer slightly.
          This is worth discussing with the professor.
        </div>
        """, unsafe_allow_html=True)

        # Download
        results_df = df_dist_pred[["region_name", "season"]].copy()
        results_df["actual_NO2_mol_m2"]    = df_dist_pred["NO2_column_number_density"].values
        results_df["predicted_NO2_mol_m2"] = pred_vals
        results_df["residual"]             = pred_vals - results_df["actual_NO2_mol_m2"]
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_bytes,
            file_name=f"NO2_predictions_{pred_district}_{future_date}.csv",
            mime="text/csv",
        )
    else:
        st.markdown(f"""
        <div class='info-card' style='margin-top:20px;'>
          Select a district and date above, then click <b>Predict</b> to run the model.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 4 — ANALYTICS & SEASONAL
# ─────────────────────────────────────────────
elif page == "📊  Analytics & Seasonal":
    st.markdown("<h1>📊 Analytics & Seasonal Trends</h1>", unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([2, 1])
    with col_a1:
        sel_season_label = st.selectbox("Select Season", SEASONS, key="analytics_season")
    with col_a2:
        st.markdown("<div style='margin-top:28px;'>", unsafe_allow_html=True)
        st.button("🔍 Filter Data", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    sel_season_code = SEASON_MAP[sel_season_label]
    df_season       = df_all[df_all["season"] == sel_season_code]

    # Seasonal comparison bar
    st.markdown("### 🌦️ Seasonal Comparison — All Districts")
    season_comp = (
        df_all.groupby(["season", "region_name"])["NO2_column_number_density"]
        .mean().reset_index()
    )
    season_comp["NO2_umol"] = season_comp["NO2_column_number_density"] * 1e6
    fig_bar = px.bar(
        season_comp, x="season", y="NO2_umol", color="region_name",
        barmode="group", color_discrete_map=DISTRICT_COLORS,
        labels={"NO2_umol": "Mean NO₂ (µmol/m²)", "season": "Season", "region_name": "District"},
        category_orders={"season": ["DJF", "MAM", "JJA", "SON"]},
    )
    fig_bar.update_layout(title="Mean NO₂ by Season & District", **PLOTLY_LAYOUT)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Pie chart
    st.markdown(f"### 🍩 Pollution Distribution — {sel_season_label}")
    pie_data = df_season.groupby("region_name")["NO2_column_number_density"].mean().reset_index()
    fig_pie  = px.pie(
        pie_data, values="NO2_column_number_density", names="region_name",
        color="region_name", color_discrete_map=DISTRICT_COLORS, hole=0.45,
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["white"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Stats table
    st.markdown(f"### 📋 Statistical Summary — {sel_season_label}")
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

# ─────────────────────────────────────────────
# PAGE 5 — PM10 & PM2.5 FUTURE INTEGRATION
# ─────────────────────────────────────────────
elif page == "🏭  PM10 & PM2.5 (Future)":
    st.markdown("<h1>🏭 PM10 & PM2.5 — Future Integration Plan</h1>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='info-card'>
      <b>⚠️ Why PM10 and PM2.5 are not in the current dataset</b><br><br>
      Sentinel-5P TROPOMI measures column-integrated atmospheric gases (NO₂, SO₂, CO, O₃, CH₄)
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
          It can be converted to surface PM2.5 using empirical regression models
          (e.g. Liu et al. 2004).<br><br>
          <code>ee.ImageCollection("MODIS/061/MOD08_D3")<br>
          .select("Aerosol_Optical_Depth_Land_Mean")</code>
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
          These can be summed to estimate total PM2.5 and PM10 without requiring
          ground station calibration.<br><br>
          <code>ee.ImageCollection("NASA/GSFC/MERRA/flx/2")<br>
          .select(["DUSMASS25","OCSMASS","BCSMASS"])</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🔄 Planned Pipeline")
    st.markdown(f"""
    <div style='background:{COLORS["mid_blue"]}; border-radius:12px; padding:20px;
      font-size:13px; color:{COLORS["white"]}; line-height:2.2;'>
      <b>Step 1 — Data extraction (Google Earth Engine)</b><br>
      &nbsp;&nbsp;→ Pull MODIS AOD and MERRA-2 PM fields for Kartal, Kağıthane, Üsküdar<br>
      &nbsp;&nbsp;→ Filter 2024–2025, apply QA masks, export to CSV<br><br>
      <b>Step 2 — PM estimation</b><br>
      &nbsp;&nbsp;→ PM2.5 = DUSMASS25 + OCSMASS + BCSMASS + SSSMASS25 (MERRA-2)<br>
      &nbsp;&nbsp;→ PM10  = PM2.5 + coarse fraction (DUSMASS – DUSMASS25)<br><br>
      <b>Step 3 — Feature engineering</b><br>
      &nbsp;&nbsp;→ Same structure as NO₂: region_name + season (one-hot) + lat + lon<br>
      &nbsp;&nbsp;→ Add AOD as additional predictor feature<br><br>
      <b>Step 4 — Model training</b><br>
      &nbsp;&nbsp;→ Separate Random Forest regressors for PM2.5 and PM10<br>
      &nbsp;&nbsp;→ Evaluate with MAE and RMSE, compare against WHO thresholds<br><br>
      <b>Step 5 — Dashboard integration</b><br>
      &nbsp;&nbsp;→ New tabs for PM2.5 and PM10 alongside existing NO₂ pages
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📏 WHO PM Thresholds (2021 Guidelines)")
    pm_thresh = pd.DataFrame({
        "Pollutant":          ["PM2.5", "PM2.5", "PM10", "PM10"],
        "Averaging period":   ["Annual mean", "24-hour mean", "Annual mean", "24-hour mean"],
        "WHO Guideline":      ["5 µg/m³",  "15 µg/m³", "15 µg/m³", "45 µg/m³"],
        "Interim Target 1":   ["35 µg/m³", "75 µg/m³", "70 µg/m³", "150 µg/m³"],
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
