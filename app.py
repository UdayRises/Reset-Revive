"""
SpringRevive AI — SIH26240
AI-Based Spring Revival and Recharge Planning for Tribal Areas
Ministry of Tribal Affairs

Run:
    python -m streamlit run app.py

Honest scope:
- Random Forest classifier trained on synthetic samples derived from
  transparent domain rules. NOT trained on real groundwater data.
- Recharge probability is an indicative screening output.
- All interventions require field hydrogeological assessment.
"""

import datetime
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap, MarkerCluster
import geopandas as gpd
import plotly.express as px
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SpringRevive AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CONSTANTS
# =========================================================
REQUIRED_COLUMNS = [
    "spring_id", "village", "latitude", "longitude",
    "rainfall_mm", "slope_deg", "infiltration_mm_hr",
    "geology", "forest_cover_pct", "flow_decline_pct",
]

NUMERIC_COLUMNS = [
    "latitude", "longitude", "rainfall_mm", "slope_deg",
    "infiltration_mm_hr", "forest_cover_pct", "flow_decline_pct",
]

GEOLOGY_SCORES = {
    "Fractured": 100,
    "Moderate": 70,
    "Low permeability": 35,
    "Unknown": 50,
}
GEOLOGY_TYPES = list(GEOLOGY_SCORES.keys())

FEATURES = [
    "rainfall_score", "infiltration_score", "slope_score",
    "geology_score", "forest_score", "flow_decline_pct",
]

EXPORT_COLUMNS = [
    "spring_id", "village", "latitude", "longitude",
    "rainfall_mm", "slope_deg", "infiltration_mm_hr",
    "geology", "forest_cover_pct", "flow_decline_pct",
    "rainfall_score", "infiltration_score", "slope_score",
    "geology_score", "forest_score",
    "recharge_probability", "confidence", "suitability_score",
    "suitability_class", "priority", "risk_flags", "primary_intervention",
]

COLOR_MAP = {"High": "#2E8B57", "Medium": "#E69F00", "Low": "#D9534F"}

DEMO_DATA = [
    ["SPR-001", "Village A", 30.0668, 79.0193, 1550, 12, 35, "Fractured", 72, 45],
    ["SPR-002", "Village B", 30.0850, 79.0450, 1250, 28, 18, "Low permeability", 35, 70],
    ["SPR-003", "Village C", 30.0450, 79.0500, 1750,  8, 42, "Fractured", 85, 25],
    ["SPR-004", "Village D", 30.1000, 78.9900, 1100, 32, 12, "Unknown", 28, 80],
    ["SPR-005", "Village E", 30.0300, 78.9800, 1450, 17, 30, "Moderate", 60, 55],
    ["SPR-006", "Village F", 30.1150, 79.0650, 1650, 22, 25, "Moderate", 68, 35],
    ["SPR-007", "Village G", 30.0200, 79.0300, 1350,  5, 45, "Fractured", 78, 40],
    ["SPR-008", "Village H", 30.0750, 79.0800, 1000, 38, 10, "Low permeability", 22, 90],
]

# =========================================================
# DATA LOADING & VALIDATION
# =========================================================
def load_data(uploaded_file=None):
    if uploaded_file is None:
        df = pd.DataFrame(DEMO_DATA, columns=REQUIRED_COLUMNS)
        source = "demo"
    else:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            raise ValueError(f"Could not read CSV: {e}")
        source = "uploaded"

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(missing))

    df = df[REQUIRED_COLUMNS].copy()

    if df.empty:
        raise ValueError("CSV contains no records.")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[NUMERIC_COLUMNS].isna().any().any():
        raise ValueError("Numeric columns contain missing or invalid values.")

    for col in ["spring_id", "village"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
        if (df[col] == "").any():
            raise ValueError(f"{col} cannot be blank.")

    if df["spring_id"].duplicated().any():
        raise ValueError("Every spring_id must be unique.")

    if not df["latitude"].between(-90, 90).all():
        raise ValueError("Latitude must be between -90 and 90.")

    if not df["longitude"].between(-180, 180).all():
        raise ValueError("Longitude must be between -180 and 180.")

    if (df["rainfall_mm"] < 0).any():
        raise ValueError("Rainfall cannot be negative.")

    if not df["slope_deg"].between(0, 90).all():
        raise ValueError("Slope must be between 0 and 90 degrees.")

    if (df["infiltration_mm_hr"] < 0).any():
        raise ValueError("Infiltration cannot be negative.")

    for col in ["forest_cover_pct", "flow_decline_pct"]:
        if not df[col].between(0, 100).all():
            raise ValueError(f"{col} must be between 0 and 100.")

    df["geology"] = df["geology"].fillna("Unknown").astype(str).str.strip()
    df.loc[df["geology"] == "", "geology"] = "Unknown"
    df.loc[~df["geology"].isin(GEOLOGY_TYPES), "geology"] = "Unknown"

    return df, source


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def slope_score(slope):
    if 5 <= slope <= 15:
        return 100
    if 15 < slope <= 25:
        return 75
    if 2 <= slope < 5:
        return 70
    if 25 < slope <= 35:
        return 40
    return 20


def compute_component_scores(df):
    result = df.copy()
    result["rainfall_score"] = (result["rainfall_mm"] / 1800.0 * 100).clip(0, 100)
    result["infiltration_score"] = (result["infiltration_mm_hr"] / 50.0 * 100).clip(0, 100)
    result["slope_score"] = result["slope_deg"].apply(slope_score)
    result["geology_score"] = result["geology"].map(GEOLOGY_SCORES).fillna(50).astype(float)
    result["forest_score"] = result["forest_cover_pct"].clip(0, 100).astype(float)
    return result


def compute_confidence(df):
    n = len(df)
    conf = pd.Series(0.55, index=df.index)
    conf += min(0.15, n * 0.01)
    conf += np.where(df["geology"] == "Unknown", -0.15, 0.05)
    conf -= np.where(df["slope_deg"] > 35, 0.10, 0)
    conf -= np.where(df["infiltration_mm_hr"] < 5, 0.10, 0)
    return conf.clip(0.30, 0.95).round(2)


# =========================================================
# ML MODEL
# =========================================================
@st.cache_resource(show_spinner=False)
def train_recharge_model():
    rng = np.random.default_rng(42)
    n = 3000

    syn = pd.DataFrame({
        "rainfall_mm": rng.uniform(400, 2500, n),
        "slope_deg": rng.uniform(0, 45, n),
        "infiltration_mm_hr": rng.uniform(0, 60, n),
        "forest_cover_pct": rng.uniform(0, 100, n),
        "flow_decline_pct": rng.uniform(0, 100, n),
        "geology": rng.choice(GEOLOGY_TYPES, n),
    })
    syn = compute_component_scores(syn)

    X = syn[FEATURES].values

    score = (
        0.30 * syn["rainfall_score"]
        + 0.30 * syn["infiltration_score"]
        + 0.15 * syn["slope_score"]
        + 0.15 * syn["geology_score"]
        + 0.10 * syn["forest_score"]
        - 0.20 * syn["flow_decline_pct"]
    )
    score = score + rng.normal(0, 5, n)
    y = (score >= 55).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_leaf=4, random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_importances": dict(zip(FEATURES, clf.feature_importances_.round(4))),
    }
    return clf, metrics


# =========================================================
# SUITABILITY / PRIORITY / RISK / INTERVENTION
# =========================================================
def classify_suitability(prob):
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def classify_priority(row):
    if row["flow_decline_pct"] >= 60 and row["recharge_probability"] >= 0.5:
        return "Urgent field assessment"
    if row["flow_decline_pct"] >= 35:
        return "Monitor closely"
    if row["recharge_probability"] >= 0.70:
        return "High-potential site"
    return "Routine monitoring"


def compute_risk_flags(row):
    flags = []
    if row["slope_deg"] > 45:
        flags.append("Very high landslide risk")
    elif row["slope_deg"] > 30:
        flags.append("Steep slope (>30°)")
    if row["infiltration_mm_hr"] < 8:
        flags.append("Very low infiltration")
    if row["flow_decline_pct"] >= 80:
        flags.append("Severe flow decline")
    if row["forest_cover_pct"] < 20:
        flags.append("Sparse vegetation")
    if row["geology"] == "Low permeability":
        flags.append("Low permeability geology")
    if row["geology"] == "Unknown":
        flags.append("Unknown geology")
    return flags if flags else ["None"]


def recommend_intervention(row):
    slope = row["slope_deg"]
    infil = row["infiltration_mm_hr"]
    geo = row["geology"]
    fc = row["forest_cover_pct"]
    decl = row["flow_decline_pct"]

    if slope > 30:
        primary = "Contour trench + vegetative barrier (avoid excavation on steep slope)"
    elif infil >= 25 and geo == "Fractured":
        primary = "Recharge pit / injection well near fracture zone"
    elif infil >= 15:
        primary = "Check dam / percolation pond"
    elif fc < 30:
        primary = "Afforestation + contour bunding"
    elif decl >= 60:
        primary = "Spring-box + recharge trench network"
    else:
        primary = "Percolation tank / farm pond"

    secondary = []
    if fc < 40:
        secondary.append("Vegetation restoration")
    if decl >= 60:
        secondary.append("Urgent hydrological survey")
    if geo == "Unknown":
        secondary.append("Detailed geological mapping")
    if geo == "Low permeability":
        secondary.append("Subsurface investigation before construction")
    if slope > 35:
        secondary.append("Geotechnical stability check")
    if not secondary:
        secondary = ["Standard monitoring"]

    return primary, secondary


def analyze_data(df, clf):
    result = compute_component_scores(df)

    probs = clf.predict_proba(result[FEATURES].values)[:, 1]
    result["recharge_probability"] = probs.round(3)

    result["suitability_class"] = result["recharge_probability"].apply(classify_suitability)

    rule_score = (
        0.30 * result["rainfall_score"]
        + 0.30 * result["infiltration_score"]
        + 0.15 * result["slope_score"]
        + 0.15 * result["geology_score"]
        + 0.10 * result["forest_score"]
    )
    blended = 0.6 * (result["recharge_probability"] * 100) + 0.4 * rule_score
    result["suitability_score"] = blended.clip(0, 100).round(1)

    result["confidence"] = compute_confidence(result)
    result["priority"] = result.apply(classify_priority, axis=1)
    result["risk_flags"] = result.apply(lambda r: "; ".join(compute_risk_flags(r)), axis=1)
    result["primary_intervention"] = result.apply(
        lambda r: recommend_intervention(r)[0], axis=1
    )
    return result



# =========================================================
# MAP TILE CONFIGURATIONS
# =========================================================
# Every entry carries an explicit tile URL + attribution so that
# Folium never raises "Custom tiles must have an attribution".
MAP_TILE_CONFIGS = {
    "OpenStreetMap": {
        "tiles": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": "Map data: OpenStreetMap contributors (ODbL)",
    },
    "Satellite": {
        "tiles": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "attr": (
            "Tiles Esri - Source: Esri, Maxar, Earthstar Geographics, "
            "CNES/Airbus DS, USDA FSA, USGS, AeroGRID, IGN, GIS User Community"
        ),
    },
    "Terrain": {
        "tiles": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        "attr": "Tiles Esri - Esri, HERE, Garmin, FAO, NOAA, USGS",
    },
    "Topographic": {
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": (
            "Map data: OpenStreetMap contributors, SRTM | "
            "Map style: OpenTopoMap (CC-BY-SA)"
        ),
    },
}


# =========================================================
# MAP BUILDER
# =========================================================
def build_map(df, show_heatmap=False, show_confidence=False,
              extra_points=None, map_type="OpenStreetMap"):
    """Build a Folium map with selectable basemap types.

    The selected map_type becomes the active basemap.
    All other types are added as toggleable layers inside LayerControl.
    """
    center = [float(df["latitude"].mean()), float(df["longitude"].mean())]

    # Start with tiles=None so we control every layer explicitly
    m = folium.Map(location=center, zoom_start=11, tiles=None)

    # Active basemap
    cfg = MAP_TILE_CONFIGS.get(map_type, MAP_TILE_CONFIGS["OpenStreetMap"])
    folium.TileLayer(
        tiles=cfg["tiles"],
        attr=cfg["attr"],
        name=map_type,
    ).add_to(m)

    # Alternative basemaps (all have attribution ? no ValueError)
    for label, tcfg in MAP_TILE_CONFIGS.items():
        if label == map_type:
            continue
        folium.TileLayer(
            tiles=tcfg["tiles"],
            attr=tcfg["attr"],
            name=label,
        ).add_to(m)

    cluster = MarkerCluster(name="Springs").add_to(m)
    for _, row in df.iterrows():
        color = COLOR_MAP.get(row["suitability_class"], "#888888")

        if show_confidence:
            folium.Circle(
                location=[row["latitude"], row["longitude"]],
                radius=150 + 300 * float(row["confidence"]),
                color=color, fill=True, fill_opacity=0.12, weight=0,
            ).add_to(m)

        popup_html = (
            f"<b>Spring:</b> {row['spring_id']}<br>"
            f"<b>Village:</b> {row['village']}<br>"
            f"<b>Recharge probability:</b> {row['recharge_probability']:.2f}<br>"
            f"<b>Suitability:</b> {row['suitability_class']} "
            f"(score {row['suitability_score']})<br>"
            f"<b>Confidence:</b> {row['confidence']:.2f}<br>"
            f"<b>Flow decline:</b> {row['flow_decline_pct']}%<br>"
            f"<b>Priority:</b> {row['priority']}<br>"
            f"<b>Risk flags:</b> {row['risk_flags']}<br>"
            f"<b>Primary intervention:</b> {row['primary_intervention']}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=9, color=color, fill=True,
            fill_color=color, fill_opacity=0.85,
            tooltip=f"{row['spring_id']} · {row['suitability_class']}",
            popup=folium.Popup(popup_html, max_width=380),
        ).add_to(cluster)

    if show_heatmap:
        heat_data = [
            [row["latitude"], row["longitude"], float(row["recharge_probability"])]
            for _, row in df.iterrows()
        ]
        HeatMap(heat_data, radius=28, blur=18, name="Recharge probability").add_to(m)

    if extra_points:
        fg = folium.FeatureGroup(name="Field validation points")
        for p in extra_points:
            folium.Marker(
                location=[p["lat"], p["lon"]],
                tooltip=p.get("label", "Field point"),
                icon=folium.Icon(color="purple", icon="flag"),
            ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Controls")

st.sidebar.header("1. Data")
uploaded_file = st.sidebar.file_uploader("Upload spring CSV", type=["csv"])
if uploaded_file is None:
    st.sidebar.info("Using illustrative demo data (8 springs).")

st.sidebar.header("2. Map layers")
map_type = st.sidebar.selectbox(
    "Map type",
    options=list(MAP_TILE_CONFIGS.keys()),
    index=0,
    help="OpenStreetMap: street map. Satellite: aerial imagery. Terrain: shaded relief. Topographic: contour lines.",
)
show_heatmap = st.sidebar.checkbox("Recharge probability heatmap", value=False)
show_confidence = st.sidebar.checkbox("Confidence halos", value=False)

st.sidebar.header("3. Model info")
st.sidebar.caption(
    "Random Forest (200 trees, depth 10) trained on 3,000 synthetic samples "
    "derived from transparent domain rules. Not trained on real groundwater data."
)

# =========================================================
# LOAD & ANALYZE
# =========================================================
try:
    raw_df, source = load_data(uploaded_file)
    clf, metrics = train_recharge_model()
    df = analyze_data(raw_df, clf)
except Exception as exc:
    st.error(f"Data processing error: {exc}")
    st.info("Fix the CSV columns/values and re-upload.")
    st.stop()

# =========================================================
# HEADER
# =========================================================
st.title("💧 SpringRevive AI")
st.subheader("AI-Based Spring Revival and Recharge Planning for Tribal Areas")
st.caption("SIH26240 · Ministry of Tribal Affairs · Preliminary decision-support prototype")

st.warning(
    "**DEMO MODE:** Default records are illustrative, not real field measurements. "
    "Scores are preliminary screening indicators, not validated groundwater predictions. "
    "All identified sites require field hydrogeological assessment before intervention."
)

# =========================================================
# SUMMARY METRICS
# =========================================================
a, b, c, d, e = st.columns(5)
a.metric("Springs", len(df))
b.metric("High suitability", int((df["suitability_class"] == "High").sum()))
c.metric("Medium suitability", int((df["suitability_class"] == "Medium").sum()))
d.metric("Low suitability", int((df["suitability_class"] == "Low").sum()))
e.metric(
    "Urgent assessments",
    int((df["priority"] == "Urgent field assessment").sum()),
)

st.divider()

# =========================================================
# TABS
# =========================================================
tab_map, tab_site, tab_rank, tab_model, tab_field = st.tabs([
    "🗺️ Map & Overview",
    "🔍 Spring Assessment",
    "📊 Compare & Rank",
    "🧪 Model & Method",
    "📥 Field Validation & Export",
])

# ---------- TAB 1: MAP & OVERVIEW ----------
with tab_map:
    st.header("Interactive spring map")
    st.caption(
        "Green = High · Orange = Medium · Red = Low preliminary suitability. "
        "Online basemap tiles require internet."
    )

    m = build_map(df, show_heatmap=show_heatmap, show_confidence=show_confidence, map_type=map_type)
    st_folium(m, width="stretch", height=520, key="main_map")

    st.subheader("Summary table")
    show_cols = [
        "spring_id", "village", "suitability_class", "suitability_score",
        "recharge_probability", "confidence", "priority",
        "flow_decline_pct", "primary_intervention",
    ]
    st.dataframe(
        df[show_cols].sort_values("suitability_score", ascending=False),
        width="stretch", hide_index=True,
    )

    st.subheader("Suitability distribution")
    fig = px.histogram(
        df, x="suitability_class", color="suitability_class",
        color_discrete_map=COLOR_MAP,
        category_orders={"suitability_class": ["Low", "Medium", "High"]},
        title="Springs by suitability class",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

# ---------- TAB 2: SPRING ASSESSMENT ----------
with tab_site:
    st.header("Spring assessment")

    selected_id = st.selectbox("Select a spring", df["spring_id"].tolist())
    site = df.loc[df["spring_id"] == selected_id].iloc[0]

    x, y, z = st.columns(3)
    x.metric("Recharge probability", f"{site['recharge_probability']:.2f}")
    y.metric("Suitability", f"{site['suitability_class']} ({site['suitability_score']})")
    z.metric("Confidence", f"{site['confidence']:.2f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Site information")
        st.write(f"**Spring ID:** {site['spring_id']}")
        st.write(f"**Village:** {site['village']}")
        st.write(f"**Latitude:** {site['latitude']}")
        st.write(f"**Longitude:** {site['longitude']}")
        st.write(f"**Rainfall:** {site['rainfall_mm']} mm/year")
        st.write(f"**Slope:** {site['slope_deg']}°")
        st.write(f"**Infiltration:** {site['infiltration_mm_hr']} mm/hr")
        st.write(f"**Geology:** {site['geology']}")
        st.write(f"**Forest cover:** {site['forest_cover_pct']}%")
        st.write(f"**Flow decline:** {site['flow_decline_pct']}%")
        st.write(f"**Priority:** {site['priority']}")
        st.write(f"**Risk flags:** {site['risk_flags']}")

    with right:
        st.subheader("Component scores")
        comp = pd.DataFrame({
            "Indicator": ["Rainfall", "Infiltration", "Slope", "Geology", "Forest"],
            "Score": [
                site["rainfall_score"], site["infiltration_score"],
                site["slope_score"], site["geology_score"], site["forest_score"],
            ],
        })
        chart = px.bar(
            comp, x="Indicator", y="Score", range_y=[0, 100],
            color="Score", color_continuous_scale="YlGn",
            title="Indicator scores (0-100)",
        )
        st.plotly_chart(chart, width="stretch")

    st.subheader("Recommendations")
    primary, secondary = recommend_intervention(site)
    st.success(f"**Primary intervention:** {primary}")
    if secondary:
        st.markdown("**Supporting actions:**")
        for s in secondary:
            st.markdown(f"- {s}")

    if site["risk_flags"] != "None":
        st.error(f"⚠️ Risk flags: {site['risk_flags']}")

# ---------- TAB 3: COMPARE & RANK ----------
with tab_rank:
    st.header("Compare spring locations")

    ranked = df.sort_values("suitability_score", ascending=False)
    ranking_chart = px.bar(
        ranked, x="spring_id", y="suitability_score",
        color="suitability_class", color_discrete_map=COLOR_MAP,
        range_y=[0, 100], title="Preliminary suitability comparison",
    )
    st.plotly_chart(ranking_chart, width="stretch")

    st.subheader("Recharge probability vs flow decline")
    scatter = px.scatter(
        df, x="flow_decline_pct", y="recharge_probability",
        size="confidence", color="suitability_class",
        color_discrete_map=COLOR_MAP,
        hover_data=["spring_id", "village"],
        title="Recharge probability vs reported flow decline",
    )
    scatter.update_yaxes(range=[0, 1])
    st.plotly_chart(scatter, width="stretch")

# ---------- TAB 4: MODEL & METHOD ----------
with tab_model:
    st.header("Model & method")

    col1, col2, col3 = st.columns(3)
    col1.metric("Test accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    col2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    col3.metric("Training samples", f"{metrics['n_train']:,}")

    st.markdown(
        """
        **Pipeline**

        1. **Input validation** — required columns, numeric ranges, uniqueness.
        2. **Feature engineering** — rainfall, infiltration, slope, geology and forest
           are normalised to 0–100 component scores.
        3. **AI model** — a Random Forest classifier (200 trees, max depth 10) is
           trained on 3,000 synthetic samples whose labels follow a transparent
           domain-informed scoring function. It is **not** trained on real
           groundwater measurements.
        4. **Blending** — final suitability score blends the ML probability (60%)
           with the explicit rule-based score (40%) for explainability.
        5. **Confidence** — estimated from sample size, geology certainty and
           presence of extreme values.
        6. **Risk flags and interventions** — rule-based and clearly indicative.
        """
    )

    st.subheader("Feature importance")
    fi = pd.DataFrame(
        list(metrics["feature_importances"].items()),
        columns=["Feature", "Importance"],
    ).sort_values("Importance", ascending=True)

    fig = px.bar(
        fi, x="Importance", y="Feature", orientation="h",
        title="Random Forest feature importance",
        color="Importance", color_continuous_scale="YlGn",
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Scoring formula (rule-based component)"):
        st.markdown(
            """
            - **Rainfall score** = `rainfall / 1800 × 100`, capped at 100.
            - **Infiltration score** = `infiltration / 50 × 100`, capped at 100.
            - **Slope score** — piece-wise: 5–15° → 100; 15–25° → 75;
              2–5° → 70; 25–35° → 40; else 20.
            - **Geology score** — Fractured 100, Moderate 70, Low permeability 35,
              Unknown 50.
            - **Forest score** — equal to forest-cover percentage.
            - **Rule-based total** = 0.30·rainfall + 0.30·infiltration
              + 0.15·slope + 0.15·geology + 0.10·forest.

            Final score = 0.6 × (ML probability × 100) + 0.4 × rule-based total.
            """
        )

# ---------- TAB 5: FIELD VALIDATION & EXPORT ----------
with tab_field:
    st.header("Field validation")

    if "field_observations" not in st.session_state:
        st.session_state.field_observations = []

    with st.form("field_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        fid = c1.text_input("Field ID / Spring ID")
        flat = c2.number_input(
            "Latitude", value=float(df["latitude"].mean()), format="%.6f"
        )
        flon = c3.number_input(
            "Longitude", value=float(df["longitude"].mean()), format="%.6f"
        )

        c4, c5, c6 = st.columns(3)
        fdisc = c4.number_input("Measured discharge (L/s)", min_value=0.0, value=1.0)
        fnotes = c5.text_input("Observations / community input")
        fphoto = c6.text_input("Photo reference (URL or ID)")

        submitted = st.form_submit_button("Add field observation")
        if submitted:
            st.session_state.field_observations.append({
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "field_id": fid,
                "lat": flat,
                "lon": flon,
                "discharge_lps": fdisc,
                "notes": fnotes,
                "photo_ref": fphoto,
            })
            st.success(f"Observation for {fid or 'unnamed point'} saved.")

    if st.session_state.field_observations:
        obs_df = pd.DataFrame(st.session_state.field_observations)
        st.subheader("Recorded observations")
        st.dataframe(obs_df, width="stretch", hide_index=True)

        st.download_button(
            "Download field observations (CSV)",
            data=obs_df.to_csv(index=False).encode("utf-8"),
            file_name="field_observations.csv",
            mime="text/csv",
        )

        extra_points = [
            {"lat": o["lat"], "lon": o["lon"], "label": o.get("field_id", "Field")}
            for o in st.session_state.field_observations
        ]
        st.subheader("Field points on map")
        m2 = build_map(df, extra_points=extra_points, map_type=map_type)
        st_folium(m2, width="stretch", height=420, key="field_map")
    else:
        st.info("No field observations recorded yet.")

    st.divider()
    st.header("Download results")

    export_df = df[EXPORT_COLUMNS].copy()
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    try:
        gdf = gpd.GeoDataFrame(
            export_df.copy(),
            geometry=gpd.points_from_xy(export_df["longitude"], export_df["latitude"]),
            crs="EPSG:4326",
        )
        geojson_bytes = gdf.to_json(drop_id=True).encode("utf-8")
    except Exception as e:
        geojson_bytes = b"{}"
        st.warning(f"GeoJSON export issue: {e}")

    map_main = build_map(df, show_heatmap=show_heatmap, show_confidence=show_confidence, map_type=map_type)
    map_html_bytes = map_main.get_root().render().encode("utf-8")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download CSV", data=csv_bytes,
            file_name="springrevive_report.csv",
            mime="text/csv", width="stretch",
        )
    with col2:
        st.download_button(
            "Download GeoJSON", data=geojson_bytes,
            file_name="springrevive_locations.geojson",
            mime="application/geo+json", width="stretch",
        )
    with col3:
        st.download_button(
            "Download interactive map (HTML)", data=map_html_bytes,
            file_name="springrevive_map.html",
            mime="text/html", width="stretch",
        )

    st.subheader("Full results table")
    st.dataframe(export_df, width="stretch", hide_index=True)

st.caption(
    "SpringRevive AI · SIH26240 · Verify all real-world inputs and "
    "recommendations with field measurements and qualified experts."
)