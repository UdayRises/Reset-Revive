# SpringRevive AI

**SIH26240 · AI-Based Spring Revival and Recharge Planning for Tribal Areas**  
Ministry of Tribal Affairs · Preliminary decision-support prototype

---

## Overview

SpringRevive AI is a Streamlit web application that helps field teams screen spring sites in tribal areas for groundwater recharge potential using a Random Forest classifier trained on domain-informed synthetic data.

> **Disclaimer:** Scores are preliminary screening indicators, not validated groundwater predictions. All identified sites require field hydrogeological assessment before any intervention.

## Features

- 🗺️ **Interactive Map** — Spring markers with suitability colour-coding; satellite, terrain, and topographic basemap options
- 🔍 **Spring Assessment** — Per-site recharge probability, component scores, risk flags, and intervention recommendations
- 📊 **Compare & Rank** — Side-by-side suitability comparison and scatter analysis
- 🧪 **Model & Method** — Transparent feature importances and scoring formula
- 📥 **Field Validation & Export** — Log field observations, export CSV / GeoJSON / interactive HTML map

## Tech Stack

- Python 3.x
- [Streamlit](https://streamlit.io/)
- [Folium](https://python-visualization.github.io/folium/) + streamlit-folium
- scikit-learn (Random Forest)
- Plotly, GeoPandas, Pandas, NumPy

## Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## CSV Input Format

Upload a CSV with these columns:

| Column | Description |
|---|---|
| `spring_id` | Unique identifier |
| `village` | Village name |
| `latitude` | Decimal degrees (-90 to 90) |
| `longitude` | Decimal degrees (-180 to 180) |
| `rainfall_mm` | Annual rainfall (mm) |
| `slope_deg` | Slope in degrees (0–90) |
| `infiltration_mm_hr` | Infiltration rate (mm/hr) |
| `geology` | One of: Fractured, Moderate, Low permeability, Unknown |
| `forest_cover_pct` | Forest cover percentage (0–100) |
| `flow_decline_pct` | Flow decline percentage (0–100) |

## License

For academic and government demonstration purposes only.
