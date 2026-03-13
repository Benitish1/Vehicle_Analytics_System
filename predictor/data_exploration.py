import pandas as pd
import plotly.graph_objects as go
import json
import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, "dummy-data", "rwanda_districts.json")

def load_district_data():
    """Load district coordinates and province mapping from JSON."""
    try:
        with open(JSON_PATH, "r") as f:
            return json.load(f)["districts"]
    except Exception as e:
        print(f"Error loading district JSON: {e}")
        return {}

def district_map_chart(df: pd.DataFrame) -> str:
    """Return a Plotly bubble-map HTML div showing vehicle-client counts per district."""
    counts = df["district"].value_counts().reset_index()
    counts.columns = ["district", "clients"]

    districts_data = load_district_data()
    
    # Enrich with coordinates
    rows = []
    for _, row in counts.iterrows():
        d = row["district"]
        if d in districts_data:
            data = districts_data[d]
            rows.append({
                "district": d,
                "clients":  row["clients"],
                "lat":      data["lat"],
                "lon":      data.get("lon") or data.get("8.9328") or data.get("29.7789"), # Handle possible typos in manual JSON
                "colour":   data["color"],
                "province": data["province"],
            })
    
    if not rows:
        return "<div class='alert alert-warning'>No district data found for map.</div>"
        
    map_df = pd.DataFrame(rows)

    fig = go.Figure()

    # One trace per province for a clean legend
    for prov, grp in map_df.groupby("province"):
        fig.add_trace(go.Scattermapbox(
    lat=grp["lat"],
    lon=grp["lon"],
    mode="text",
    marker=dict(
        size=1,
        opacity=0
    ),
    text=grp["district"],
    textposition="top center",
    customdata=list(zip(grp["clients"], [prov] * len(grp))),
    hovertemplate=(
        "<b>%{text}</b><br>"
        "Province: %{customdata[1]}<br>"
        "Vehicle Clients: %{customdata[0]}<extra></extra>"
    ),
    name=f"{prov} Province",
))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=-1.9403, lon=29.8739),
            zoom=7.2,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=780,
        title=dict(
            text="Rwanda — Vehicle Clients per District",
            font=dict(size=16, color="#1a1a2e"),
            x=0.5,
        ),
        legend=dict(
            orientation="v",
            x=1.0, y=1.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
        paper_bgcolor="#f8f9fa",
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


# existing functions

def dataset_exploration(df):
    return df.head().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
        index=False,
    )


def data_exploration(df):
    return df.describe().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
    )
