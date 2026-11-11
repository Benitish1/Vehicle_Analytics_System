
import pandas as pd
import plotly.graph_objects as go
import json
import os
import plotly.express as px

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
    """Create an interactive Rwanda district choropleth without external map tiles."""

    GEOJSON_PATH = os.path.join(BASE_DIR, "Rwanda_Districts.geojson")

    try:
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        # Extract district names from GeoJSON
        district_names = [
            feature["properties"]["shapeName"]
            for feature in geojson["features"]
        ]

        if not district_names:
            return "<div class='alert alert-warning'>No district boundaries found.</div>"

        # Count vehicle clients per district
        counts = (
            df["district"]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
        )

        # Create a row for every district, including districts with zero clients
        map_df = pd.DataFrame({"district": district_names})
        map_df["clients"] = (
            map_df["district"].map(counts).fillna(0).astype(int)
        )

        # Identify unmatched dataset district names
        unmatched = set(counts.index) - set(district_names)
        if unmatched:
            print("District names not matched in GeoJSON:", unmatched)

        # Build interactive choropleth
        fig = px.choropleth(
            map_df,
            geojson=geojson,
            locations="district",
            featureidkey="properties.shapeName",
            color="clients",
            color_continuous_scale="Purples",
            hover_name="district",
            hover_data={
                "district": False,
                "clients": ":,d",
            },
            labels={
                "clients": "Vehicle Clients",
            },
            title="Rwanda — Vehicle Clients per District",
        )

        # Display district boundaries without external map tiles
        fig.update_geos(
            fitbounds="locations",
            visible=False,
            showcountries=True,
            showcoastlines=False,
            showland=True,
            landcolor="#f8f9fa",
            bgcolor="#f8f9fa",
        )

        fig.update_layout(
            height=780,
            margin=dict(l=0, r=0, t=60, b=0),
            paper_bgcolor="#f8f9fa",
            title=dict(
                font=dict(size=18, color="#1a1a2e"),
                x=0.5,
            ),
            coloraxis_colorbar=dict(
                title="Vehicle Clients",
            ),
        )

        return fig.to_html(
            full_html=False,
            include_plotlyjs="cdn"
        )

    except Exception as e:
        print(f"Error generating Rwanda district map: {e}")
        return (
            "<div class='alert alert-danger'>"
            "Unable to generate the Rwanda district map."
            "</div>"
        )


# Existing functions

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
