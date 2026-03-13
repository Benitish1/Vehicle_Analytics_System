import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

SEGMENT_FEATURES = ["estimated_income", "selling_price"]
CORE_SAMPLE_PERCENTILE = 0.25

df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

# ---------------------------
# STEP 1: Reduce data skew
# ---------------------------
df["estimated_income"] = np.log1p(df["estimated_income"])
df["selling_price"] = np.log1p(df["selling_price"])

# ---------------------------
# STEP 2: Feature Scaling
# ---------------------------
scaler = StandardScaler()
X = scaler.fit_transform(df[SEGMENT_FEATURES])

# Save scaler
joblib.dump(scaler, "model_generators/clustering/scaler.pkl")

# ---------------------------
# STEP 3: Train KMeans
# ---------------------------
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=100,
    init="k-means++"
)

df["cluster_id"] = kmeans.fit_predict(X)

centers = kmeans.cluster_centers_

# ---------------------------
# STEP 4: Map clusters to client classes
# ---------------------------
# Sort by estimated_income
sorted_clusters = centers[:, 0].argsort()

cluster_mapping = {
    sorted_clusters[0]: "Economy",
    sorted_clusters[1]: "Standard",
    sorted_clusters[2]: "Premium"
}

df["client_class"] = df["cluster_id"].map(cluster_mapping)

# Save model
joblib.dump(kmeans, "model_generators/clustering/clustering_model.pkl")

# ---------------------------
# STEP 5: Silhouette Score
# ---------------------------
silhouette_avg = round(silhouette_score(X, df["cluster_id"]), 2)

# ---------------------------
# STEP 6: Cluster Distances
# ---------------------------
cluster_distances = np.linalg.norm(X - centers[df["cluster_id"]], axis=1)

distance_cv = round(
    (cluster_distances.std(ddof=1) / cluster_distances.mean()) * 100,
    2
)

# ---------------------------
# STEP 7: Core Samples
# ---------------------------
core_distance_threshold = np.quantile(cluster_distances, CORE_SAMPLE_PERCENTILE)

core_mask = cluster_distances <= core_distance_threshold

refined_silhouette = round(
    silhouette_score(X[core_mask], df.loc[core_mask, "cluster_id"]),
    2
)

refined_distance_cv = round(
    (cluster_distances[core_mask].std(ddof=1) /
     cluster_distances[core_mask].mean()) * 100,
    2
)

# ---------------------------
# STEP 8: Cluster Statistics
# ---------------------------
cluster_stats = df.groupby("client_class")[SEGMENT_FEATURES].agg(["mean", "std"])

for feature in SEGMENT_FEATURES:
    cluster_stats[(feature, "cv")] = (
        cluster_stats[(feature, "std")] /
        cluster_stats[(feature, "mean")]
    ) * 100

cluster_summary = cluster_stats.copy()

cluster_summary.columns = [
    f"{col[0]}_{col[1]}" for col in cluster_summary.columns
]

cluster_summary = cluster_summary.reset_index()

cluster_counts = df["client_class"].value_counts() \
    .rename_axis("client_class") \
    .reset_index(name="count")

cluster_summary = cluster_summary.merge(
    cluster_counts,
    on="client_class"
)

# ---------------------------
# STEP 9: Overall CV
# ---------------------------
overall_income_cv = round(
    (df["estimated_income"].std() /
     df["estimated_income"].mean()) * 100,
    2
)

overall_price_cv = round(
    (df["selling_price"].std() /
     df["selling_price"].mean()) * 100,
    2
)

income_cv_avg = round(cluster_summary["estimated_income_cv"].mean(), 2)
price_cv_avg = round(cluster_summary["selling_price_cv"].mean(), 2)

# ---------------------------
# STEP 10: Verification
# ---------------------------
cv_improvement_income = all(
    cluster_summary["estimated_income_cv"] < overall_income_cv
)

cv_improvement_price = all(
    cluster_summary["selling_price_cv"] < overall_price_cv
)

income_cv_target_met = overall_income_cv < 30

# ---------------------------
# Comparison Table
# ---------------------------
comparison_df = df[
    ["client_name", "estimated_income", "selling_price", "client_class"]
]

# ---------------------------
# Prediction Functions
# ---------------------------
def predict_client_segment(estimated_income, predicted_price):
    try:
        model = joblib.load(
            "model_generators/clustering/clustering_model.pkl"
        )

        scaler = joblib.load(
            "model_generators/clustering/scaler.pkl"
        )

        x_raw = np.array([[estimated_income, predicted_price]])

        x_log = np.log1p(x_raw)

        x_scaled = scaler.transform(x_log)

        cluster_id = model.predict(x_scaled)[0]

        return cluster_id

    except:
        return 0


def get_client_class_name(cluster_id):

    mapping = {
        0: "Economy",
        1: "Standard",
        2: "Premium"
    }

    return mapping.get(cluster_id, "Unknown")


# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate_clustering_model():
    return {
        "silhouette": silhouette_avg,
        "coefficient_variation": distance_cv,
        "refined_silhouette": refined_silhouette,
        "refined_coefficient_variation": refined_distance_cv,
        "income_cv": income_cv_avg,
        "price_cv": price_cv_avg,
        "overall_income_cv": overall_income_cv,
        "overall_price_cv": overall_price_cv,
        "cv_improvement_income": cv_improvement_income,
        "cv_improvement_price": cv_improvement_price,
        "income_cv_target_met": income_cv_target_met,
        "refined_sample_size": int(core_mask.sum()),
        "summary": cluster_summary.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False
        ),
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False
        )
    }


# ---------------------------
# CLI Output
# ---------------------------
if __name__ == "__main__":

    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Distance CV: {distance_cv}%")
    print(f"Refined Silhouette: {refined_silhouette}")
    print(f"Refined CV: {refined_distance_cv}%")

    print(f"Global Income CV: {overall_income_cv}% (Target < 30%: {income_cv_target_met})")
    print(f"Global Price CV: {overall_price_cv}%")

    print(f"Within-Cluster Income CV improved: {cv_improvement_income}")
    print(f"Within-Cluster Price CV improved: {cv_improvement_price}")

    print(f"Refined Sample Size: {core_mask.sum()}")

    print("\nCluster Summary:")
    print(cluster_summary[[
        "client_class",
        "estimated_income_cv",
        "selling_price_cv",
        "count"
    ]])