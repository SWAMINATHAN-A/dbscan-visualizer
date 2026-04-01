import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="DBSCAN Visualizer", layout="wide")
st.title("🚀 DBSCAN Interactive Visualizer")

# -------------------------
# INPUT
# -------------------------
st.sidebar.header("📥 Input")
option = st.sidebar.radio("Input Method", ["Upload CSV", "Manual"])

df = None

if option == "Upload CSV":
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df.columns = [c.strip().rstrip(":").strip() for c in df.columns]
else:
    n = int(st.sidebar.number_input("Points", 5, 30, 10))
    data = []
    for i in range(n):
        x = st.sidebar.number_input(f"X{i+1}", value=float(i+1))
        y = st.sidebar.number_input(f"Y{i+1}", value=float(i+2))
        data.append([x, y])
    df = pd.DataFrame(data, columns=["X", "Y"])

# -------------------------
# COLUMN SELECTION
# -------------------------
st.sidebar.markdown("---")
st.sidebar.header("🗂️ Column Selection")

x_col, y_col, label_col = None, None, None
numeric_cols = []

if df is not None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    if len(numeric_cols) >= 2:
        x_col = st.sidebar.selectbox("X-axis column", numeric_cols, index=0)
        y_col = st.sidebar.selectbox(
            "Y-axis column",
            numeric_cols,
            index=1 if len(numeric_cols) > 1 else 0
        )
    else:
        st.error("CSV must have at least 2 numeric columns.")
        st.stop()

    non_numeric = [c for c in all_cols if c not in numeric_cols]

    if non_numeric:
        use_label = st.sidebar.checkbox("Color by class column?", value=True)
        if use_label:
            label_col = st.sidebar.selectbox("Class column", non_numeric)

    # Limit dataset size
    MAX_POINTS = 500
    if len(df) > MAX_POINTS:
        st.sidebar.warning(f"⚠️ Dataset has {len(df)} rows. Sampling {MAX_POINTS}.")
        df = df.sample(MAX_POINTS, random_state=42).reset_index(drop=True)

# -------------------------
# PARAMETERS
# -------------------------
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parameters")

eps = 1.0  # default
eps_default = 1.0

if df is not None and x_col is not None and y_col is not None:
    if len(df) > 0:
        col_range = float(df[x_col].max() - df[x_col].min())
    else:
        col_range = 1.0

    if col_range > 100:
        eps_max = round(col_range * 0.3, 1)
        eps_default = round(col_range * 0.05, 1)
        eps_step = round(col_range * 0.005, 1)
    elif col_range > 10:
        eps_max = round(col_range * 0.5, 1)
        eps_default = round(col_range * 0.05, 1)
        eps_step = 0.5
    else:
        eps_max = round(col_range, 1)
        eps_default = round(col_range * 0.1, 2)
        eps_step = 0.05

    eps_max = max(eps_max, 1.0)
    eps_default = max(eps_default, 0.1)
    eps_step = max(eps_step, 0.05)

    eps = st.sidebar.slider("Epsilon (ε)", 0.1, eps_max, eps_default, step=eps_step)
else:
    eps = st.sidebar.slider("Epsilon (ε)", 0.1, 10.0, 1.0, step=0.1)

min_samples = st.sidebar.slider("MinPts", 1, 10, 5)

# -------------------------
# SIDEBAR HINT
# -------------------------
st.sidebar.markdown("---")
if df is not None and x_col is not None and y_col is not None:
    st.sidebar.info(
        f"💡 Auto-suggested for `{x_col}` vs `{y_col}`:\n\n"
        f"ε ≈ {eps_default},  MinPts = 5"
    )

# -------------------------
# MAIN LOGIC
# -------------------------
if df is not None and x_col is not None and y_col is not None:
    X = df[[x_col, y_col]].values

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(20))

    # -------------------------
    # NEIGHBORS
    # -------------------------
    nbrs = NearestNeighbors(radius=eps).fit(X)
    distances, indices = nbrs.radius_neighbors(X)

    # -------------------------
    # CLASSIFICATION
    # -------------------------
    point_types = ["Noise"] * len(X)

    for i, neigh in enumerate(indices):
        if len(neigh) >= min_samples:
            point_types[i] = "Core"

    for i, neigh in enumerate(indices):
        if point_types[i] != "Core":
            for nb in neigh:
                if point_types[nb] == "Core":
                    point_types[i] = "Border"
                    break

    df_result = df[[x_col, y_col]].copy()
    df_result["Type"] = point_types

    # -------------------------
    # DBSCAN
    # -------------------------
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    df_result["Cluster"] = labels

    # -------------------------
    # SUMMARY
    # -------------------------
    st.subheader("📈 Summary")
    st.write(df_result.head())

    # -------------------------
    # SIMPLE PLOT (safe version)
    # -------------------------
    st.subheader("📍 Visualization")

    fig, ax = plt.subplots()

    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("DBSCAN Clustering")

    st.pyplot(fig)
    plt.close(fig)

else:
    st.warning("⚠️ Please provide input data from the sidebar.")
