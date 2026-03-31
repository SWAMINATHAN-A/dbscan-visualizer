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
else:
    n = int(st.sidebar.number_input("Points", 5, 30, 10))
    data = []
    for i in range(n):
        x = st.sidebar.number_input(f"X{i+1}", value=float(i+1))
        y = st.sidebar.number_input(f"Y{i+1}", value=float(i+2))
        data.append([x, y])
    df = pd.DataFrame(data, columns=["X", "Y"])

# -------------------------
# PARAMETERS
# -------------------------
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parameters")
eps = st.sidebar.slider("Epsilon (ε)", 0.1, 10.0, 1.0, step=0.1)
min_samples = st.sidebar.slider("MinPts", 1, 10, 3)

# -------------------------
# SIDEBAR HINT
# -------------------------
st.sidebar.markdown("---")
st.sidebar.info("💡 **Recommended for example CSV:**\n\nε = 1.0, MinPts = 3")

if df is not None:
    X = df[["X", "Y"]].values  # only use X and Y columns

    st.subheader("📊 Dataset")
    st.dataframe(df)

    # -------------------------
    # NEIGHBORS
    # -------------------------
    nbrs = NearestNeighbors(radius=eps).fit(X)
    distances, indices = nbrs.radius_neighbors(X)

    # -------------------------
    # CLASSIFICATION
    # -------------------------
    point_types = ["Noise"] * len(X)

    # Core points
    for i, neigh in enumerate(indices):
        if len(neigh) >= min_samples:
            point_types[i] = "Core"

    # Border points
    for i, neigh in enumerate(indices):
        if point_types[i] != "Core":
            for n in neigh:
                if point_types[n] == "Core":
                    point_types[i] = "Border"
                    break

    df_result = df[["X", "Y"]].copy()
    df_result["Type"] = point_types

    # -------------------------
    # DBSCAN
    # -------------------------
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    df_result["Cluster"] = labels

    # -------------------------
    # SUMMARY STATS
    # -------------------------
    core_count   = point_types.count("Core")
    border_count = point_types.count("Border")
    noise_count  = point_types.count("Noise")
    n_clusters   = len(set(labels)) - (1 if -1 in labels else 0)

    st.subheader("📈 Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🟢 Core Points",   core_count)
    col2.metric("🟠 Border Points", border_count)
    col3.metric("🔴 Noise Points",  noise_count)
    col4.metric("🔵 Clusters Found", n_clusters)

    st.subheader("📋 Classification Table")
    st.dataframe(df_result)

    # -------------------------
    # STEP-BY-STEP VISUALIZATION
    # -------------------------
    st.subheader("🎞 Step-by-Step Visualization")

    step_labels = {
        1: "Step 1 — Raw Data",
        2: "Step 2 — ε Neighborhoods",
        3: "Step 3 — Core / Border / Noise",
        4: "Step 4 — Cluster Expansion",
        5: "Step 5 — Final Result"
    }

    step = st.slider("Step", 1, 5, 1)
    st.caption(f"**{step_labels[step]}**")

    fig, ax = plt.subplots(figsize=(8, 6))

    # STEP 1 — Raw Data
    if step == 1:
        ax.scatter(X[:, 0], X[:, 1], color="steelblue", s=80, zorder=5)
        for i, (xi, yi) in enumerate(X):
            ax.annotate(f"P{i+1}", (xi, yi),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax.set_title("Step 1: Raw Data — All Points")
        ax.margins(0.2)

    # STEP 2 — ε Neighborhoods
    elif step == 2:
        ax.scatter(X[:, 0], X[:, 1], color="steelblue", s=80, zorder=5)
        for p in X:
            circle = plt.Circle((p[0], p[1]), eps,
                                 fill=False, color="gray",
                                 linewidth=0.8, linestyle="--")
            ax.add_patch(circle)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.margins(0.2)
        ax.set_title(f"Step 2: ε={eps} Neighborhoods — Dashed circles show reach")

    # STEP 3 — Core / Border / Noise
    elif step == 3:
        color_map  = {"Core": "#2ecc71", "Border": "#f39c12", "Noise": "#e74c3c"}
        marker_map = {"Core": "o",       "Border": "s",       "Noise": "X"}
        size_map   = {"Core": 120,       "Border": 90,        "Noise": 90}

        for ptype in ["Core", "Border", "Noise"]:
            idxs = [i for i, t in enumerate(point_types) if t == ptype]
            if idxs:
                pts = X[idxs]
                ax.scatter(pts[:, 0], pts[:, 1],
                           color=color_map[ptype],
                           marker=marker_map[ptype],
                           s=size_map[ptype],
                           label=f"{ptype} ({len(idxs)})",
                           zorder=5, edgecolors="black", linewidths=0.5)

        # Draw epsilon circles for core points only
        for i, p in enumerate(X):
            if point_types[i] == "Core":
                circle = plt.Circle((p[0], p[1]), eps,
                                     fill=True,
                                     facecolor="#2ecc71",
                                     alpha=0.08,
                                     edgecolor="#2ecc71",
                                     linewidth=0.8)
                ax.add_patch(circle)

        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.margins(0.2)
        ax.legend(title="Point Type", fontsize=9)
        ax.set_title("Step 3: Core (●), Border (■), Noise (✕)")

    # STEP 4 — Cluster Expansion
    elif step == 4:
        colors = plt.cm.tab10.colors
        unique = sorted(set(labels))
        for l in unique:
            pts = X[labels == l]
            if l == -1:
                ax.scatter(pts[:, 0], pts[:, 1],
                           color="#e74c3c", marker="X",
                           s=100, label="Noise", zorder=5,
                           edgecolors="black", linewidths=0.5)
            else:
                ax.scatter(pts[:, 0], pts[:, 1],
                           color=colors[l % len(colors)],
                           s=100, label=f"Cluster {l}",
                           zorder=5, edgecolors="black", linewidths=0.5)
        ax.legend(title="Clusters", fontsize=9)
        ax.set_title("Step 4: Cluster Expansion — Colors show cluster groups")
        ax.margins(0.2)

    # STEP 5 — Final Result with annotations
    elif step == 5:
        colors = plt.cm.tab10.colors
        unique = sorted(set(labels))
        for l in unique:
            pts = X[labels == l]
            idxs = np.where(labels == l)[0]
            if l == -1:
                ax.scatter(pts[:, 0], pts[:, 1],
                           color="#e74c3c", marker="X",
                           s=100, label="Noise", zorder=5,
                           edgecolors="black", linewidths=0.5)
            else:
                ax.scatter(pts[:, 0], pts[:, 1],
                           color=colors[l % len(colors)],
                           s=100, label=f"Cluster {l}",
                           zorder=5, edgecolors="black", linewidths=0.5)

        # Annotate each point with its type
        type_short = {"Core": "C", "Border": "B", "Noise": "N"}
        for i, (xi, yi) in enumerate(X):
            ax.annotate(type_short[point_types[i]],
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(6, 4), fontsize=8, fontweight="bold",
                        color="black")

        ax.legend(title="Clusters", fontsize=9)
        ax.set_title("Step 5: Final Result — C=Core, B=Border, N=Noise")
        ax.margins(0.2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)
    plt.close(fig)

else:
    st.warning("⚠️ Please provide input data from the sidebar.")