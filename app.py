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

if df is not None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols     = df.columns.tolist()

    if len(numeric_cols) >= 2:
        x_col = st.sidebar.selectbox("X-axis column", numeric_cols, index=0)
        y_col = st.sidebar.selectbox("Y-axis column", numeric_cols,
                                      index=1 if len(numeric_cols) > 1 else 0)
    else:
        st.error("CSV must have at least 2 numeric columns.")
        st.stop()

    non_numeric = [c for c in all_cols if c not in numeric_cols]
    label_col = None
    if non_numeric:
        use_label = st.sidebar.checkbox("Color by class column?", value=True)
        if use_label:
            label_col = st.sidebar.selectbox("Class column", non_numeric)

    MAX_POINTS = 500
    if len(df) > MAX_POINTS:
        st.sidebar.warning(f"⚠️ Dataset has {len(df)} rows. Sampling {MAX_POINTS} for performance.")
        df = df.sample(MAX_POINTS, random_state=42).reset_index(drop=True)

# -------------------------
# AUTO EPSILON RANGE based on selected columns
# -------------------------
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parameters")

if df is not None and len(numeric_cols) >= 2:
    col_range = float(df[x_col].max() - df[x_col].min())
    # Dynamically set slider max and default based on data range
    if col_range > 100:
        eps_max     = round(col_range * 0.3, 1)
        eps_default = round(col_range * 0.05, 1)
        eps_step    = round(col_range * 0.005, 1)
    elif col_range > 10:
        eps_max     = round(col_range * 0.5, 1)
        eps_default = round(col_range * 0.05, 1)
        eps_step    = 0.5
    else:
        eps_max     = round(col_range, 1)
        eps_default = round(col_range * 0.1, 2)
        eps_step    = 0.05

    eps_max     = max(eps_max, 1.0)
    eps_default = max(eps_default, 0.1)
    eps_step    = max(eps_step, 0.05)

    eps = st.sidebar.slider("Epsilon (ε)", 0.1, eps_max, eps_default, step=eps_step)
else:
    eps = st.sidebar.slider("Epsilon (ε)", 0.1, 10.0, 1.0, step=0.1)

min_samples = st.sidebar.slider("MinPts", 1, 10, 5)

# -------------------------
# SIDEBAR HINT
# -------------------------
st.sidebar.markdown("---")
if df is not None:
    st.sidebar.info(
        f"💡 **Auto-suggested for `{x_col}` vs `{y_col}`:**\n\n"
        f"ε ≈ {eps_default},  MinPts = 5"
    )

# -------------------------
# MAIN LOGIC
# -------------------------
if df is not None:
    X = df[[x_col, y_col]].values

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(20))
    st.caption(
        f"Showing first 20 of {len(df)} rows — "
        f"Columns used: **{x_col}** (X) and **{y_col}** (Y)"
    )

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
    model  = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    df_result["Cluster"] = labels

    # -------------------------
    # SUMMARY
    # -------------------------
    core_count   = point_types.count("Core")
    border_count = point_types.count("Border")
    noise_count  = point_types.count("Noise")
    n_clusters   = len(set(labels)) - (1 if -1 in labels else 0)

    st.subheader("📈 Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🟢 Core Points",    core_count)
    col2.metric("🟠 Border Points",  border_count)
    col3.metric("🔴 Noise Points",   noise_count)
    col4.metric("🔵 Clusters Found", n_clusters)

    st.subheader("📋 Classification Table")
    st.dataframe(df_result)

    # -------------------------
    # CLASS vs CLUSTER COMPARISON
    # -------------------------
    if label_col:
        st.subheader("🔍 Class vs Cluster Comparison")
        df_result["Class"] = df[label_col].values
        comparison = pd.crosstab(
            df_result["Class"],
            df_result["Cluster"],
            margins=True
        )
        comparison.columns = [
            f"Cluster {c}" if c != "All" else "Total"
            for c in comparison.columns
        ]
        st.dataframe(comparison)

    # -------------------------
    # K-DISTANCE PLOT (helps user pick eps)
    # -------------------------
    st.subheader("📐 K-Distance Plot (helps choose ε)")
    st.caption("Find the 'elbow' point — that's your ideal ε value")
    k = min_samples
    nbrs_k = NearestNeighbors(n_neighbors=k).fit(X)
    k_distances, _ = nbrs_k.kneighbors(X)
    k_dist_sorted  = np.sort(k_distances[:, -1])[::-1]

    fig_k, ax_k = plt.subplots(figsize=(8, 3))
    ax_k.plot(k_dist_sorted, color="steelblue", linewidth=1.5)
    ax_k.axhline(y=eps, color="red", linestyle="--",
                 linewidth=1, label=f"Current ε = {eps}")
    ax_k.set_xlabel("Points sorted by distance")
    ax_k.set_ylabel(f"{k}-NN Distance")
    ax_k.set_title(f"K-Distance Graph (k={k}) — Elbow = ideal ε")
    ax_k.legend()
    ax_k.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig_k)
    plt.close(fig_k)

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

    show_annotations = len(X) <= 30
    show_circles     = len(X) <= 100

    color_map  = {"Core": "#2ecc71", "Border": "#f39c12", "Noise": "#e74c3c"}
    marker_map = {"Core": "o",       "Border": "s",       "Noise": "X"}
    size_map   = {"Core": 60,        "Border": 45,        "Noise": 45}

    fig, ax = plt.subplots(figsize=(9, 6))

    # STEP 1 — Raw Data
    if step == 1:
        if label_col:
            classes        = df[label_col].values
            unique_classes = sorted(set(classes))
            cmap           = plt.cm.tab10.colors
            for idx, cls in enumerate(unique_classes):
                mask = classes == cls
                ax.scatter(X[mask, 0], X[mask, 1],
                           color=cmap[idx % len(cmap)],
                           s=30, label=str(cls), alpha=0.7, zorder=5)
            ax.legend(title="Class", fontsize=8)
            ax.set_title(f"Step 1: Raw Data — colored by '{label_col}'")
        else:
            ax.scatter(X[:, 0], X[:, 1],
                       color="steelblue", s=30, alpha=0.7, zorder=5)
            ax.set_title("Step 1: Raw Data — All Points")

        if show_annotations:
            for i, (xi, yi) in enumerate(X):
                ax.annotate(f"P{i+1}", (xi, yi),
                            textcoords="offset points",
                            xytext=(5, 3), fontsize=7)
        ax.margins(0.1)

    # STEP 2 — ε Neighborhoods
    elif step == 2:
        ax.scatter(X[:, 0], X[:, 1],
                   color="steelblue", s=30, alpha=0.7, zorder=5)
        if show_circles:
            for p in X:
                circle = plt.Circle((p[0], p[1]), eps,
                                     fill=False, color="gray",
                                     linewidth=0.5, linestyle="--", alpha=0.5)
                ax.add_patch(circle)
            ax.set_aspect("equal")
            ax.autoscale_view()
        else:
            st.info("ℹ️ ε circles hidden for datasets > 100 points to avoid clutter.")
        ax.margins(0.1)
        ax.set_title(f"Step 2: ε = {eps} Neighborhoods")

    # STEP 3 — Core / Border / Noise
    elif step == 3:
        for ptype in ["Core", "Border", "Noise"]:
            idxs = [i for i, t in enumerate(point_types) if t == ptype]
            if idxs:
                pts = X[idxs]
                ax.scatter(pts[:, 0], pts[:, 1],
                           color=color_map[ptype],
                           marker=marker_map[ptype],
                           s=size_map[ptype],
                           label=f"{ptype} ({len(idxs)})",
                           zorder=5,
                           edgecolors="black",
                           linewidths=0.3,
                           alpha=0.8)

        if show_circles:
            for i, p in enumerate(X):
                if point_types[i] == "Core":
                    circle = plt.Circle((p[0], p[1]), eps,
                                         fill=True,
                                         facecolor="#2ecc71",
                                         alpha=0.05,
                                         edgecolor="#2ecc71",
                                         linewidth=0.5)
                    ax.add_patch(circle)
            ax.set_aspect("equal")
            ax.autoscale_view()

        ax.margins(0.1)
        ax.legend(title="Point Type", fontsize=9)
        ax.set_title("Step 3: 🟢 Core  🟠 Border  🔴 Noise")

    # STEP 4 — Cluster Expansion
    elif step == 4:
        colors = plt.cm.tab10.colors
        unique = sorted(set(labels))
        for l in unique:
            pts = X[labels == l]
            if l == -1:
                ax.scatter(pts[:, 0], pts[:, 1],
                           color="#e74c3c", marker="X",
                           s=40, label="Noise",
                           zorder=5, alpha=0.7)
            else:
                ax.scatter(pts[:, 0], pts[:, 1],
                           color=colors[l % len(colors)],
                           s=40, label=f"Cluster {l}",
                           zorder=5, alpha=0.7)
        ax.legend(title="Clusters", fontsize=8,
                  loc="upper right", ncol=2)
        ax.set_title("Step 4: Cluster Expansion")
        ax.margins(0.1)

    # STEP 5 — Final Result
    elif step == 5:
        if show_circles:
            for i, p in enumerate(X):
                if point_types[i] == "Core":
                    circle = plt.Circle((p[0], p[1]), eps,
                                         fill=True,
                                         facecolor="#2ecc71",
                                         alpha=0.05,
                                         edgecolor="#2ecc71",
                                         linewidth=0.5)
                    ax.add_patch(circle)
            ax.set_aspect("equal")
            ax.autoscale_view()

        for ptype in ["Core", "Border", "Noise"]:
            idxs = [i for i, t in enumerate(point_types) if t == ptype]
            if idxs:
                pts = X[idxs]
                ax.scatter(pts[:, 0], pts[:, 1],
                           color=color_map[ptype],
                           marker=marker_map[ptype],
                           s=size_map[ptype],
                           label=f"{ptype} ({len(idxs)})",
                           zorder=5,
                           edgecolors="black",
                           linewidths=0.3,
                           alpha=0.8)

        if show_annotations:
            type_short = {"Core": "C", "Border": "B", "Noise": "N"}
            for i, (xi, yi) in enumerate(X):
                cluster_label = f"C{labels[i]}" if labels[i] != -1 else "N"
                ax.annotate(
                    f"{type_short[point_types[i]]}/{cluster_label}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(5, 3), fontsize=7, fontweight="bold"
                )

        ax.margins(0.1)
        ax.legend(title="Point Type", fontsize=9)
        ax.set_title("Step 5: Final Result — 🟢 Core  🟠 Border  🔴 Noise")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

else:
    st.warning("⚠️ Please provide input data from the sidebar.")
