import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull

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

x_col      = "X"
y_col      = "Y"
label_col  = None
numeric_cols = []

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
    if non_numeric:
        use_label = st.sidebar.checkbox("Color by class column?", value=True)
        if use_label:
            label_col = st.sidebar.selectbox("Class column", non_numeric)

    MAX_POINTS = 500
    if len(df) > MAX_POINTS:
        st.sidebar.warning(
            f"⚠️ Dataset has {len(df)} rows. Sampling {MAX_POINTS} for performance.")
        df = df.sample(MAX_POINTS, random_state=42).reset_index(drop=True)

# -------------------------
# AUTO EPSILON RANGE
# -------------------------
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parameters")

eps_default = 1.0
eps_max     = 10.0
eps_step    = 0.1

if df is not None and len(numeric_cols) >= 2:
    col_range = float(df[x_col].max() - df[x_col].min())
    if col_range > 100:
        eps_max     = round(col_range * 0.3, 1)
        eps_default = round(col_range * 0.05, 1)
        eps_step    = max(round(col_range * 0.005, 1), 0.1)
    elif col_range > 10:
        eps_max     = round(col_range * 0.5, 1)
        eps_default = round(col_range * 0.05, 1)
        eps_step    = 0.5
    else:
        eps_max     = max(round(col_range, 1), 1.0)
        eps_default = max(round(col_range * 0.1, 2), 0.1)
        eps_step    = 0.05

    eps_max     = max(eps_max, 1.0)
    eps_default = max(eps_default, 0.1)

eps         = st.sidebar.slider("Epsilon (ε)", 0.1, eps_max, eps_default, step=eps_step)
min_samples = st.sidebar.slider("MinPts", 1, 10, 5)

st.sidebar.markdown("---")
if df is not None:
    st.sidebar.info(
        f"💡 **Suggested for `{x_col}` vs `{y_col}`:**\n\nε ≈ {eps_default},  MinPts = 5")

# -------------------------
# HELPER: Draw convex hull around a cluster
# -------------------------
def draw_hull(ax, points, color, alpha=0.15):
    if len(points) < 3:
        ax.scatter(points[:, 0], points[:, 1],
                   s=200, color=color, alpha=alpha,
                   edgecolors=color, linewidths=2)
        return
    try:
        hull = ConvexHull(points)
        hull_pts = np.append(hull.vertices, hull.vertices[0])
        ax.fill(points[hull_pts, 0], points[hull_pts, 1],
                alpha=alpha, color=color)
        ax.plot(points[hull_pts, 0], points[hull_pts, 1],
                color=color, linewidth=1.2, alpha=0.6)
    except Exception:
        pass

# -------------------------
# MAIN LOGIC
# -------------------------
if df is not None:
    X = df[[x_col, y_col]].values

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(20))
    st.caption(
        f"Showing first 20 of {len(df)} rows — "
        f"Columns used: **{x_col}** (X), **{y_col}** (Y)")

    # NEIGHBORS
    nbrs = NearestNeighbors(radius=eps).fit(X)
    distances, indices = nbrs.radius_neighbors(X)

    # CLASSIFICATION
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

    df_result          = df[[x_col, y_col]].copy()
    df_result["Type"]  = point_types

    # DBSCAN
    model              = DBSCAN(eps=eps, min_samples=min_samples)
    labels             = model.fit_predict(X)
    df_result["Cluster"] = labels

    # SUMMARY
    core_count   = point_types.count("Core")
    border_count = point_types.count("Border")
    noise_count  = point_types.count("Noise")
    n_clusters   = len(set(labels)) - (1 if -1 in labels else 0)

    st.subheader("📈 Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Core Points",    core_count)
    c2.metric("🟠 Border Points",  border_count)
    c3.metric("🔴 Noise Points",   noise_count)
    c4.metric("🔵 Clusters Found", n_clusters)

    st.subheader("📋 Classification Table")
    st.dataframe(df_result)

    # CLASS vs CLUSTER
    if label_col:
        st.subheader("🔍 Class vs Cluster Comparison")
        df_comp = df_result.copy()
        df_comp["Class"] = df[label_col].values
        comparison = pd.crosstab(df_comp["Class"], df_comp["Cluster"], margins=True)
        comparison.columns = [
            f"Cluster {c}" if c != "All" else "Total"
            for c in comparison.columns]
        st.dataframe(comparison)

    # K-DISTANCE PLOT
    st.subheader("📐 K-Distance Plot — helps choose ε")
    st.caption("Look for the **elbow point** — that's your ideal ε")
    nbrs_k        = NearestNeighbors(n_neighbors=min_samples).fit(X)
    k_distances,_ = nbrs_k.kneighbors(X)
    k_dist_sorted = np.sort(k_distances[:, -1])[::-1]

    fig_k, ax_k = plt.subplots(figsize=(8, 3))
    ax_k.plot(k_dist_sorted, color="steelblue", linewidth=1.5)
    ax_k.axhline(y=eps, color="red", linestyle="--",
                 linewidth=1.2, label=f"Current ε = {eps}")
    ax_k.fill_between(range(len(k_dist_sorted)), k_dist_sorted,
                       alpha=0.1, color="steelblue")
    ax_k.set_xlabel("Points sorted by distance")
    ax_k.set_ylabel(f"{min_samples}-NN Distance")
    ax_k.set_title(f"K-Distance Graph (k={min_samples})")
    ax_k.legend(); ax_k.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig_k); plt.close(fig_k)

    # -------------------------
    # STEP-BY-STEP VISUALIZATION
    # -------------------------
    st.subheader("🎞 Step-by-Step Visualization")

    step_labels = {
        1: "Step 1 — Raw Data",
        2: "Step 2 — ε Neighborhoods",
        3: "Step 3 — Core / Border / Noise Classification",
        4: "Step 4 — Cluster Expansion",
        5: "Step 5 — Final Result"
    }

    step = st.slider("Step", 1, 5, 1)
    st.caption(f"**{step_labels[step]}**")

    show_annotations = len(X) <= 30
    show_circles     = len(X) <= 100

    CORE_COLOR   = "#27ae60"
    BORDER_COLOR = "#e67e22"
    NOISE_COLOR  = "#e74c3c"

    color_map  = {"Core": CORE_COLOR, "Border": BORDER_COLOR, "Noise": NOISE_COLOR}
    marker_map = {"Core": "o", "Border": "s", "Noise": "X"}

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    # ------ STEP 1: Raw Data ------
    if step == 1:
        if label_col:
            classes        = df[label_col].values
            unique_classes = sorted(set(classes))
            cmap           = plt.cm.tab10.colors
            for idx, cls in enumerate(unique_classes):
                mask = classes == cls
                ax.scatter(X[mask, 0], X[mask, 1],
                           color=cmap[idx % len(cmap)],
                           s=40, label=str(cls), alpha=0.75,
                           edgecolors="white", linewidths=0.4, zorder=5)
            ax.legend(title=label_col, fontsize=9,
                      framealpha=0.9, edgecolor="gray")
            ax.set_title(f"Step 1: Raw Data — colored by '{label_col}'",
                         fontsize=13, fontweight="bold")
        else:
            ax.scatter(X[:, 0], X[:, 1], color="steelblue",
                       s=40, alpha=0.75,
                       edgecolors="white", linewidths=0.4, zorder=5)
            ax.set_title("Step 1: Raw Data — All Points",
                         fontsize=13, fontweight="bold")

        if show_annotations:
            for i, (xi, yi) in enumerate(X):
                ax.annotate(f"P{i+1}", (xi, yi),
                            textcoords="offset points",
                            xytext=(5, 3), fontsize=7, color="#333")
        ax.margins(0.1)

    # ------ STEP 2: ε Neighborhoods ------
    elif step == 2:
        ax.scatter(X[:, 0], X[:, 1], color="steelblue",
                   s=40, alpha=0.8,
                   edgecolors="white", linewidths=0.4, zorder=5)
        if show_circles:
            for p in X:
                circle = plt.Circle((p[0], p[1]), eps,
                                     fill=True,
                                     facecolor="steelblue",
                                     alpha=0.04,
                                     edgecolor="steelblue",
                                     linewidth=0.6, linestyle="--")
                ax.add_patch(circle)
            ax.set_aspect("equal")
            ax.autoscale_view()
        else:
            st.info(f"ℹ️ ε circles hidden for {len(X)} points — too cluttered to display.")
        ax.margins(0.1)
        ax.set_title(f"Step 2: ε = {eps} Neighborhoods",
                     fontsize=13, fontweight="bold")

    # ------ STEP 3: Core / Border / Noise ------
    elif step == 3:
        # Background shading per type
        for ptype, color, size, zord, alpha in [
            ("Noise",  NOISE_COLOR,  45,  3, 0.6),
            ("Border", BORDER_COLOR, 70,  4, 0.85),
            ("Core",   CORE_COLOR,   100, 5, 1.0),
        ]:
            idxs = [i for i, t in enumerate(point_types) if t == ptype]
            if not idxs:
                continue
            pts = X[idxs]

            # Glow effect for core points
            if ptype == "Core":
                ax.scatter(pts[:, 0], pts[:, 1],
                           color=color, s=size * 3,
                           alpha=0.12, zorder=zord - 1)

            ax.scatter(pts[:, 0], pts[:, 1],
                       color=color,
                       marker=marker_map[ptype],
                       s=size,
                       label=f"{ptype}  ({len(idxs)} pts)",
                       zorder=zord,
                       edgecolors="white",
                       linewidths=0.5,
                       alpha=alpha)

        if show_annotations:
            for i, (xi, yi) in enumerate(X):
                ax.annotate(point_types[i][0],
                            (xi, yi),
                            textcoords="offset points",
                            xytext=(5, 3), fontsize=7,
                            fontweight="bold",
                            color=color_map[point_types[i]])

        ax.margins(0.1)
        legend = ax.legend(title="Point Type", fontsize=10,
                           title_fontsize=10,
                           framealpha=0.95, edgecolor="gray",
                           loc="best")
        ax.set_title(
            "Step 3: Point Classification\n"
            "🟢 Core = Dense center  |  🟠 Border = Edge of cluster  |  🔴 Noise = Isolated",
            fontsize=12, fontweight="bold")

    # ------ STEP 4: Cluster Expansion ------
    elif step == 4:
        cluster_colors = plt.cm.tab10.colors
        unique_labels  = sorted(set(labels))

        # Draw convex hulls first (background)
        for l in unique_labels:
            if l == -1:
                continue
            pts = X[labels == l]
            color = cluster_colors[l % len(cluster_colors)]
            draw_hull(ax, pts, color, alpha=0.12)

        # Draw points on top
        for l in unique_labels:
            pts = X[labels == l]
            if l == -1:
                ax.scatter(pts[:, 0], pts[:, 1],
                           color=NOISE_COLOR, marker="X",
                           s=50, label="Noise",
                           zorder=5, alpha=0.7,
                           edgecolors="white", linewidths=0.3)
            else:
                color = cluster_colors[l % len(cluster_colors)]
                ax.scatter(pts[:, 0], pts[:, 1],
                           color=color, s=50,
                           label=f"Cluster {l}  ({np.sum(labels==l)} pts)",
                           zorder=5, alpha=0.85,
                           edgecolors="white", linewidths=0.3)

        ax.legend(title="Clusters", fontsize=9,
                  title_fontsize=9,
                  framealpha=0.95, edgecolor="gray",
                  loc="best", ncol=2)
        ax.set_title(
            f"Step 4: Cluster Expansion — {n_clusters} cluster(s) found\n"
            f"Shaded regions = cluster boundaries (convex hull)",
            fontsize=12, fontweight="bold")
        ax.margins(0.1)

    # ------ STEP 5: Final Result ------
    elif step == 5:
        cluster_colors = plt.cm.tab10.colors
        unique_labels  = sorted(set(labels))

        # Draw convex hulls per cluster (background)
        for l in unique_labels:
            if l == -1:
                continue
            pts   = X[labels == l]
            color = cluster_colors[l % len(cluster_colors)]
            draw_hull(ax, pts, color, alpha=0.10)

        # Draw Noise first (bottom layer)
        noise_idxs = [i for i, t in enumerate(point_types) if t == "Noise"]
        if noise_idxs:
            ax.scatter(X[noise_idxs, 0], X[noise_idxs, 1],
                       color=NOISE_COLOR, marker="X",
                       s=50, zorder=3, alpha=0.7,
                       edgecolors="white", linewidths=0.3,
                       label=f"Noise  ({len(noise_idxs)} pts)")

        # Draw Border points
        border_idxs = [i for i, t in enumerate(point_types) if t == "Border"]
        if border_idxs:
            ax.scatter(X[border_idxs, 0], X[border_idxs, 1],
                       color=BORDER_COLOR, marker="s",
                       s=70, zorder=4, alpha=0.85,
                       edgecolors="white", linewidths=0.4,
                       label=f"Border  ({len(border_idxs)} pts)")

        # Draw Core points (top layer, with glow)
        core_idxs = [i for i, t in enumerate(point_types) if t == "Core"]
        if core_idxs:
            # Glow
            ax.scatter(X[core_idxs, 0], X[core_idxs, 1],
                       color=CORE_COLOR, s=200,
                       alpha=0.12, zorder=4)
            ax.scatter(X[core_idxs, 0], X[core_idxs, 1],
                       color=CORE_COLOR, marker="o",
                       s=90, zorder=5, alpha=1.0,
                       edgecolors="white", linewidths=0.4,
                       label=f"Core  ({len(core_idxs)} pts)")

        # Cluster number labels at centroid
        for l in unique_labels:
            if l == -1:
                continue
            pts      = X[labels == l]
            centroid = pts.mean(axis=0)
            ax.annotate(
                f"C{l}",
                centroid,
                fontsize=11, fontweight="bold",
                color=cluster_colors[l % len(cluster_colors)],
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec=cluster_colors[l % len(cluster_colors)],
                          alpha=0.85, linewidth=1.2),
                zorder=10)

        if show_annotations:
            type_short = {"Core": "C", "Border": "B", "Noise": "N"}
            for i, (xi, yi) in enumerate(X):
                ax.annotate(type_short[point_types[i]],
                            (xi, yi),
                            textcoords="offset points",
                            xytext=(6, 4), fontsize=7,
                            fontweight="bold",
                            color=color_map[point_types[i]])

        ax.legend(title="Point Type", fontsize=10,
                  title_fontsize=10,
                  framealpha=0.95, edgecolor="gray", loc="best")
        ax.set_title(
            f"Step 5: Final Result — {n_clusters} Cluster(s)  |  "
            f"{core_count} Core  |  {border_count} Border  |  {noise_count} Noise",
            fontsize=12, fontweight="bold")
        ax.margins(0.1)

    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.25, color="gray")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

else:
    st.warning("⚠️ Please provide input data from the sidebar.")
