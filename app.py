import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="DBSCAN Visualizer", layout="wide")
st.title("🚀 DBSCAN Interactive Visualizer")

# -------------------------
# INPUT
# -------------------------
st.sidebar.header("📥 Input")
option = st.sidebar.radio("Input Method", ["Upload CSV", "Generate Random"])

df = None

if option == "Upload CSV":
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df.columns = [c.strip().rstrip(":").strip() for c in df.columns]
else:
    n = int(st.sidebar.number_input("Number of Points", min_value=10, max_value=500, value=200, step=10))
    n_clusters_gen = int(st.sidebar.number_input("Number of Blobs", min_value=1, max_value=10, value=4))
    noise_frac = st.sidebar.slider("Noise Fraction", 0.0, 0.3, 0.05, step=0.01)
    seed = int(st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42))

    rng = np.random.default_rng(seed)
    centers = rng.uniform(1, 9, size=(n_clusters_gen, 2))
    n_noise = int(n * noise_frac)
    n_blob  = n - n_noise

    points_per_blob = np.array_split(np.arange(n_blob), n_clusters_gen)
    data = []
    for ci, idx_group in enumerate(points_per_blob):
        spread = rng.uniform(0.3, 1.0)
        pts = rng.normal(loc=centers[ci], scale=spread, size=(len(idx_group), 2))
        data.append(pts)
    blob_pts = np.vstack(data)

    noise_pts = rng.uniform(0, 10, size=(n_noise, 2))
    all_pts   = np.vstack([blob_pts, noise_pts]) if n_noise > 0 else blob_pts

    df = pd.DataFrame(all_pts, columns=["X", "Y"])
    st.sidebar.success(f"✅ Generated {len(df)} points ({n_clusters_gen} blobs + {n_noise} noise)")

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
        y_col = st.sidebar.selectbox(
            "Y-axis column", numeric_cols,
            index=1 if len(numeric_cols) > 1 else 0
        )
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
        st.sidebar.warning(
            f"⚠️ Dataset has {len(df)} rows. Sampling {MAX_POINTS} for performance."
        )
        df = df.sample(MAX_POINTS, random_state=42).reset_index(drop=True)

# -------------------------
# AUTO EPSILON RANGE
# -------------------------
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parameters")

if df is not None and len(numeric_cols) >= 2:
    col_range = float(df[x_col].max() - df[x_col].min())
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

    # ── Radius neighbors (for type classification) ──────────────────────
    nbrs = NearestNeighbors(radius=eps).fit(X)
    distances, indices = nbrs.radius_neighbors(X)

    # ── Point type classification ────────────────────────────────────────
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

    # ── DBSCAN clustering ────────────────────────────────────────────────
    model  = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    df_result["Cluster"] = labels

    # ── Summary metrics ──────────────────────────────────────────────────
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

    # ── Class vs Cluster comparison ──────────────────────────────────────
    if label_col:
        st.subheader("🔍 Class vs Cluster Comparison")
        df_result["Class"] = df[label_col].values
        comparison = pd.crosstab(
            df_result["Class"], df_result["Cluster"], margins=True
        )
        comparison.columns = [
            f"Cluster {c}" if c != "All" else "Total"
            for c in comparison.columns
        ]
        st.dataframe(comparison)

    # ── K-Distance plot ──────────────────────────────────────────────────
    st.subheader("📐 K-Distance Plot (helps choose ε)")
    st.caption("Find the 'elbow' point — that's your ideal ε value")
    k = min_samples
    nbrs_k = NearestNeighbors(n_neighbors=k).fit(X)
    k_distances, _ = nbrs_k.kneighbors(X)
    k_dist_sorted  = np.sort(k_distances[:, -1])[::-1]

    fig_k, ax_k = plt.subplots(figsize=(8, 3))
    ax_k.plot(k_dist_sorted, color="steelblue", linewidth=1.5)
    ax_k.axhline(
        y=eps, color="red", linestyle="--",
        linewidth=1, label=f"Current ε = {eps}"
    )
    ax_k.set_xlabel("Points sorted by distance")
    ax_k.set_ylabel(f"{k}-NN Distance")
    ax_k.set_title(f"K-Distance Graph (k={k}) — Elbow = ideal ε")
    ax_k.legend()
    ax_k.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig_k)
    plt.close(fig_k)

    # ════════════════════════════════════════════════════════════════════
    # STEP-BY-STEP VISUALIZATION
    # ════════════════════════════════════════════════════════════════════
    st.subheader("🎞 Step-by-Step Visualization")

    step_labels = {
        1: "Step 1 — Raw Data",
        2: "Step 2 — ε Neighborhoods (all points get a circle)",
        3: "Step 3 — Core / Border / Noise  (circles clarify who qualifies)",
        4: "Step 4 — Cluster Expansion",
        5: "Step 5 — Cluster Report Card  ✨",
    }

    step = st.slider("Step", 1, 5, 1)
    st.caption(f"**{step_labels[step]}**")

    show_annotations = len(X) <= 30
    show_circles     = len(X) <= 100

    color_map  = {"Core": "#2ecc71", "Border": "#f39c12", "Noise": "#e74c3c"}
    marker_map = {"Core": "o",       "Border": "s",       "Noise": "X"}
    size_map   = {"Core": 70,        "Border": 55,        "Noise": 55}

    # ─── Steps 1–4 ───────────────────────────────────────────────────────
    if step != 5:
        fig, ax = plt.subplots(figsize=(9, 6))

        # STEP 1 — Raw Data
        if step == 1:
            if label_col:
                classes        = df[label_col].values
                unique_classes = sorted(set(classes))
                cmap           = plt.cm.tab10.colors
                for idx, cls in enumerate(unique_classes):
                    mask = classes == cls
                    ax.scatter(
                        X[mask, 0], X[mask, 1],
                        color=cmap[idx % len(cmap)],
                        s=30, label=str(cls), alpha=0.7, zorder=5
                    )
                ax.legend(title="Class", fontsize=8)
                ax.set_title(f"Step 1: Raw Data — colored by '{label_col}'")
            else:
                ax.scatter(X[:, 0], X[:, 1], color="steelblue", s=30, alpha=0.7, zorder=5)
                ax.set_title("Step 1: Raw Data — All Points")

            if show_annotations:
                for i, (xi, yi) in enumerate(X):
                    ax.annotate(
                        f"P{i+1}", (xi, yi),
                        textcoords="offset points", xytext=(5, 3), fontsize=7
                    )
            ax.margins(0.1)

        # STEP 2 — ε Neighborhoods: every point gets a circle
        elif step == 2:
            ax.scatter(X[:, 0], X[:, 1], color="steelblue", s=30, alpha=0.8, zorder=5)
            if show_circles:
                for p in X:
                    circle = plt.Circle(
                        (p[0], p[1]), eps,
                        fill=False, color="steelblue",
                        linewidth=0.6, linestyle="--", alpha=0.4
                    )
                    ax.add_patch(circle)
                ax.set_aspect("equal")
                ax.autoscale_view()
            else:
                st.info("ℹ️ ε circles hidden for datasets > 100 points.")
            ax.margins(0.1)
            ax.set_title(f"Step 2: Every point gets an imaginary ε = {eps} circle")

        # STEP 3 — Core / Border / Noise  ← FIXED (was duplicate of step 5)
        elif step == 3:
            # Draw ε circle around EVERY point, color-coded by type
            if show_circles:
                circle_colors = {"Core": "#2ecc71", "Border": "#f39c12", "Noise": "#e74c3c"}
                circle_fills  = {"Core": 0.07,      "Border": 0.04,      "Noise": 0.03}
                for i, p in enumerate(X):
                    ptype = point_types[i]
                    circle = plt.Circle(
                        (p[0], p[1]), eps,
                        fill=True,
                        facecolor=circle_colors[ptype],
                        alpha=circle_fills[ptype],
                        edgecolor=circle_colors[ptype],
                        linewidth=0.7,
                        linestyle="--"
                    )
                    ax.add_patch(circle)
                ax.set_aspect("equal")
                ax.autoscale_view()

            # Draw points on top
            for ptype in ["Core", "Border", "Noise"]:
                idxs = [i for i, t in enumerate(point_types) if t == ptype]
                if idxs:
                    pts_sub = X[idxs]
                    ax.scatter(
                        pts_sub[:, 0], pts_sub[:, 1],
                        color=color_map[ptype],
                        marker=marker_map[ptype],
                        s=size_map[ptype],
                        label=f"{ptype} ({len(idxs)})",
                        zorder=5,
                        edgecolors="black",
                        linewidths=0.4,
                        alpha=0.9
                    )

            if show_annotations:
                for i, (xi, yi) in enumerate(X):
                    nbr_count = len(indices[i])
                    ax.annotate(
                        f"{nbr_count}pts", (xi, yi),
                        textcoords="offset points",
                        xytext=(5, 3), fontsize=6.5, color="#555"
                    )

            ax.margins(0.1)
            ax.legend(title="Point Type", fontsize=9)
            ax.set_title(
                f"Step 3: Circles show who each point can 'see' (ε={eps}, MinPts={min_samples})\n"
                "🟢 Core = enough neighbors inside circle  "
                "🟠 Border = reachable  "
                "🔴 Noise = isolated"
            )

        # STEP 4 — Cluster Expansion
        elif step == 4:
            colors = plt.cm.tab10.colors
            unique = sorted(set(labels))
            for lbl in unique:
                pts_sub = X[labels == lbl]
                if lbl == -1:
                    ax.scatter(
                        pts_sub[:, 0], pts_sub[:, 1],
                        color="#e74c3c", marker="X",
                        s=40, label="Noise", zorder=5, alpha=0.7
                    )
                else:
                    ax.scatter(
                        pts_sub[:, 0], pts_sub[:, 1],
                        color=colors[lbl % len(colors)],
                        s=40, label=f"Cluster {lbl}",
                        zorder=5, alpha=0.7
                    )
            ax.legend(title="Clusters", fontsize=8, loc="upper right", ncol=2)
            ax.set_title("Step 4: Cluster Expansion — core points recruit their neighbors")
            ax.margins(0.1)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # ─── STEP 5 — Cluster Report Card ────────────────────────────────────
    else:
        st.markdown("### 🗂️ Cluster Report Card")
        st.caption(
            "One panel per cluster — showing spatial footprint, point-type breakdown, "
            "and density relative to the largest cluster."
        )

        tab_names = [f"Cluster {i}" for i in range(n_clusters)]
        if noise_count > 0:
            tab_names.append("🔴 Noise")

        if not tab_names:
            st.warning("No clusters found. Adjust ε or MinPts.")
        else:
            tabs = st.tabs(tab_names)

            cluster_colors = plt.cm.tab10.colors

            for tab_i, tab in enumerate(tabs):
                with tab:
                    is_noise_tab = (tab_i == n_clusters)

                    if is_noise_tab:
                        mask      = labels == -1
                        tab_label = "Noise"
                        tab_color = "#e74c3c"
                    else:
                        mask      = labels == tab_i
                        tab_label = f"Cluster {tab_i}"
                        tab_color = cluster_colors[tab_i % len(cluster_colors)]

                    pts_in   = X[mask]
                    types_in = [point_types[i] for i in range(len(X)) if mask[i]]

                    if len(pts_in) == 0:
                        st.write("No points.")
                        continue

                    c_core   = types_in.count("Core")
                    c_border = types_in.count("Border")
                    c_noise  = types_in.count("Noise")
                    total    = len(pts_in)

                    x_span = pts_in[:, 0].max() - pts_in[:, 0].min()
                    y_span = pts_in[:, 1].max() - pts_in[:, 1].min()
                    cx     = pts_in[:, 0].mean()
                    cy     = pts_in[:, 1].mean()

                    # ── Metric row ───────────────────────────────────────
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Points",       total)
                    m2.metric("Core",         c_core)
                    m3.metric("Border",       c_border)
                    m4.metric("X-span / Y-span",
                              f"{x_span:.2f} / {y_span:.2f}")

                    # ── Split: scatter + bar chart ───────────────────────
                    left, right = st.columns([2, 1])

                    with left:
                        fig_s, ax_s = plt.subplots(figsize=(5, 4))

                        # All other points as ghost
                        ax_s.scatter(
                            X[~mask, 0], X[~mask, 1],
                            color="#cccccc", s=15, alpha=0.3,
                            zorder=2, label="Other"
                        )

                        # ε circles for core points in this cluster
                        if show_circles:
                            for i in range(len(X)):
                                if mask[i] and point_types[i] == "Core":
                                    circ = plt.Circle(
                                        (X[i, 0], X[i, 1]), eps,
                                        fill=True,
                                        facecolor=tab_color,
                                        alpha=0.06,
                                        edgecolor=tab_color,
                                        linewidth=0.6
                                    )
                                    ax_s.add_patch(circ)
                            ax_s.set_aspect("equal")
                            ax_s.autoscale_view()

                        # Cluster points, colored by type
                        for ptype in ["Core", "Border", "Noise"]:
                            sub_idx = [
                                i for i in range(len(X))
                                if mask[i] and point_types[i] == ptype
                            ]
                            if sub_idx:
                                ax_s.scatter(
                                    X[sub_idx, 0], X[sub_idx, 1],
                                    color=color_map[ptype],
                                    marker=marker_map[ptype],
                                    s=size_map[ptype],
                                    label=ptype,
                                    zorder=5,
                                    edgecolors="black",
                                    linewidths=0.3
                                )

                        # Centroid crosshair
                        ax_s.axvline(cx, color=tab_color, linewidth=0.8,
                                     linestyle=":", alpha=0.7)
                        ax_s.axhline(cy, color=tab_color, linewidth=0.8,
                                     linestyle=":", alpha=0.7)
                        ax_s.scatter([cx], [cy], marker="+", color=tab_color,
                                     s=120, linewidths=1.5, zorder=6,
                                     label=f"Centroid ({cx:.1f}, {cy:.1f})")

                        ax_s.set_xlabel(x_col)
                        ax_s.set_ylabel(y_col)
                        ax_s.set_title(f"{tab_label} — spatial view")
                        ax_s.legend(fontsize=8)
                        ax_s.grid(True, linestyle="--", alpha=0.25)
                        ax_s.margins(0.15)
                        st.pyplot(fig_s)
                        plt.close(fig_s)

                    with right:
                        fig_b, ax_b = plt.subplots(figsize=(3, 4))

                        # Stacked horizontal bar: composition
                        bar_types  = ["Core", "Border", "Noise"]
                        bar_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
                        bar_counts = [c_core, c_border, c_noise]

                        left_val = 0
                        for bt, bc, bcount in zip(bar_types, bar_colors, bar_counts):
                            if bcount > 0:
                                ax_b.barh(
                                    0, bcount, left=left_val,
                                    color=bc, edgecolor="white",
                                    linewidth=0.5, height=0.4,
                                    label=bt
                                )
                                ax_b.text(
                                    left_val + bcount / 2, 0,
                                    f"{bcount}",
                                    ha="center", va="center",
                                    fontsize=9, color="white", fontweight="bold"
                                )
                                left_val += bcount

                        # Density bar — this cluster vs largest cluster
                        max_cluster_size = max(
                            np.sum(labels == c) for c in range(n_clusters)
                        ) if n_clusters > 0 else 1
                        density_pct = total / max_cluster_size

                        ax_b.barh(
                            1, density_pct, color=tab_color,
                            alpha=0.75, height=0.4
                        )
                        ax_b.barh(
                            1, 1, color="#eeeeee",
                            alpha=0.4, height=0.4
                        )
                        ax_b.text(
                            density_pct / 2, 1,
                            f"{density_pct * 100:.0f}%",
                            ha="center", va="center",
                            fontsize=9, color="white", fontweight="bold"
                        )

                        ax_b.set_yticks([0, 1])
                        ax_b.set_yticklabels(["Composition", "Density\nvs largest"], fontsize=9)
                        ax_b.set_xlim(0, max(1.05, density_pct + 0.05))
                        ax_b.set_title("Breakdown", fontsize=10)
                        ax_b.legend(fontsize=7, loc="lower right")
                        ax_b.grid(axis="x", linestyle="--", alpha=0.3)
                        ax_b.margins(0.1)
                        st.pyplot(fig_b)
                        plt.close(fig_b)

                    # ── Qualitative verdict ──────────────────────────────
                    if not is_noise_tab:
                        core_ratio = c_core / total if total else 0
                        if core_ratio > 0.7:
                            verdict = "🟢 Dense, well-defined cluster — high core ratio."
                        elif core_ratio > 0.3:
                            verdict = "🟡 Moderate cluster — mix of core and border points."
                        else:
                            verdict = "🔴 Sparse cluster — mostly border points, consider increasing MinPts."
                        st.info(verdict)

else:
    st.warning("⚠️ Please provide input data from the sidebar.")
