import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="DBSCAN Visualizer", layout="wide", page_icon="🔬")

# ── Top header banner ────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 1.6rem 2rem; border-radius: 12px; margin-bottom: 1rem;
            border: 1px solid #e94560;">
  <h1 style="color:#e94560; margin:0; font-size:2rem; letter-spacing:1px;">
      🔬 DBSCAN Interactive Visualizer
  </h1>
  <p style="color:#a0aec0; margin:0.3rem 0 0 0; font-size:0.9rem;">
      Density-Based Spatial Clustering of Applications with Noise — step-by-step exploration
  </p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# SIDEBAR — NAVIGATION
# ════════════════════════════════════════════════════════════════════
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/120px-Scikit_learn_logo_small.svg.png", width=90)
st.sidebar.markdown("## 🧭 Navigation")
nav = st.sidebar.radio("Go to", ["🏠 Visualizer", "❓ Help & Theory", "⬇️ Download Results", "👥 About / Credits"])
st.sidebar.markdown("---")

# ════════════════════════════════════════════════════════════════════
# SECTION: HELP & THEORY
# ════════════════════════════════════════════════════════════════════
if nav == "❓ Help & Theory":
    st.markdown("## ❓ Help & DBSCAN Theory")
    st.markdown("---")

    with st.expander("🔎 What is DBSCAN?", expanded=True):
        st.markdown("""
**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised
machine learning algorithm that groups together points that are **closely packed** in space,
and marks points that lie alone in low-density regions as **outliers (noise)**.

Unlike K-Means, DBSCAN:
- ✅ Does **not** require specifying the number of clusters in advance
- ✅ Can find clusters of **arbitrary shapes** (not just spherical)
- ✅ Naturally handles **noise and outliers**
- ❌ Struggles with datasets of **varying density**
- ❌ Sensitive to the choice of **ε** and **MinPts**
        """)

    with st.expander("⚙️ Key Parameters — ε (Epsilon) and MinPts"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
### ε (Epsilon)
The **radius** of the neighborhood around each point.

- Think of it as: *"How far can two points be and still be considered neighbors?"*
- **Too small** → Most points become noise (under-clustering)
- **Too large** → Everything merges into one cluster (over-clustering)

💡 **How to choose ε:**
Use the **K-Distance Graph** (provided in the app).
Plot the k-nearest-neighbor distances and look for the **"elbow"** — that's your ideal ε.
            """)
        with col2:
            st.markdown("""
### MinPts (Minimum Points)
The **minimum number of points** required within ε-radius to form a dense region.

- Think of it as: *"How many neighbors must a point have to be a 'core'?"*
- **Too low** → Too many core points, merges everything
- **Too high** → Too many noise points, misses real clusters

💡 **Rule of thumb:**
`MinPts ≥ dimensionality + 1`  
For 2D data: start with **MinPts = 3 to 5**
            """)

    with st.expander("🟢🟠🔴 Point Types — Core, Border, Noise"):
        st.markdown("""
| Point Type | Definition | Symbol |
|---|---|---|
| 🟢 **Core** | Has ≥ MinPts neighbors within ε (including itself) | Large circle |
| 🟠 **Border** | Has < MinPts neighbors but lies within ε of a Core point | Square |
| 🔴 **Noise** | Neither Core nor Border — completely isolated | X mark |

**How the algorithm works:**
1. For each unvisited point, count neighbors within ε
2. If neighbors ≥ MinPts → it's a **Core** point; start a new cluster
3. Recursively add all density-reachable points to the cluster
4. Points unreachable from any Core → **Noise**
        """)

    with st.expander("📊 Step-by-Step Visual Guide"):
        st.markdown("""
### Step 1 — Raw Data
Just the original dataset plotted. If a class column is available, points are
colored by their true class labels so you can later compare with DBSCAN's output.

### Step 2 — ε-Neighborhood Density Field
A **heatmap** is drawn over the scatter plot showing how many neighbors each
region of space has within radius ε. The cyan dashed contour marks the MinPts
threshold — points above this line are candidates to become Core points.
The **brightest (hottest) region** marks the densest point, with an ε-circle spotlight.

### Step 3 — Core / Border / Noise Classification
Every point is now classified:
- 🟢 **Core** points have ≥ MinPts neighbors — shown as circles
- 🟠 **Border** points live inside a Core's ε-circle — shown as squares
- 🔴 **Noise** points are isolated — shown as X marks
Each point also shows a dashed ε-circle to reveal its neighborhood visually.

### Step 4 — Cluster Expansion
DBSCAN starts from each Core point and **expands** outward — recruiting neighboring
Core and Border points into the same cluster. Different clusters get different colors.
Noise remains red. This step shows the **final cluster assignment**.

### Step 5 — Cluster Report Card
A **deep-dive dashboard** per cluster showing:
- Spatial scatter with centroid crosshair and ε-circles
- Composition bar (Core vs Border vs Noise within each cluster)
- Density relative to the largest cluster
- Qualitative verdict on cluster quality
        """)

    with st.expander("📐 K-Distance Graph — How to Find the Best ε"):
        st.markdown("""
The **K-Distance Graph** helps you select the right ε:

1. For each point, compute the distance to its **k-th nearest neighbor** (k = MinPts)
2. Sort these distances in **descending order** and plot them
3. Look for the **"elbow"** — a sharp bend in the curve

**Interpretation:**
- The elbow indicates the transition from dense regions (cluster) to sparse (noise)
- The ε value at the elbow is your optimal choice
- If the curve is very smooth → your data may not have clear cluster structure

The red dashed line in the K-Distance plot shows your current ε for reference.
        """)

    with st.expander("🆚 DBSCAN vs K-Means vs Hierarchical Clustering"):
        st.markdown("""
| Feature | DBSCAN | K-Means | Hierarchical |
|---|---|---|---|
| Cluster shape | Any shape | Spherical | Any |
| # clusters needed | ❌ Not required | ✅ Required | ❌ Not required |
| Noise handling | ✅ Built-in | ❌ None | ❌ Poor |
| Scalability | Medium | High | Low |
| Varying density | ❌ Difficult | ❌ Difficult | ✅ Better |
| Parameters | ε, MinPts | k | linkage, threshold |
        """)

    st.info("💡 **Tip:** Navigate to 🏠 Visualizer in the sidebar to start experimenting with your data!")
    st.stop()

# ════════════════════════════════════════════════════════════════════
# SECTION: ABOUT / CREDITS
# ════════════════════════════════════════════════════════════════════
if nav == "👥 About / Credits":
    st.markdown("## 👥 About This Project")
    st.markdown("---")

    st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 1px solid #e94560; border-radius: 14px; padding: 2rem; margin-bottom: 1.5rem;">
  <h2 style="color:#e94560; margin-top:0;">🔬 DBSCAN Interactive Visualizer</h2>
  <p style="color:#a0aec0; font-size:1rem; line-height:1.7;">
    An interactive educational tool built to help students and practitioners understand
    the DBSCAN clustering algorithm through live parameter tuning, step-by-step animations,
    and detailed cluster analytics.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🛠️ Developed By")
    dev1, dev2 = st.columns(2)
    with dev1:
        st.markdown("""
<div style="background:#1e293b; border-radius:12px; padding:1.5rem;
            border-left: 4px solid #e94560; text-align:center;">
  <div style="font-size:3rem;">👨‍💻</div>
  <h3 style="color:#e94560; margin:0.5rem 0 0.2rem 0;">Roohith R</h3>
  <p style="color:#94a3b8; margin:0; font-size:0.9rem;">Developer</p>
</div>
""", unsafe_allow_html=True)

    with dev2:
        st.markdown("""
<div style="background:#1e293b; border-radius:12px; padding:1.5rem;
            border-left: 4px solid #3b82f6; text-align:center;">
  <div style="font-size:3rem;">👨‍💻</div>
  <h3 style="color:#3b82f6; margin:0.5rem 0 0.2rem 0;">Lakkshanth R</h3>
  <p style="color:#94a3b8; margin:0; font-size:0.9rem;">Developer</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🎓 Mentored By")
    st.markdown("""
<div style="background:#1e293b; border-radius:12px; padding:1.5rem;
            border-left: 4px solid #10b981; text-align:center; max-width:400px; margin:auto;">
  <div style="font-size:3rem;">🧑‍🏫</div>
  <h3 style="color:#10b981; margin:0.5rem 0 0.2rem 0;">Swaminathan A</h3>
  <p style="color:#94a3b8; margin:0; font-size:0.9rem;">Mentor & Guide</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧰 Tech Stack")
    tech_cols = st.columns(5)
    techs = [("🐍", "Python"), ("📊", "Streamlit"), ("🔢", "NumPy"), ("🐼", "Pandas"), ("🤖", "Scikit-learn")]
    for col, (icon, name) in zip(tech_cols, techs):
        col.markdown(f"""
<div style="background:#1e293b; border-radius:10px; padding:0.8rem;
            text-align:center; border: 1px solid #334155;">
  <div style="font-size:1.8rem;">{icon}</div>
  <p style="color:#94a3b8; margin:0; font-size:0.8rem;">{name}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📦 Features")
    st.markdown("""
- 📁 Upload your own CSV or generate synthetic blobs with noise control  
- 🎛️ Real-time DBSCAN parameter tuning (ε and MinPts)  
- 📐 K-Distance graph for optimal ε selection  
- 🎞️ 5-step animated walkthrough of the clustering process  
- 🟢🟠🔴 Core / Border / Noise point classification  
- 📋 Cluster Report Card with per-cluster metrics and breakdown charts  
- 🔍 Class vs Cluster comparison for labeled datasets  
- ⬇️ Download results as CSV  
    """)

    st.markdown("---")
    st.caption("© 2025 Roohith R & Lakkshanth R — Mentored by Swaminathan A")
    st.stop()

# ════════════════════════════════════════════════════════════════════
# SIDEBAR — INPUT (only shown for Visualizer / Download)
# ════════════════════════════════════════════════════════════════════
st.sidebar.header("📥 Input")
option = st.sidebar.radio("Input Method", ["Upload CSV", "Use Sample Dataset"])

df = None

SAMPLE_CSV_PATH = "large.csv"   # place large.csv in the same folder as app.py

if option == "Upload CSV":
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df.columns = [c.strip().rstrip(":").strip() for c in df.columns]
else:
    # ── Load the bundled MAGIC Gamma Telescope dataset ──────────────
    import os
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SAMPLE_CSV_PATH)

    if os.path.exists(sample_path):
        @st.cache_data
        def load_sample():
            d = pd.read_csv(sample_path)
            d.columns = [c.strip().rstrip(":").strip() for c in d.columns]
            return d

        df = load_sample()
        st.sidebar.success(
            f"✅ Sample dataset loaded\n\n"
            f"📄 **MAGIC Gamma Telescope**\n"
            f"🔢 {len(df):,} rows · {len(df.columns)} columns"
        )
        st.sidebar.caption(
            "Features: fLength, fWidth, fSize, fConc, fConc1, "
            "fAsym, fM3Long, fM3Trans, fAlpha, fDist · Class: g / h"
        )
    else:
        st.sidebar.error(
            f"⚠️ `{SAMPLE_CSV_PATH}` not found next to app.py.\n\n"
            "Please place `large.csv` in the same folder as `app.py` and restart."
        )

# ── Column Selection ─────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🗂️ Column Selection")

label_col   = None
numeric_cols = []

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
    if non_numeric:
        use_label = st.sidebar.checkbox("Color by class column?", value=True)
        if use_label:
            label_col = st.sidebar.selectbox("Class column", non_numeric)

    MAX_POINTS = 500
    if len(df) > MAX_POINTS:
        st.sidebar.warning(f"⚠️ Dataset has {len(df)} rows. Sampling {MAX_POINTS} for performance.")
        df = df.sample(MAX_POINTS, random_state=42).reset_index(drop=True)

# ── Parameters ───────────────────────────────────────────────────────
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
    eps         = st.sidebar.slider("Epsilon (ε)", 0.1, 10.0, 1.0, step=0.1)
    eps_default = 1.0

min_samples = st.sidebar.slider("MinPts", 1, 10, 5)

st.sidebar.markdown("---")
if df is not None:
    st.sidebar.info(
        f"💡 **Auto-suggested for `{x_col}` vs `{y_col}`:**\n\n"
        f"ε ≈ {eps_default},  MinPts = 5"
    )

# ── Credits footer in sidebar ─────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.75rem; color:#64748b; line-height:1.6;">
  <b style="color:#94a3b8;">Developed by</b><br>
  Roohith R &amp; Lakkshanth R<br>
  <b style="color:#94a3b8;">Mentored by</b><br>
  Swaminathan A
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ════════════════════════════════════════════════════════════════════
if df is None:
    st.warning("⚠️ Please provide input data from the sidebar.")
    st.stop()

X = df[[x_col, y_col]].values

# ── Radius neighbors (for type classification) ────────────────────────
nbrs = NearestNeighbors(radius=eps).fit(X)
distances, indices = nbrs.radius_neighbors(X)

# ── Point type classification ─────────────────────────────────────────
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

# ── DBSCAN clustering ─────────────────────────────────────────────────
model  = DBSCAN(eps=eps, min_samples=min_samples)
labels = model.fit_predict(X)
df_result["Cluster"] = labels

# ── Summary metrics ───────────────────────────────────────────────────
core_count   = point_types.count("Core")
border_count = point_types.count("Border")
noise_count  = point_types.count("Noise")
n_clusters   = len(set(labels)) - (1 if -1 in labels else 0)

# ════════════════════════════════════════════════════════════════════
# SECTION: DOWNLOAD RESULTS
# ════════════════════════════════════════════════════════════════════
if nav == "⬇️ Download Results":
    st.markdown("## ⬇️ Download Results")
    st.markdown("---")

    if label_col:
        df_result["Class"] = df[label_col].values

    st.markdown("### 📋 Full Classification Table")
    st.dataframe(df_result, use_container_width=True)

    # Summary CSV
    summary_data = {
        "Metric": ["Total Points", "Core Points", "Border Points", "Noise Points",
                   "Clusters Found", "Epsilon (ε)", "MinPts"],
        "Value": [len(X), core_count, border_count, noise_count,
                  n_clusters, eps, min_samples]
    }
    df_summary = pd.DataFrame(summary_data)

    col_a, col_b, col_c = st.columns(3)

    # ── Download 1: Full results ──────────────────────────────────────
    csv_results = df_result.to_csv(index=False).encode("utf-8")
    col_a.download_button(
        label="📥 Download Classification CSV",
        data=csv_results,
        file_name="dbscan_classification_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ── Download 2: Summary ───────────────────────────────────────────
    csv_summary = df_summary.to_csv(index=False).encode("utf-8")
    col_b.download_button(
        label="📊 Download Summary CSV",
        data=csv_summary,
        file_name="dbscan_summary.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ── Download 3: Cluster-only (no noise) ───────────────────────────
    df_clusters_only = df_result[df_result["Cluster"] != -1]
    csv_clusters = df_clusters_only.to_csv(index=False).encode("utf-8")
    col_c.download_button(
        label="🔵 Download Clusters Only (no noise)",
        data=csv_clusters,
        file_name="dbscan_clusters_only.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### 📊 Summary Statistics")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🟢 Core Points",    core_count)
    s2.metric("🟠 Border Points",  border_count)
    s3.metric("🔴 Noise Points",   noise_count)
    s4.metric("🔵 Clusters Found", n_clusters)
    st.dataframe(df_summary, use_container_width=True)

    if label_col:
        st.markdown("### 🔍 Class vs Cluster Comparison")
        comparison = pd.crosstab(df_result["Class"], df_result["Cluster"], margins=True)
        comparison.columns = [f"Cluster {c}" if c != "All" else "Total" for c in comparison.columns]
        st.dataframe(comparison, use_container_width=True)
        csv_comp = comparison.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Download Class vs Cluster CSV",
            data=csv_comp,
            file_name="dbscan_class_vs_cluster.csv",
            mime="text/csv"
        )

    st.stop()

# ════════════════════════════════════════════════════════════════════
# SECTION: MAIN VISUALIZER
# ════════════════════════════════════════════════════════════════════
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(20))
st.caption(
    f"Showing first 20 of {len(df)} rows — "
    f"Columns used: **{x_col}** (X) and **{y_col}** (Y)"
)

st.subheader("📈 Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("🟢 Core Points",    core_count)
col2.metric("🟠 Border Points",  border_count)
col3.metric("🔴 Noise Points",   noise_count)
col4.metric("🔵 Clusters Found", n_clusters)

st.subheader("📋 Classification Table")
st.dataframe(df_result)

if label_col:
    st.subheader("🔍 Class vs Cluster Comparison")
    df_result["Class"] = df[label_col].values
    comparison = pd.crosstab(df_result["Class"], df_result["Cluster"], margins=True)
    comparison.columns = [f"Cluster {c}" if c != "All" else "Total" for c in comparison.columns]
    st.dataframe(comparison)

# ── K-Distance Plot ───────────────────────────────────────────────────
st.subheader("📐 K-Distance Plot (helps choose ε)")
st.caption("Find the 'elbow' point — that's your ideal ε value")
k = min_samples
nbrs_k = NearestNeighbors(n_neighbors=k).fit(X)
k_distances, _ = nbrs_k.kneighbors(X)
k_dist_sorted  = np.sort(k_distances[:, -1])[::-1]

fig_k, ax_k = plt.subplots(figsize=(8, 3))
ax_k.plot(k_dist_sorted, color="steelblue", linewidth=1.5)
ax_k.axhline(y=eps, color="red", linestyle="--", linewidth=1, label=f"Current ε = {eps}")
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

step_info = {
    1: {
        "label": "Step 1 — Raw Data",
        "icon": "📍",
        "description": (
            "**What you see:** The unprocessed dataset plotted as a scatter plot.\n\n"
            "This is the starting point — DBSCAN has not run yet. "
            "If your data has a class/label column, points are colored by their **true class** so you can later compare "
            "against DBSCAN's discovered clusters. \n\n"
            "👉 **What to look for:** Are there visible groupings? Are there isolated points far from clusters? "
            "This helps you form an intuition about what ε and MinPts values might work well."
        )
    },
    2: {
        "label": "Step 2 — ε Neighborhoods",
        "icon": "🌡️",
        "description": (
            "**What you see:** A **density heatmap** overlaid on the scatter plot.\n\n"
            "For every point in the space, we count how many data points fall within its ε-radius circle. "
            "Brighter/warmer regions = more neighbors = denser. "
            "The **cyan dashed contour** marks the MinPts threshold — points above this boundary "
            "are candidates to become Core points.\n\n"
            "The **highlighted circle** spotlights the densest point in the dataset with an ε-radius annotation.\n\n"
            "👉 **The side bar chart** shows the distribution of neighbor counts — "
            "red bars = below MinPts (potential noise/border), green bars = above MinPts (core candidates)."
        )
    },
    3: {
        "label": "Step 3 — Core / Border / Noise Classification",
        "icon": "🏷️",
        "description": (
            "**What you see:** Each point is now labeled as one of three types:\n\n"
            "- 🟢 **Core Point (circle):** Has ≥ MinPts neighbors within its ε-radius. "
            "These are the 'anchors' of clusters.\n"
            "- 🟠 **Border Point (square):** Fewer than MinPts neighbors, but lies within the ε-radius of a Core point. "
            "These are 'recruited' into clusters but cannot expand them.\n"
            "- 🔴 **Noise Point (X mark):** Not a Core, and not within ε of any Core. "
            "These are outliers.\n\n"
            "Dashed ε-circles are drawn around every point to show their neighborhood visually.\n\n"
            "👉 **Tip:** If you see too many red X marks, try increasing ε or decreasing MinPts. "
            "Too many Core points merging everything? Decrease ε or increase MinPts."
        )
    },
    4: {
        "label": "Step 4 — Cluster Expansion",
        "icon": "🌱",
        "description": (
            "**What you see:** The final cluster assignments after DBSCAN's expansion phase.\n\n"
            "DBSCAN works by picking an unvisited Core point and recursively adding all density-reachable "
            "points into the same cluster:\n"
            "1. Start from a Core point\n"
            "2. Add all neighbors within ε\n"
            "3. For each new Core neighbor, repeat step 2\n"
            "4. When no more Core points can be added, the cluster is complete\n"
            "5. Pick the next unvisited Core point and start a new cluster\n\n"
            "Each cluster gets a **distinct color**. Noise points remain **red X marks**.\n\n"
            "👉 **What to look for:** Are the clusters separated cleanly? "
            "Clusters sharing the same color should belong to the same dense region."
        )
    },
    5: {
        "label": "Step 5 — Cluster Report Card",
        "icon": "📊",
        "description": (
            "**What you see:** A detailed **per-cluster dashboard** with tabs for each cluster.\n\n"
            "Each tab shows:\n"
            "- 📍 **Spatial scatter** with centroid crosshair and ε-circles for Core points\n"
            "- 📊 **Composition bar** — breakdown of Core, Border, and Noise within the cluster\n"
            "- 📏 **Density bar** — this cluster's size relative to the largest cluster\n"
            "- 🟢/🟡/🔴 **Quality verdict** based on the core-to-total ratio\n\n"
            "👉 **Quality verdicts:**\n"
            "- 🟢 > 70% core ratio → Dense, well-defined cluster\n"
            "- 🟡 30–70% core ratio → Moderate cluster\n"
            "- 🔴 < 30% core ratio → Sparse — consider tuning parameters"
        )
    },
}

step = st.slider("Step", 1, 5, 1)

# ── Detailed Step Description Panel ──────────────────────────────────
info = step_info[step]
st.markdown(f"""
<div style="background:#1e293b; border-left:4px solid #e94560;
            border-radius:8px; padding:1rem 1.2rem; margin-bottom:1rem;">
  <h4 style="color:#e94560; margin:0 0 0.4rem 0;">{info['icon']} {info['label']}</h4>
  <div style="color:#cbd5e1; font-size:0.9rem; line-height:1.7;">{info['description'].replace(chr(10), '<br>')}</div>
</div>
""", unsafe_allow_html=True)

show_annotations = len(X) <= 30
show_circles     = len(X) <= 100

color_map  = {"Core": "#2ecc71", "Border": "#f39c12", "Noise": "#e74c3c"}
marker_map = {"Core": "o",       "Border": "s",       "Noise": "X"}
size_map   = {"Core": 70,        "Border": 55,        "Noise": 55}

# ── Steps 1–4 ─────────────────────────────────────────────────────────
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
                ax.scatter(X[mask, 0], X[mask, 1],
                           color=cmap[idx % len(cmap)], s=30,
                           label=str(cls), alpha=0.7, zorder=5)
            ax.legend(title="Class", fontsize=8)
            ax.set_title(f"Step 1: Raw Data — colored by '{label_col}'")
        else:
            ax.scatter(X[:, 0], X[:, 1], color="steelblue", s=30, alpha=0.7, zorder=5)
            ax.set_title("Step 1: Raw Data — All Points")

        if show_annotations:
            for i, (xi, yi) in enumerate(X):
                ax.annotate(f"P{i+1}", (xi, yi),
                            textcoords="offset points", xytext=(5, 3), fontsize=7)
        ax.margins(0.1)

    # STEP 2 — ε Neighborhood Density Field
    elif step == 2:
        from scipy.spatial import cKDTree
        plt.close(fig)

        tree         = cKDTree(X)
        neighbor_cnt = np.array([len(tree.query_ball_point(p, eps)) - 1 for p in X])

        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        pad_x = max((x_max - x_min) * 0.12, 1.0)
        pad_y = max((y_max - y_min) * 0.12, 1.0)

        grid_res  = 100
        gx = np.linspace(x_min - pad_x, x_max + pad_x, grid_res)
        gy = np.linspace(y_min - pad_y, y_max + pad_y, grid_res)
        GX, GY   = np.meshgrid(gx, gy)
        grid_pts = np.c_[GX.ravel(), GY.ravel()]
        grid_counts = np.array([
            len(tree.query_ball_point(gp, eps)) for gp in grid_pts
        ]).reshape(grid_res, grid_res)

        fig2, (ax_main, ax_bar) = plt.subplots(
            1, 2, figsize=(12, 6),
            gridspec_kw={"width_ratios": [3, 1]}, facecolor="white"
        )
        fig2.subplots_adjust(wspace=0.35)

        cf = ax_main.contourf(GX, GY, grid_counts, levels=25, cmap="magma", alpha=0.75, zorder=1)
        cbar = fig2.colorbar(cf, ax=ax_main, shrink=0.88, pad=0.02)
        cbar.set_label("Neighbors within ε", fontsize=9, labelpad=8)
        cbar.ax.tick_params(labelsize=8)

        ax_main.contour(GX, GY, grid_counts, levels=[min_samples - 0.5],
                        colors=["#00e5ff"], linewidths=1.5, linestyles="--", zorder=2)
        ax_main.plot([], [], color="#00e5ff", linestyle="--",
                     linewidth=1.5, label=f"MinPts = {min_samples} threshold")

        sc = ax_main.scatter(X[:, 0], X[:, 1],
                             c=neighbor_cnt, cmap="cool",
                             vmin=0, vmax=neighbor_cnt.max(),
                             s=18, edgecolors="white", linewidths=0.2, alpha=0.95, zorder=5)

        densest_idx = int(np.argmax(neighbor_cnt))
        dp = X[densest_idx]
        ax_main.add_patch(plt.Circle((dp[0], dp[1]), eps,
                                     fill=True, facecolor="#00e5ff", alpha=0.15,
                                     edgecolor="#00e5ff", linewidth=2.2, zorder=4))
        ax_main.annotate(f"  ε = {eps}",
                         xy=(dp[0] + eps, dp[1]),
                         xytext=(dp[0] + eps + pad_x * 0.6, dp[1]),
                         arrowprops=dict(arrowstyle="-|>", color="#00e5ff", lw=1.4, mutation_scale=10),
                         fontsize=9, color="#00e5ff", fontweight="bold", va="center")
        ax_main.scatter([dp[0]], [dp[1]], s=80, color="#00e5ff",
                        edgecolors="white", linewidths=1.2, zorder=6,
                        label=f"Densest ({int(neighbor_cnt[densest_idx])} neighbors)")

        ax_main.set_facecolor("#0d0d0d")
        ax_main.set_xlabel(x_col, fontsize=10)
        ax_main.set_ylabel(y_col, fontsize=10)
        ax_main.set_title(f"ε-Neighborhood Density Field   (ε = {eps})",
                          fontsize=12, fontweight="bold", pad=12)
        ax_main.legend(fontsize=8, loc="upper right",
                       framealpha=0.4, labelcolor="white", facecolor="#222")
        ax_main.tick_params(labelsize=8)
        ax_main.set_aspect("equal")
        ax_main.autoscale_view()

        bins       = np.arange(0, neighbor_cnt.max() + 2)
        counts_hist, edges = np.histogram(neighbor_cnt, bins=bins)
        bar_colors = ["#e74c3c" if b < min_samples else "#2ecc71" for b in edges[:-1]]
        ax_bar.barh(edges[:-1], counts_hist, color=bar_colors,
                    edgecolor="white", linewidth=0.3, height=0.8, align="edge")
        ax_bar.axhline(min_samples - 0.5, color="#f39c12", linewidth=1.5, linestyle="--")
        ax_bar.text(
            ax_bar.get_xlim()[1] if counts_hist.max() == 0 else counts_hist.max() * 0.02,
            min_samples, f"  MinPts = {min_samples}",
            color="#f39c12", fontsize=8, va="bottom"
        )
        ax_bar.set_xlabel("# of points", fontsize=9)
        ax_bar.set_ylabel("Neighbor count", fontsize=9)
        ax_bar.set_title("Distribution", fontsize=10, fontweight="bold")
        ax_bar.tick_params(labelsize=8)
        ax_bar.legend(
            handles=[
                mpatches.Patch(color="#e74c3c", label="< MinPts (noise/border)"),
                mpatches.Patch(color="#2ecc71", label="≥ MinPts (core candidate)"),
            ],
            fontsize=7, loc="lower right"
        )
        ax_bar.grid(axis="x", linestyle="--", alpha=0.3)

        st.pyplot(fig2)
        plt.close(fig2)
        ax.set_visible(False)

    # STEP 3 — Core / Border / Noise
    elif step == 3:
        if show_circles:
            circle_colors = {"Core": "#2ecc71", "Border": "#f39c12", "Noise": "#e74c3c"}
            circle_fills  = {"Core": 0.07,      "Border": 0.04,      "Noise": 0.03}
            for i, p in enumerate(X):
                ptype  = point_types[i]
                circle = plt.Circle((p[0], p[1]), eps,
                                    fill=True, facecolor=circle_colors[ptype],
                                    alpha=circle_fills[ptype],
                                    edgecolor=circle_colors[ptype],
                                    linewidth=0.7, linestyle="--")
                ax.add_patch(circle)
            ax.set_aspect("equal")
            ax.autoscale_view()

        for ptype in ["Core", "Border", "Noise"]:
            idxs = [i for i, t in enumerate(point_types) if t == ptype]
            if idxs:
                pts_sub = X[idxs]
                ax.scatter(pts_sub[:, 0], pts_sub[:, 1],
                           color=color_map[ptype], marker=marker_map[ptype],
                           s=size_map[ptype], label=f"{ptype} ({len(idxs)})",
                           zorder=5, edgecolors="black", linewidths=0.4, alpha=0.9)

        if show_annotations:
            for i, (xi, yi) in enumerate(X):
                nbr_count = len(indices[i])
                ax.annotate(f"{nbr_count}pts", (xi, yi),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=6.5, color="#555")

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
                ax.scatter(pts_sub[:, 0], pts_sub[:, 1],
                           color="#e74c3c", marker="X",
                           s=40, label="Noise", zorder=5, alpha=0.7)
            else:
                ax.scatter(pts_sub[:, 0], pts_sub[:, 1],
                           color=colors[lbl % len(colors)],
                           s=40, label=f"Cluster {lbl}", zorder=5, alpha=0.7)
        ax.legend(title="Clusters", fontsize=8, loc="upper right", ncol=2)
        ax.set_title("Step 4: Cluster Expansion — core points recruit their neighbors")
        ax.margins(0.1)

    if step != 2:
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ── STEP 5 — Cluster Report Card ──────────────────────────────────────
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

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Points",   total)
                m2.metric("Core",     c_core)
                m3.metric("Border",   c_border)
                m4.metric("X-span / Y-span", f"{x_span:.2f} / {y_span:.2f}")

                left, right = st.columns([2, 1])

                with left:
                    fig_s, ax_s = plt.subplots(figsize=(5, 4))
                    ax_s.scatter(X[~mask, 0], X[~mask, 1],
                                 color="#cccccc", s=15, alpha=0.3, zorder=2, label="Other")

                    if show_circles:
                        for i in range(len(X)):
                            if mask[i] and point_types[i] == "Core":
                                circ = plt.Circle((X[i, 0], X[i, 1]), eps,
                                                  fill=True, facecolor=tab_color,
                                                  alpha=0.06, edgecolor=tab_color, linewidth=0.6)
                                ax_s.add_patch(circ)
                        ax_s.set_aspect("equal")
                        ax_s.autoscale_view()

                    for ptype in ["Core", "Border", "Noise"]:
                        sub_idx = [i for i in range(len(X)) if mask[i] and point_types[i] == ptype]
                        if sub_idx:
                            ax_s.scatter(X[sub_idx, 0], X[sub_idx, 1],
                                         color=color_map[ptype], marker=marker_map[ptype],
                                         s=size_map[ptype], label=ptype,
                                         zorder=5, edgecolors="black", linewidths=0.3)

                    ax_s.axvline(cx, color=tab_color, linewidth=0.8, linestyle=":", alpha=0.7)
                    ax_s.axhline(cy, color=tab_color, linewidth=0.8, linestyle=":", alpha=0.7)
                    ax_s.scatter([cx], [cy], marker="+", color=tab_color, s=120,
                                 linewidths=1.5, zorder=6, label=f"Centroid ({cx:.1f}, {cy:.1f})")

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
                    bar_types  = ["Core", "Border", "Noise"]
                    bar_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
                    bar_counts = [c_core, c_border, c_noise]

                    left_val = 0
                    for bt, bc, bcount in zip(bar_types, bar_colors, bar_counts):
                        if bcount > 0:
                            ax_b.barh(0, bcount, left=left_val,
                                      color=bc, edgecolor="white", linewidth=0.5,
                                      height=0.4, label=bt)
                            ax_b.text(left_val + bcount / 2, 0, f"{bcount}",
                                      ha="center", va="center",
                                      fontsize=9, color="white", fontweight="bold")
                            left_val += bcount

                    max_cluster_size = max(np.sum(labels == c) for c in range(n_clusters)) if n_clusters > 0 else 1
                    density_pct = total / max_cluster_size
                    ax_b.barh(1, density_pct, color=tab_color, alpha=0.75, height=0.4)
                    ax_b.barh(1, 1, color="#eeeeee", alpha=0.4, height=0.4)
                    ax_b.text(density_pct / 2, 1, f"{density_pct * 100:.0f}%",
                              ha="center", va="center",
                              fontsize=9, color="white", fontweight="bold")

                    ax_b.set_yticks([0, 1])
                    ax_b.set_yticklabels(["Composition", "Density\nvs largest"], fontsize=9)
                    ax_b.set_xlim(0, max(1.05, density_pct + 0.05))
                    ax_b.set_title("Breakdown", fontsize=10)
                    ax_b.legend(fontsize=7, loc="lower right")
                    ax_b.grid(axis="x", linestyle="--", alpha=0.3)
                    ax_b.margins(0.1)
                    st.pyplot(fig_b)
                    plt.close(fig_b)

                if not is_noise_tab:
                    core_ratio = c_core / total if total else 0
                    if core_ratio > 0.7:
                        verdict = "🟢 Dense, well-defined cluster — high core ratio."
                    elif core_ratio > 0.3:
                        verdict = "🟡 Moderate cluster — mix of core and border points."
                    else:
                        verdict = "🔴 Sparse cluster — mostly border points, consider increasing MinPts."
                    st.info(verdict)

# ── Footer credits ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.8rem; padding:0.5rem;">
  🔬 <b>DBSCAN Interactive Visualizer</b> &nbsp;|&nbsp;
  Developed by <b style="color:#e94560;">Roohith R</b> &amp; <b style="color:#3b82f6;">Lakkshanth R</b>
  &nbsp;|&nbsp; Mentored by <b style="color:#10b981;">Swaminathan A</b>
</div>
""", unsafe_allow_html=True)
