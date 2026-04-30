# DBSCAN Interactive Visualizer
### Web-based Interactive Simulation Tool

---

## Project Overview

**DBSCAN Interactive Visualizer** is a fully interactive, browser-accessible tool for understanding, visualising, and experimenting with the **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** algorithm. Built with **Streamlit** and powered by **scikit-learn**, it offers step-by-step visual walkthroughs, real-time parameter tuning, point-type classification, cluster analysis, and full result export — all in a single-page app with no setup beyond Python.

**Live App:** https://dbscan-visualizer-ycfrxf7rz58uvrsgy3efhn.streamlit.app/

---

## Project Structure

```
dbscan-visualizer/
│
├── app.py                  <- Main Streamlit app — DBSCAN logic, all pages & visualisations
├── requirements.txt        <- Python dependencies
├── large.csv               <- Sample dataset for quick testing
└── README.md               <- Project documentation
```

---

## Features

### Visualizer (Main Page)

| Feature | Description |
|---|---|
| **CSV Upload** | Upload any CSV dataset — columns auto-detected |
| **Random Data Generator** | Generate synthetic clustered data instantly |
| **Parameter Tuning** | Interactive sliders for epsilon and MinPts |
| **K-Distance Graph** | Plots k-nearest-neighbour distances to help choose the optimal epsilon value |
| **Run DBSCAN** | Runs scikit-learn's `DBSCAN` on your dataset in real time |
| **Point Classification** | Classifies every point as Core, Border, or Noise |
| **Cluster Summary Metrics** | Number of clusters found, noise count, silhouette score |
| **Class vs Cluster Comparison** | Side-by-side comparison if the dataset includes true class labels |

### Step-by-Step Walkthrough (6 Steps)

| Step | Description |
|---|---|
| **Step 1 — Raw Data** | Original dataset plotted; points coloured by true class (if available) |
| **Step 2 — Epsilon-Neighborhood Density Field** | Heatmap showing neighbour density; dashed contour marks the MinPts threshold |
| **Step 3 — Core / Border / Noise Classification** | Each point labelled and styled by type; epsilon-circles drawn per point |
| **Step 4 — Cluster Formation** | Density-connected Core points merged into labelled clusters |
| **Step 5 — Final Cluster Map** | Fully coloured cluster plot with noise highlighted separately |
| **Step 6 — Class vs Cluster Comparison** | Ground-truth labels vs DBSCAN output in a split view |

### Help & Theory Page

- Full concept definition of DBSCAN and its advantages over K-Means
- Explanation of epsilon and MinPts with tuning guidance
- Point-type reference table (Core, Border, Noise)
- Step-by-step visual guide for interpreting each chart
- Pros and cons of DBSCAN

### Download Results Page

- Download cluster assignments as a **CSV file**
- Export all visualisation plots as **PNG images**

### About / Credits Page

- Project description, tech stack summary, and team details

---

## How to Run

### Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
```

### Step 2 — Launch the Streamlit app

```bash
streamlit run app.py
```

The app opens automatically. Alternatively, access the live hosted version at:

https://dbscan-visualizer-ycfrxf7rz58uvrsgy3efhn.streamlit.app/

---

## How to Use

1. Open https://dbscan-visualizer-ycfrxf7rz58uvrsgy3efhn.streamlit.app/ in your browser
2. **Upload a CSV** or click **Generate Random Data** to start instantly
3. Use the **epsilon** and **MinPts** sliders to tune the algorithm
4. Consult the **K-Distance Graph** to pick an optimal epsilon value
5. Click **Run DBSCAN** — results appear in real time
6. Walk through **Steps 1–6** to watch the algorithm execute visually
7. Review the **Cluster Summary** metrics and point-type breakdown
8. If your dataset has class labels, compare them with DBSCAN's output
9. Head to **Download Results** to export your CSV and chart PNGs

---

## Input Data Format

- **File type:** CSV (`.csv`)
- **First row:** Column headers
- **Feature columns:** Any numeric columns (X, Y, or more)
- **Optional class column:** A categorical/integer column for ground-truth comparison
- **Sample file:** `large.csv` is included for quick testing

---

## Algorithm Parameters

| Parameter | Description | Guidance |
|---|---|---|
| **Epsilon** | Radius of the neighbourhood around each point | Use the K-Distance Graph — look for the elbow |
| **MinPts** | Minimum neighbours required within epsilon to form a Core point | Rule of thumb: MinPts >= dimensions + 1; for 2D start with 3-5 |

---

## Point Types

| Type | Symbol | Definition |
|---|---|---|
| **Core** | Circle | Has >= MinPts neighbours within epsilon |
| **Border** | Square | Fewer than MinPts neighbours, but within epsilon of a Core point |
| **Noise** | X mark | Neither Core nor Border — fully isolated |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **App Framework** | Python 3, Streamlit |
| **ML / Clustering** | scikit-learn (DBSCAN, NearestNeighbors) |
| **Data Handling** | Pandas, NumPy |
| **Visualisation** | Matplotlib |

---

## About DBSCAN

**DBSCAN** groups together points that are closely packed in space and marks isolated points as outliers (noise). Unlike K-Means, DBSCAN does not require specifying the number of clusters in advance, can find clusters of arbitrary shapes (not just spherical), and naturally handles noise and outliers. However, it struggles with datasets of varying density and is sensitive to the choice of epsilon and MinPts.

---

## References

- [DBSCAN — scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [Original DBSCAN Paper — Ester et al., 1996 (KDD)](https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf)
- [A Density-Based Algorithm for Discovering Clusters — Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)
- [Choosing Epsilon with the K-Distance Graph — Towards Data Science](https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc)

---

## Team

**Developed by**
- Roohith R
- Lakkshanth R

**Mentored by**
- Swaminathan A

---

*DBSCAN Interactive Visualizer — An educational tool for exploring density-based clustering, built for learners and practitioners alike.*
