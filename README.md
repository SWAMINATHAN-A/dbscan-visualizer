DBSCAN Interactive Visualizer

An interactive web application built using **Streamlit** to visualize and understand the DBSCAN clustering algorithm step-by-step.

---

Features

- Upload your own CSV dataset or generate random data
- Interactive DBSCAN parameter tuning (ε and MinPts)
- Step-by-step visualization of clustering process
- Core, Border, and Noise point classification
- Cluster analysis and summary metrics
- K-Distance graph to help choose optimal ε
- Class vs Cluster comparison (if labeled data provided)

Tech Stack

- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

Installation

Make sure you have Python installed (>= 3.8)

Install dependencies

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
