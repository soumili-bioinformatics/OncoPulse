# OncoPulse 🧬
### Multi-Cancer Transcriptomic Biomarker Discovery & Risk Stratification Engine

> **CV tagline:** *"End-to-end Python bioinformatics pipeline for cancer biomarker discovery, integrating differential expression analysis, SHAP-explained machine learning, and survival risk stratification on 2,000+ gene × 300+ sample TCGA-style RNA-seq data."*

---

## What makes this project stand out

| Feature | Why it impresses hiring managers |
|---|---|
| Real public data (TCGA/GEO) | Not a toy dataset — clinically relevant |
| 5-stage modular pipeline | Shows software engineering discipline |
| DEA + ML + Survival | Three complementary analytical lenses |
| SHAP explainability | State-of-the-art interpretable AI |
| Interactive Streamlit dashboard | Instantly demoable in interviews |
| Kaplan-Meier + log-rank test | Clinical-grade statistical analysis |
| Connects to your MSc work | Extends single-cell skills to bulk RNA-seq |

---

## Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install all dependencies
pip install pandas numpy scipy matplotlib seaborn scikit-learn \
            statsmodels plotly streamlit GEOparse umap-learn \
            shap lifelines Pillow
```

---

## Quick start (simulated data — runs in ~2 min)

```bash
python run_pipeline.py simulate
streamlit run stage5_dashboard.py
```

## Real TCGA data (requires internet, ~30–60 min download)

```bash
python run_pipeline.py geo
streamlit run stage5_dashboard.py
```

---

## Project structure

```
oncopulse/
├── stage1_data_ingestion.py      # GEO/TCGA download + simulated data
├── stage2_preprocessing.py       # Normalisation, PCA, UMAP, heatmap
├── stage3_biomarker_discovery.py # DEA, Random Forest, SHAP
├── stage4_ml_risk_model.py       # SVM, ANN, Kaplan-Meier survival
├── stage5_dashboard.py           # Streamlit interactive dashboard
├── run_pipeline.py               # Master runner
├── oncopulse_data/               # Generated data files (parquet/csv)
└── oncopulse_plots/              # Generated plots (PNG)
```

---

## Pipeline overview

```
TCGA RNA-seq data
       ↓
[Stage 1] Data ingestion (GEOparse / simulated)
       ↓
[Stage 2] Log2-CPM normalisation · PCA · UMAP · heatmap
       ↓
[Stage 3] Differential expression · Random Forest · SHAP
       ↓                                   ↓
  Volcano plots              Top 100 biomarker genes
       ↓
[Stage 4] SVM (ROC/AUC) · ANN · K-Means survival clustering
       ↓
[Stage 5] Streamlit dashboard (interactive, shareable)
```

---

## Technologies used

| Category | Tools |
|---|---|
| Data | pandas, NumPy, GEOparse, pyarrow |
| Stats | scipy, statsmodels (FDR correction) |
| ML | scikit-learn (SVM, RF, MLP, KMeans) |
| Explainability | SHAP (TreeExplainer) |
| Survival | lifelines (Kaplan-Meier, log-rank) |
| Visualisation | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Dimensionality | UMAP, PCA |

---


*Built by Soumili Paul | MSc Bioinformatics, SPPU | soumili77paul@gmail.com*
