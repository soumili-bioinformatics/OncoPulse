"""
OncoPulse — Stage 3: Biomarker Discovery
==========================================
Uses three complementary methods to identify cancer biomarkers:

1. Differential Expression Analysis
   - Log2 fold-change (tumour vs all others)
   - Mann-Whitney U test with Benjamini-Hochberg FDR correction
   - Volcano plots per cancer type

2. Random Forest Feature Importance
   - Trains a multi-class RF to discriminate cancer types
   - Ranks genes by mean decrease in impurity

3. SHAP (SHapley Additive exPlanations)
   - Explains which genes drive each prediction
   - SHAP beeswarm + waterfall plots
   - Model-agnostic, publishable-quality insights

Run after stage2_preprocessing.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[warn] shap not installed — install with: pip install shap")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "oncopulse_data"
OUTPUT_DIR = "oncopulse_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANCER_PALETTE = {
    "BRCA": "#E07B54", "LUAD": "#5A9BD4",
    "COAD": "#6BBF6B", "GBM": "#C05CA0", "PRAD": "#F0C040",
}


# ── LOAD PROCESSED DATA ────────────────────────────────────────────────────────
def load_processed():
    log2cpm = pd.read_parquet(os.path.join(DATA_DIR, "processed_log2cpm.parquet"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata_clean.csv"), index_col=0)
    return log2cpm, meta


# ── DEA: Differential Expression ──────────────────────────────────────────────
def differential_expression(log2cpm: pd.DataFrame, meta: pd.DataFrame,
                              cancer_type: str, top_n: int = 20) -> pd.DataFrame:
    """
    One-vs-rest differential expression for a given cancer type.
    Returns a DataFrame with log2FC, p-value, adjusted p-value, and significance label.
    """
    cancer_col = "cancer_type"
    is_target = meta[cancer_col] == cancer_type
    group_a = log2cpm.loc[:, is_target].values       # target cancer samples
    group_b = log2cpm.loc[:, ~is_target].values       # all other samples

    # Log2 fold-change
    log2fc = np.mean(group_a, axis=1) - np.mean(group_b, axis=1)

    # Mann-Whitney U test (non-parametric, robust for RNA-seq)
    pvals = np.array([
        stats.mannwhitneyu(group_a[i], group_b[i], alternative="two-sided").pvalue
        for i in range(log2cpm.shape[0])
    ])

    # Benjamini-Hochberg FDR correction
    _, padj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    results = pd.DataFrame({
        "gene":    log2cpm.index,
        "log2FC":  log2fc,
        "pval":    pvals,
        "padj":    padj,
        "-log10p": -np.log10(pvals + 1e-300),
    })

    # Significance labelling
    results["significance"] = "NS"
    results.loc[(results["padj"] < 0.05) & (results["log2FC"] > 1),   "significance"] = "Up"
    results.loc[(results["padj"] < 0.05) & (results["log2FC"] < -1),  "significance"] = "Down"
    results = results.sort_values("padj")

    # Save
    results.to_csv(os.path.join(DATA_DIR, f"dea_{cancer_type}.csv"), index=False)
    return results


def volcano_plot(results: pd.DataFrame, cancer_type: str):
    """Classic volcano plot — log2FC on x, -log10(p) on y."""
    fig, ax = plt.subplots(figsize=(8, 6))

    color_map = {"Up": "#E07B54", "Down": "#5A9BD4", "NS": "#CCCCCC"}
    for sig, color in color_map.items():
        mask = results["significance"] == sig
        ax.scatter(results.loc[mask, "log2FC"],
                   results.loc[mask, "-log10p"],
                   c=color, alpha=0.5 if sig == "NS" else 0.8,
                   s=12, edgecolors="none", label=sig)

    # Label top 10 genes
    top10 = results[results["significance"].isin(["Up", "Down"])].head(10)
    for _, row in top10.iterrows():
        ax.annotate(row["gene"], (row["log2FC"], row["-log10p"]),
                    fontsize=7, alpha=0.85,
                    xytext=(4, 4), textcoords="offset points")

    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(1,  color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(-1, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xlabel("log₂ Fold-Change (tumour / rest)", fontsize=11)
    ax.set_ylabel("-log₁₀(p-value)", fontsize=11)
    ax.set_title(f"Volcano plot — {cancer_type} differential expression",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)

    n_up   = (results["significance"] == "Up").sum()
    n_down = (results["significance"] == "Down").sum()
    ax.text(0.02, 0.97, f"↑ {n_up} up   ↓ {n_down} down",
            transform=ax.transAxes, fontsize=9, va="top",
            color="#555555")

    path = os.path.join(OUTPUT_DIR, f"volcano_{cancer_type}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Volcano plot → {path}")


def run_all_dea(log2cpm, meta):
    """Runs DEA and volcano plots for all cancer types."""
    cancer_col = "cancer_type"
    cancer_types = meta[cancer_col].unique()
    all_top_genes = set()

    print("\n[Differential Expression Analysis]")
    for cancer in cancer_types:
        print(f"  Processing {cancer}...")
        results = differential_expression(log2cpm, meta, cancer)
        volcano_plot(results, cancer)
        sig = results[results["significance"] != "NS"]
        print(f"    → {len(sig)} significant genes (FDR < 0.05, |log2FC| > 1)")
        all_top_genes.update(sig.head(50)["gene"].tolist())

    return list(all_top_genes)


# ── RANDOM FOREST FEATURE IMPORTANCE ──────────────────────────────────────────
def train_random_forest(log2cpm: pd.DataFrame, meta: pd.DataFrame):
    """
    Trains a Random Forest classifier to distinguish cancer types.
    Uses cross-validation to get unbiased performance estimates.
    Returns the trained model, feature importance scores, and CV accuracy.
    """
    print("\n[Random Forest — Feature Importance]")
    cancer_col = "cancer_type"
    le = LabelEncoder()
    y = le.fit_transform(meta[cancer_col])
    X = log2cpm.T.values   # samples × genes

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"   # handles unequal class sizes
    )

    # 5-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    print(f"  Cross-validated F1 (macro): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit on full data for feature importance
    rf.fit(X, y)
    importance = pd.DataFrame({
        "gene":       log2cpm.index,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    importance.to_csv(os.path.join(DATA_DIR, "rf_feature_importance.csv"), index=False)

    # Plot top 30 genes
    top30 = importance.head(30)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top30["gene"][::-1], top30["importance"][::-1],
            color="#5A9BD4", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Mean decrease in impurity (MDI)", fontsize=11)
    ax.set_title("Top 30 cancer-discriminating genes — Random Forest", fontsize=12, fontweight="bold")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "rf_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Feature importance plot → {path}")

    return rf, importance, le


# ── SHAP EXPLAINABILITY ────────────────────────────────────────────────────────
def run_shap_analysis(rf, log2cpm: pd.DataFrame, meta: pd.DataFrame,
                       le: LabelEncoder, sample_n: int = 60):
    """
    Computes SHAP values for a subset of samples.
    Produces:
      - Beeswarm summary plot (shows distribution of SHAP values per gene)
      - Bar plot of mean |SHAP| per gene
    """
    if not SHAP_AVAILABLE:
        print("[skip] shap not available")
        return None

    print("\n[SHAP Explainability]")
    X = log2cpm.T.values
    X_sample = X[:sample_n]          # use a subset for speed

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)    # list of arrays, one per class

    # Use class 0 (e.g. BRCA) for visualisation; extend to all classes in practice
    cancer_names = le.classes_
    for i, cancer in enumerate(cancer_names):
        sv = shap_values[i] if isinstance(shap_values, list) else shap_values[:, :, i]

        fig, ax = plt.subplots(figsize=(9, 6))
        # Mean absolute SHAP per gene
        mean_shap = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_shap)[-20:]          # top 20 genes

        ax.barh(log2cpm.index[top_idx],
                mean_shap[top_idx],
                color=CANCER_PALETTE.get(cancer, "#5A9BD4"),
                edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(f"SHAP — top 20 genes for {cancer} classification",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"shap_{cancer}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ SHAP plot ({cancer}) → {path}")

    return shap_values


# ── COMPILE MASTER BIOMARKER LIST ─────────────────────────────────────────────
def compile_biomarker_list(log2cpm: pd.DataFrame, meta: pd.DataFrame,
                            rf_importance: pd.DataFrame) -> pd.DataFrame:
    """
    Merges RF importance with DEA results to produce a ranked biomarker table.
    This is the key deliverable — a list of genes with multi-evidence support.
    """
    cancer_col = "cancer_type"
    cancer_types = meta[cancer_col].unique()

    # Aggregate DEA across cancer types
    dea_frames = []
    for cancer in cancer_types:
        path = os.path.join(DATA_DIR, f"dea_{cancer}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["cancer_type"] = cancer
            dea_frames.append(df[df["significance"] != "NS"])

    if dea_frames:
        dea_all = pd.concat(dea_frames).groupby("gene").agg(
            n_cancers_sig=("cancer_type", "nunique"),
            mean_log2FC=("log2FC", "mean"),
            min_padj=("padj", "min")
        ).reset_index()
    else:
        dea_all = pd.DataFrame({"gene": rf_importance["gene"], "n_cancers_sig": 0,
                                 "mean_log2FC": 0, "min_padj": 1})

    # Merge with RF importance
    biomarkers = rf_importance.merge(dea_all, on="gene", how="left").fillna(0)
    biomarkers["evidence_score"] = (
        biomarkers["importance"] / biomarkers["importance"].max() * 0.5 +
        biomarkers["n_cancers_sig"] / max(biomarkers["n_cancers_sig"].max(), 1) * 0.5
    )
    biomarkers = biomarkers.sort_values("evidence_score", ascending=False)

    path = os.path.join(DATA_DIR, "master_biomarkers.csv")
    biomarkers.to_csv(path, index=False)
    print(f"\n✓ Master biomarker list saved → {path}")
    print(biomarkers.head(10).to_string(index=False))
    return biomarkers


# ── MAIN ───────────────────────────────────────────────────────────────────────
def biomarker_pipeline():
    log2cpm, meta = load_processed()

    run_all_dea(log2cpm, meta)
    rf, importance, le = train_random_forest(log2cpm, meta)
    run_shap_analysis(rf, log2cpm, meta, le)
    biomarkers = compile_biomarker_list(log2cpm, meta, importance)

    print("\n[Stage 3 complete]")
    return rf, importance, le, biomarkers


if __name__ == "__main__":
    biomarker_pipeline()
