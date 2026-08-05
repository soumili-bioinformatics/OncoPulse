"""
OncoPulse — Stage 2: Preprocessing & Exploratory Data Analysis
===============================================================
Handles:
  - Log-normalisation (log2 CPM)
  - Zero-variance gene filtering
  - Outlier sample detection (IQR on library size)
  - Batch effect visualisation
  - PCA + UMAP for cancer-type separation
  - Correlation heatmap of top variable genes

Run after stage1_data_ingestion.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Optional UMAP (pip install umap-learn)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[warn] umap-learn not installed — UMAP step will be skipped")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "oncopulse_data"
OUTPUT_DIR = "oncopulse_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANCER_PALETTE = {
    "BRCA": "#E07B54",
    "LUAD": "#5A9BD4",
    "COAD": "#6BBF6B",
    "GBM":  "#C05CA0",
    "PRAD": "#F0C040",
}


# ── STEP 1: Load raw data ──────────────────────────────────────────────────────
def load_data():
    expr = pd.read_parquet(os.path.join(DATA_DIR, "raw_expression.parquet"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"), index_col=0)
    print(f"Loaded: {expr.shape[0]} genes × {expr.shape[1]} samples")
    return expr, meta


# ── STEP 2: Log2-CPM normalisation ────────────────────────────────────────────
def normalize_log2cpm(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Counts Per Million normalisation + log2 transform.
    Standard for bulk RNA-seq data — removes library-size bias.
    """
    lib_sizes = expr.sum(axis=0)                         # total counts per sample
    cpm = expr.div(lib_sizes, axis=1) * 1_000_000        # scale to per-million
    log2cpm = np.log2(cpm + 1)                           # +1 pseudo-count avoids log(0)
    print(f"✓ Log2-CPM normalised. Value range: [{log2cpm.min().min():.2f}, {log2cpm.max().max():.2f}]")
    return log2cpm


# ── STEP 3: Gene filtering ─────────────────────────────────────────────────────
def filter_low_variance_genes(log2cpm: pd.DataFrame,
                               min_variance_percentile: float = 20.0) -> pd.DataFrame:
    """
    Removes genes with variance in the bottom percentile — these carry no signal
    and just add noise to downstream models.
    """
    variances = log2cpm.var(axis=1)
    cutoff = np.percentile(variances, min_variance_percentile)
    filtered = log2cpm[variances > cutoff]
    print(f"✓ Gene filtering: {log2cpm.shape[0]} → {filtered.shape[0]} genes "
          f"(removed bottom {min_variance_percentile}% by variance)")
    return filtered


# ── STEP 4: Outlier sample detection ──────────────────────────────────────────
def detect_outlier_samples(log2cpm: pd.DataFrame, meta: pd.DataFrame,
                            iqr_factor: float = 3.0):
    """
    Flags samples whose mean expression is an outlier by IQR method.
    Returns cleaned expression and metadata with outliers removed.
    """
    sample_means = log2cpm.mean(axis=0)
    Q1, Q3 = sample_means.quantile(0.25), sample_means.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - iqr_factor * IQR, Q3 + iqr_factor * IQR
    is_outlier = (sample_means < lower) | (sample_means > upper)
    n_outliers = is_outlier.sum()

    if n_outliers > 0:
        print(f"[!] Detected {n_outliers} outlier samples — removing them")
        log2cpm = log2cpm.loc[:, ~is_outlier]
        meta = meta.loc[~is_outlier]
    else:
        print("✓ No outlier samples detected")
    return log2cpm, meta


# ── STEP 5: PCA ────────────────────────────────────────────────────────────────
def run_pca(log2cpm: pd.DataFrame, meta: pd.DataFrame, n_components: int = 50):
    """
    Runs PCA. Returns the fitted PCA object and the transformed coordinates.
    Plots PC1 vs PC2 coloured by cancer type.
    """
    X = log2cpm.T.values                        # samples × genes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)        # samples × PCs

    # ── Explained variance plot ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(range(1, 21), pca.explained_variance_ratio_[:20] * 100,
                color="#5A9BD4", edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Principal component", fontsize=11)
    axes[0].set_ylabel("Variance explained (%)", fontsize=11)
    axes[0].set_title("Scree plot — top 20 PCs", fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(1, 21))

    # ── PCA scatter coloured by cancer type ──
    cancer_col = "cancer_type" if "cancer_type" in meta.columns else meta.columns[0]
    for cancer, color in CANCER_PALETTE.items():
        mask = meta[cancer_col] == cancer
        if mask.any():
            axes[1].scatter(coords[mask, 0], coords[mask, 1],
                            c=color, label=cancer, alpha=0.75, s=28, edgecolors="none")

    pct = pca.explained_variance_ratio_
    axes[1].set_xlabel(f"PC1 ({pct[0]*100:.1f}%)", fontsize=11)
    axes[1].set_ylabel(f"PC2 ({pct[1]*100:.1f}%)", fontsize=11)
    axes[1].set_title("PCA — cancer type separation", fontsize=12, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pca_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ PCA plot saved → {path}")

    # Save PCA coordinates for downstream use
    pca_df = pd.DataFrame(coords[:, :10],
                          index=log2cpm.columns,
                          columns=[f"PC{i+1}" for i in range(10)])
    pca_df.to_csv(os.path.join(DATA_DIR, "pca_coords.csv"))
    return pca, coords


# ── STEP 6: UMAP ──────────────────────────────────────────────────────────────
def run_umap(log2cpm: pd.DataFrame, meta: pd.DataFrame):
    """
    Runs UMAP on top 50 PCs for fast, biologically meaningful embedding.
    UMAP handles non-linear structure that PCA misses.
    """
    if not UMAP_AVAILABLE:
        print("[skip] UMAP not available — install with: pip install umap-learn")
        return None

    # Use PCA-reduced data (50 dims) as input to UMAP for speed
    X = log2cpm.T.values
    X_pca = PCA(n_components=50, random_state=42).fit_transform(
        StandardScaler().fit_transform(X))

    reducer = umap.UMAP(n_neighbors=20, min_dist=0.3,
                        n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_pca)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    cancer_col = "cancer_type" if "cancer_type" in meta.columns else meta.columns[0]
    for cancer, color in CANCER_PALETTE.items():
        mask = meta[cancer_col] == cancer
        if mask.any():
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=color, label=cancer, alpha=0.8, s=22, edgecolors="none")

    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title("UMAP embedding — multi-cancer transcriptomes", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    path = os.path.join(OUTPUT_DIR, "umap_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ UMAP plot saved → {path}")

    umap_df = pd.DataFrame(embedding, index=log2cpm.columns, columns=["UMAP1", "UMAP2"])
    umap_df.to_csv(os.path.join(DATA_DIR, "umap_coords.csv"))
    return embedding


# ── STEP 7: Top-variable-gene heatmap ─────────────────────────────────────────
def plot_top_gene_heatmap(log2cpm: pd.DataFrame, meta: pd.DataFrame,
                           top_n: int = 50):
    """
    Heatmap of the top N most variable genes across all samples.
    Annotated by cancer type — visually shows gene signatures.
    """
    top_genes = log2cpm.var(axis=1).nlargest(top_n).index
    hm_data = log2cpm.loc[top_genes]

    cancer_col = "cancer_type" if "cancer_type" in meta.columns else meta.columns[0]
    col_colors = meta[cancer_col].map(CANCER_PALETTE).fillna("#AAAAAA")

    g = sns.clustermap(
        hm_data,
        col_colors=col_colors,
        cmap="RdBu_r",
        vmin=-2, vmax=2,
        z_score=0,          # z-score across rows (genes)
        linewidths=0,
        figsize=(12, 10),
        dendrogram_ratio=(0.15, 0.1),
        cbar_pos=(0.02, 0.8, 0.03, 0.18),
        xticklabels=False,
    )
    g.ax_heatmap.set_ylabel("Top 50 variable genes", fontsize=10)
    g.ax_heatmap.set_xlabel("Samples", fontsize=10)
    g.fig.suptitle("Gene expression heatmap — top 50 variable genes",
                   fontsize=13, fontweight="bold", y=1.01)

    # Legend
    patches = [mpatches.Patch(color=v, label=k) for k, v in CANCER_PALETTE.items()]
    g.ax_col_dendrogram.legend(handles=patches, loc="center", ncol=5,
                                fontsize=8, frameon=False)

    path = os.path.join(OUTPUT_DIR, "top_gene_heatmap.png")
    g.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Heatmap saved → {path}")


# ── MAIN ───────────────────────────────────────────────────────────────────────
def preprocess_pipeline():
    expr, meta = load_data()
    log2cpm = normalize_log2cpm(expr)
    log2cpm = filter_low_variance_genes(log2cpm)
    log2cpm, meta = detect_outlier_samples(log2cpm, meta)

    # Save processed matrix for downstream stages
    log2cpm.to_parquet(os.path.join(DATA_DIR, "processed_log2cpm.parquet"))
    meta.to_csv(os.path.join(DATA_DIR, "metadata_clean.csv"))
    print(f"\n✓ Processed matrix: {log2cpm.shape[0]} genes × {log2cpm.shape[1]} samples")

    run_pca(log2cpm, meta)
    run_umap(log2cpm, meta)
    plot_top_gene_heatmap(log2cpm, meta)

    print("\n[Stage 2 complete] All plots saved to:", OUTPUT_DIR)
    return log2cpm, meta


if __name__ == "__main__":
    preprocess_pipeline()
