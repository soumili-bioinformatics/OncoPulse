"""
OncoPulse — Stage 4: ML Risk Stratification
=============================================
Builds three complementary models:

1. SVM Cancer Type Classifier
   - Trained on top biomarker genes
   - ROC curves + AUC per class (OvR)

2. Artificial Neural Network (ANN) — via scikit-learn MLPClassifier
   - Multi-layer perceptron for cancer subtype prediction
   - Confusion matrix

3. Survival Risk Stratification
   - K-Means clustering on top-50 biomarker genes
   - Kaplan-Meier survival curves per cluster
   - Log-rank test for statistical significance

Run after stage3_biomarker_discovery.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_curve, auc, ConfusionMatrixDisplay,
                              classification_report, roc_auc_score)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

# Optional: lifelines for survival analysis
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("[warn] lifelines not installed — install with: pip install lifelines")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "oncopulse_data"
OUTPUT_DIR = "oncopulse_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANCER_PALETTE = {
    "BRCA": "#E07B54", "LUAD": "#5A9BD4",
    "COAD": "#6BBF6B", "GBM": "#C05CA0", "PRAD": "#F0C040",
}
CLUSTER_COLORS = ["#E07B54", "#5A9BD4", "#6BBF6B", "#C05CA0"]


# ── LOAD DATA ──────────────────────────────────────────────────────────────────
def load_data():
    log2cpm = pd.read_parquet(os.path.join(DATA_DIR, "processed_log2cpm.parquet"))
    meta    = pd.read_csv(os.path.join(DATA_DIR, "metadata_clean.csv"), index_col=0)
    biomarkers = pd.read_csv(os.path.join(DATA_DIR, "master_biomarkers.csv"))
    top_genes  = biomarkers["gene"].head(100).tolist()
    X = log2cpm.loc[log2cpm.index.isin(top_genes)].T
    return X, meta, top_genes


# ── MODEL 1: SVM with ROC Curves ──────────────────────────────────────────────
def train_svm_with_roc(X: pd.DataFrame, meta: pd.DataFrame):
    """
    SVM classifier (RBF kernel) with One-vs-Rest ROC curves.
    Multi-class ROC is plotted as micro and macro averages.
    """
    print("[SVM Classifier]")
    le  = LabelEncoder()
    y   = le.fit_transform(meta["cancer_type"])
    classes = le.classes_

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X.values)

    # Binarise labels for OvR ROC
    y_bin = label_binarize(y, classes=range(len(classes)))

    # SVM with probability estimates (needed for ROC)
    svm = SVC(kernel="rbf", C=10, gamma="scale",
              probability=True, random_state=42, class_weight="balanced")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = cross_val_predict(svm, X_sc, y, cv=skf, method="predict_proba")

    # ── ROC plot ──
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = list(CANCER_PALETTE.values())

    all_fpr = []
    all_tpr = []

    for i, cancer in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                label=f"{cancer} (AUC = {roc_auc:.3f})")
        all_fpr.append(fpr)
        all_tpr.append(tpr)

    # Micro-average ROC
    fpr_flat = np.concatenate(all_fpr)
    tpr_flat = np.concatenate(all_tpr)
    sort_idx = np.argsort(fpr_flat)
    micro_auc = auc(fpr_flat[sort_idx], tpr_flat[sort_idx])

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("SVM multi-class ROC curves (5-fold CV)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.text(0.55, 0.15, f"Micro-avg AUC: {micro_auc:.3f}",
            fontsize=9, color="#444444")

    path = os.path.join(OUTPUT_DIR, "svm_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ ROC curves saved → {path}")
    return svm, scaler, le


# ── MODEL 2: ANN Confusion Matrix ─────────────────────────────────────────────
def train_ann(X: pd.DataFrame, meta: pd.DataFrame):
    """
    Multi-layer perceptron (ANN) with 3 hidden layers.
    Architecture: input → 256 → 128 → 64 → softmax
    Produces confusion matrix and classification report.
    """
    print("\n[ANN Classifier]")
    le = LabelEncoder()
    y  = le.fit_transform(meta["cancer_type"])

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X.values)

    ann = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,            # L2 regularisation
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(ann, X_sc, y, cv=skf)

    # Classification report
    report = classification_report(y, y_pred, target_names=le.classes_)
    print("  Classification report (5-fold CV):\n", report)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y, y_pred, display_labels=le.classes_,
        cmap="Blues", ax=ax, colorbar=False, normalize="true"
    )
    ax.set_title("ANN confusion matrix — cancer type prediction (5-fold CV)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ann_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix saved → {path}")

    # Save model report
    with open(os.path.join(DATA_DIR, "ann_report.txt"), "w") as f:
        f.write(report)

    return ann


# ── MODEL 3: Survival Risk Stratification ─────────────────────────────────────
def survival_risk_stratification(log2cpm_full_path: str,
                                  meta: pd.DataFrame,
                                  top_genes: list,
                                  n_clusters: int = 3):
    """
    Clusters samples based on top biomarker gene expression.
    Plots Kaplan-Meier survival curves per risk cluster.
    Runs log-rank test between high-risk (cluster 0) and low-risk (cluster 1).

    Requires 'survival_months' and 'vital_status' columns in metadata.
    """
    if "survival_months" not in meta.columns or "vital_status" not in meta.columns:
        print("\n[Survival Analysis] Skipped — metadata lacks survival columns")
        return

    print("\n[Survival Risk Stratification]")
    log2cpm = pd.read_parquet(log2cpm_full_path)
    genes_present = [g for g in top_genes if g in log2cpm.index][:50]
    X_surv = log2cpm.loc[genes_present].T

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_surv.values)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_sc)
    meta = meta.copy()
    meta["risk_cluster"] = clusters

    # Map clusters to risk labels by median survival
    median_survival = meta.groupby("risk_cluster")["survival_months"].median()
    rank = median_survival.rank(ascending=False).astype(int)
    risk_labels = {c: f"Risk group {r}" for c, r in rank.items()}
    meta["risk_group"] = meta["risk_cluster"].map(risk_labels)

    # ── Kaplan-Meier plot ──
    if not LIFELINES_AVAILABLE:
        print("  [skip KM] lifelines not installed")
        return meta

    fig, ax = plt.subplots(figsize=(9, 6))
    group_labels = sorted(meta["risk_group"].unique())

    kmfs = []
    for i, label in enumerate(group_labels):
        mask  = meta["risk_group"] == label
        T     = meta.loc[mask, "survival_months"].values
        E     = (meta.loc[mask, "vital_status"] == "dead").astype(int).values

        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=E, label=label)
        kmf.plot_survival_function(ax=ax, color=CLUSTER_COLORS[i],
                                   linewidth=2.5, ci_alpha=0.12, ci_show=True)
        kmfs.append((T, E, label))

    # Log-rank test between first two groups
    T0, E0, _ = kmfs[0]
    T1, E1, _ = kmfs[1]
    result = logrank_test(T0, T1, event_observed_A=E0, event_observed_B=E1)
    p_val = result.p_value
    ax.text(0.6, 0.85, f"Log-rank p = {p_val:.4f}",
            transform=ax.transAxes, fontsize=10,
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.8))

    ax.set_xlabel("Survival time (months)", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.set_title(f"Kaplan-Meier survival curves by transcriptomic risk cluster\n"
                 f"(K-Means, k={n_clusters}, top-50 biomarker genes)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.legend(frameon=False, fontsize=9)

    path = os.path.join(OUTPUT_DIR, "kaplan_meier_risk_groups.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Kaplan-Meier plot saved → {path}")
    print(f"  Log-rank test p-value: {p_val:.6f}")

    meta.to_csv(os.path.join(DATA_DIR, "metadata_with_risk.csv"))
    return meta


# ── MAIN ───────────────────────────────────────────────────────────────────────
def ml_pipeline():
    X, meta, top_genes = load_data()
    svm, scaler, le = train_svm_with_roc(X, meta)
    ann = train_ann(X, meta)

    log2cpm_path = os.path.join(DATA_DIR, "processed_log2cpm.parquet")
    meta_risk = survival_risk_stratification(log2cpm_path, meta, top_genes, n_clusters=3)

    print("\n[Stage 4 complete] All models and plots saved.")
    return svm, ann, meta_risk


if __name__ == "__main__":
    ml_pipeline()
