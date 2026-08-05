"""
OncoPulse — Stage 5: Interactive Clinical Dashboard
=====================================================
A Streamlit web app that brings together all pipeline outputs
into an interactive, shareable dashboard.

Screens:
  1. Overview     — dataset summary cards + UMAP embedding
  2. Biomarkers   — interactive volcano plot + top-gene table
  3. ML Models    — ROC curves, confusion matrix, SHAP
  4. Survival     — Kaplan-Meier curves filtered by cancer type
  5. Patient View — query any sample ID for its risk profile

Run:
    pip install streamlit plotly pandas numpy
    streamlit run stage5_dashboard.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ── PATHS ──────────────────────────────────────────────────────────────────────
DATA_DIR   = "oncopulse_data"
PLOTS_DIR  = "oncopulse_plots"

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OncoPulse — Multi-Cancer Biomarker Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

CANCER_PALETTE = {
    "BRCA": "#E07B54", "LUAD": "#5A9BD4",
    "COAD": "#6BBF6B", "GBM": "#C05CA0", "PRAD": "#F0C040",
}

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.title("🧬 OncoPulse")
st.sidebar.markdown("**Multi-Cancer Transcriptomic Biomarker Engine**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔬 Biomarker Discovery", "🤖 ML Models", "📈 Survival Analysis", "🔍 Patient View"]
)

# ── CACHED DATA LOADERS ────────────────────────────────────────────────────────
@st.cache_data
def load_expression():
    path = os.path.join(DATA_DIR, "processed_log2cpm.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None

@st.cache_data
def load_metadata():
    path = os.path.join(DATA_DIR, "metadata_clean.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_data
def load_biomarkers():
    path = os.path.join(DATA_DIR, "master_biomarkers.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_umap():
    path = os.path.join(DATA_DIR, "umap_coords.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_data
def load_pca():
    path = os.path.join(DATA_DIR, "pca_coords.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_data
def load_dea(cancer_type):
    path = os.path.join(DATA_DIR, f"dea_{cancer_type}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_risk_meta():
    path = os.path.join(DATA_DIR, "metadata_with_risk.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None

def show_plot_or_missing(path: str, caption: str = ""):
    if os.path.exists(path):
        img = Image.open(path)
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.info(f"Plot not yet generated: `{os.path.basename(path)}`  \nRun the pipeline stages first.")


# ── PAGE 1: OVERVIEW ──────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("OncoPulse — Multi-Cancer Transcriptomic Overview")
    st.markdown("Data source: **TCGA-style RNA-seq** | Analysis: **Python bioinformatics pipeline**")

    expr = load_expression()
    meta = load_metadata()

    if expr is not None and meta is not None:
        # Summary cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total samples",    f"{expr.shape[1]:,}")
        c2.metric("Genes profiled",   f"{expr.shape[0]:,}")
        c3.metric("Cancer types",     meta["cancer_type"].nunique() if "cancer_type" in meta.columns else "–")
        c4.metric("Median expression",f"{expr.values.mean():.2f} log₂CPM")

        st.markdown("---")

        # Plotly UMAP
        umap_df = load_umap()
        if umap_df is not None:
            umap_df = umap_df.join(meta[["cancer_type"]])
            fig = px.scatter(
                umap_df, x="UMAP1", y="UMAP2",
                color="cancer_type",
                color_discrete_map=CANCER_PALETTE,
                title="UMAP embedding — transcriptomic landscape",
                hover_name=umap_df.index,
                opacity=0.75,
                width=900, height=550
            )
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(legend_title_text="Cancer type",
                              plot_bgcolor="white",
                              paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run `stage2_preprocessing.py` to generate UMAP coordinates.")

        # Cancer type distribution
        st.subheader("Sample distribution by cancer type")
        if "cancer_type" in meta.columns:
            counts = meta["cancer_type"].value_counts().reset_index()
            counts.columns = ["Cancer type", "Samples"]
            fig2 = px.bar(counts, x="Cancer type", y="Samples",
                          color="Cancer type",
                          color_discrete_map=CANCER_PALETTE,
                          text="Samples")
            fig2.update_layout(showlegend=False, plot_bgcolor="white",
                               paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Run `stage1_data_ingestion.py` and `stage2_preprocessing.py` first.")


# ── PAGE 2: BIOMARKER DISCOVERY ───────────────────────────────────────────────
elif page == "🔬 Biomarker Discovery":
    st.title("Biomarker Discovery")
    st.markdown("Interactive volcano plots + top biomarker gene table")

    meta = load_metadata()
    biomarkers = load_biomarkers()

    if meta is not None and "cancer_type" in meta.columns:
        cancer_types = sorted(meta["cancer_type"].unique().tolist())
        selected = st.selectbox("Select cancer type", cancer_types)

        dea = load_dea(selected)
        if dea is not None:
            dea["-log10p"] = -np.log10(dea["pval"] + 1e-300)

            # Significance colour
            dea["color"] = dea["significance"].map({"Up": "#E07B54", "Down": "#5A9BD4", "NS": "#CCCCCC"})

            fig = go.Figure()
            for sig, color in [("NS", "#CCCCCC"), ("Down", "#5A9BD4"), ("Up", "#E07B54")]:
                mask = dea["significance"] == sig
                fig.add_trace(go.Scatter(
                    x=dea.loc[mask, "log2FC"],
                    y=dea.loc[mask, "-log10p"],
                    mode="markers",
                    marker=dict(color=color, size=5, opacity=0.65),
                    name=sig,
                    text=dea.loc[mask, "gene"],
                    hovertemplate="<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10p: %{y:.3f}<extra></extra>"
                ))

            fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray", line_width=0.8)
            fig.add_vline(x=1,  line_dash="dash", line_color="gray", line_width=0.8)
            fig.add_vline(x=-1, line_dash="dash", line_color="gray", line_width=0.8)

            fig.update_layout(
                title=f"Volcano plot — {selected} differential expression",
                xaxis_title="log₂ Fold-Change",
                yaxis_title="-log₁₀(p-value)",
                plot_bgcolor="white", paper_bgcolor="white",
                height=520
            )
            st.plotly_chart(fig, use_container_width=True)

            # Top significant genes table
            sig_genes = dea[dea["significance"] != "NS"].sort_values("padj").head(20)
            st.subheader(f"Top 20 significant genes in {selected}")
            st.dataframe(
                sig_genes[["gene", "log2FC", "pval", "padj", "significance"]].round(4),
                use_container_width=True
            )
        else:
            st.info(f"Run `stage3_biomarker_discovery.py` to generate DEA for {selected}.")

    # Master biomarker table
    if biomarkers is not None:
        st.subheader("Master biomarker list (all cancer types)")
        st.markdown("Ranked by combined evidence: RF importance + DEA significance")
        st.dataframe(biomarkers.head(50).round(4), use_container_width=True)
        st.download_button("⬇ Download biomarker list",
                           biomarkers.to_csv(index=False),
                           "oncopulse_biomarkers.csv", "text/csv")


# ── PAGE 3: ML MODELS ─────────────────────────────────────────────────────────
elif page == "🤖 ML Models":
    st.title("Machine Learning Models")

    tab1, tab2, tab3 = st.tabs(["SVM ROC Curves", "ANN Confusion Matrix", "SHAP Explainability"])

    with tab1:
        show_plot_or_missing(os.path.join(PLOTS_DIR, "svm_roc_curves.png"),
                             "SVM ROC curves — 5-fold cross-validation")
        st.markdown("""
**Model:** Support Vector Machine (RBF kernel)  
**Setup:** One-vs-Rest multi-class ROC, 5-fold stratified cross-validation  
**Input features:** Top 100 biomarker genes from Random Forest
        """)

    with tab2:
        show_plot_or_missing(os.path.join(PLOTS_DIR, "ann_confusion_matrix.png"),
                             "ANN confusion matrix (normalised)")
        st.markdown("""
**Architecture:** 256 → 128 → 64 → softmax (ReLU activations, Adam optimiser)  
**Regularisation:** L2 weight decay + early stopping  
**Training:** 5-fold cross-validation, balanced class weights
        """)

    with tab3:
        meta = load_metadata()
        if meta is not None and "cancer_type" in meta.columns:
            selected_cancer = st.selectbox("Cancer type for SHAP", sorted(meta["cancer_type"].unique()))
            show_plot_or_missing(
                os.path.join(PLOTS_DIR, f"shap_{selected_cancer}.png"),
                f"SHAP feature importance — {selected_cancer}"
            )
        st.markdown("""
**Method:** TreeExplainer (exact SHAP values for tree-based models)  
**Interpretation:** Higher mean |SHAP| = gene has larger impact on classifying this cancer type
        """)


# ── PAGE 4: SURVIVAL ANALYSIS ─────────────────────────────────────────────────
elif page == "📈 Survival Analysis":
    st.title("Survival Risk Stratification")
    st.markdown("Kaplan-Meier curves by transcriptomic risk cluster")

    show_plot_or_missing(
        os.path.join(PLOTS_DIR, "kaplan_meier_risk_groups.png"),
        "Kaplan-Meier survival curves — risk clusters"
    )

    risk_meta = load_risk_meta()
    if risk_meta is not None and "survival_months" in risk_meta.columns:
        st.subheader("Survival summary by risk group")
        summary = risk_meta.groupby("risk_group")["survival_months"].agg(
            ["count", "median", "mean", "std"]
        ).round(2)
        summary.columns = ["Samples", "Median survival (mo)", "Mean survival (mo)", "Std dev"]
        st.dataframe(summary, use_container_width=True)

        # Interactive KM-style scatter
        st.subheader("Survival time distribution by risk group")
        fig = px.box(risk_meta, x="risk_group", y="survival_months",
                     color="risk_group",
                     points="all",
                     title="Survival time distribution per risk group")
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `stage4_ml_risk_model.py` to generate survival data.")


# ── PAGE 5: PATIENT VIEW ──────────────────────────────────────────────────────
elif page == "🔍 Patient View":
    st.title("Patient Risk Profile")
    st.markdown("Query any sample to view its expression profile and predicted risk")

    expr = load_expression()
    meta = load_metadata()
    risk_meta = load_risk_meta()
    biomarkers = load_biomarkers()

    if expr is not None and meta is not None:
        sample_ids = sorted(expr.columns.tolist())
        selected_sample = st.selectbox("Select sample ID", sample_ids)

        if selected_sample:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Sample metadata")
                sample_meta = meta.loc[selected_sample] if selected_sample in meta.index else None
                if sample_meta is not None:
                    for col, val in sample_meta.items():
                        st.metric(col.replace("_", " ").title(), str(val))

                if risk_meta is not None and selected_sample in risk_meta.index:
                    st.metric("Risk group", risk_meta.loc[selected_sample, "risk_group"])

            with col2:
                if biomarkers is not None:
                    top_genes = biomarkers["gene"].head(30).tolist()
                    available_genes = [g for g in top_genes if g in expr.index]
                    sample_expr = expr.loc[available_genes, selected_sample]

                    st.subheader("Top 30 biomarker expression profile")
                    fig = go.Figure(go.Bar(
                        x=sample_expr.values,
                        y=sample_expr.index,
                        orientation="h",
                        marker_color=[
                            "#E07B54" if v > sample_expr.median() else "#5A9BD4"
                            for v in sample_expr.values
                        ]
                    ))
                    fig.update_layout(
                        xaxis_title="log₂ CPM expression",
                        height=500,
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=120, r=20, t=20, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Run the pipeline stages first to generate data.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("""
**OncoPulse Pipeline**  
Built with Python, scikit-learn,  
Seurat concepts, Plotly & Streamlit

*Soumili Paul — Bioinformatics MSc*  
SPPU × NCL × CSIC Spain
""")
