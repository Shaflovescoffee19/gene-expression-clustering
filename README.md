# 🧬 Gene Expression Clustering — Cancer Subtype Discovery

An unsupervised Machine Learning project that discovers cancer subtypes from simulated RNA-Seq gene expression data using hierarchical clustering, PCA, and clustered heatmaps. This is **Project 6 of 10** in my ML learning roadmap toward computational biology research.

---

## 📌 Project Overview

| Feature | Details |
|---|---|
| Data | Simulated RNA-Seq gene expression (150 patients × 500 genes) |
| Subtypes | 3 cancer subtypes with distinct gene signatures |
| Techniques | Log transformation, Variance selection, PCA, Hierarchical clustering, Heatmap |
| Evaluation | Silhouette Score, Davies-Bouldin Index |
| Libraries | `scikit-learn`, `scipy`, `seaborn`, `pandas`, `matplotlib` |

---

## 🧠 Key Concepts

### Gene Expression Data
RNA-Seq measures how actively each gene is being transcribed in a cell. The output is a matrix of counts — rows are patients, columns are genes, values are expression levels. Cancer cells show dramatically different expression patterns from healthy cells.

### Log Transformation
Raw RNA-Seq counts are highly right-skewed. Log(count + 1) transformation compresses extreme values and produces a more normal-like distribution required for statistical analysis.

### Variance-Based Feature Selection
Most genes are "housekeeping genes" — expressed similarly in all patients regardless of disease status. Selecting only the top most variable genes removes noise and retains the genes most likely to discriminate between subtypes.

### PCA for Dimensionality Reduction
500 genes → top 100 variable → PCA reduces to the minimum components explaining 80% of variance. This removes remaining noise and makes clustering more effective.

### Hierarchical Clustering (Ward Linkage)
Builds a tree of clusters bottom-up — starts with every patient as its own cluster, merges the most similar pair repeatedly. Ward linkage minimises within-cluster variance at each merge. The result is a dendrogram showing the full merge history.

### Clustered Heatmap
The iconic genomics visualisation — a colour-coded matrix of patients × genes with dendrograms on both axes. Red = high expression, Blue = low expression. Co-expressed gene blocks reveal biological pathways. Patient clusters reveal cancer subtypes.

### Davies-Bouldin Index
Complements Silhouette Score — measures the ratio of within-cluster scatter to between-cluster separation. Lower is better (minimum 0).

---

## 📊 Visualisations Generated

| Plot | What It Shows |
|---|---|
| Raw Data Overview | Count distribution, library sizes, gene variance |
| Log Transform | Before vs after transformation comparison |
| Variance Selection | Gene variance profile with selection cutoff |
| PCA Scree Plot | Variance explained per component + cumulative |
| PCA Scatter | Patients in PC space coloured by subtype |
| Dendrogram | Hierarchical clustering tree with cut point |
| Clustered Heatmap | Patients × genes with cluster annotations |
| Marker Genes | Top differentially expressed genes per cluster |

---

## 📂 Project Structure

```
gene-expression-clustering/
├── gene_expression_clustering.py    # Main script
├── plot1_raw_data.png
├── plot2_log_transform.png
├── plot3_variance_selection.png
├── plot4_pca_scree.png
├── plot5_pca_scatter.png
├── plot6_dendrogram.png
├── plot7_heatmap.png
├── plot8_marker_genes.png
└── README.md
```

---

## ⚙️ Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/Shaflovescoffee19/gene-expression-clustering.git
cd gene-expression-clustering
```

**2. Install dependencies**
```bash
pip3 install scikit-learn scipy seaborn pandas matplotlib numpy
```

**3. Run the script**
```bash
python3 gene_expression_clustering.py
```

---

## 🔬 Connection to Research Proposal

This project implements the core computational pipeline of **Aim 2** of a research proposal on colorectal cancer risk prediction in the Emirati population:

> *"Taxonomic composition and functional pathways will be profiled... Alpha metrics and beta metrics will reveal ecological diversity. CRC-associated microbial features will be identified through differential abundance testing"*

The complete pipeline — normalisation → variance selection → PCA → hierarchical clustering → heatmap → marker identification — is **identical** whether applied to gene expression data (this project) or microbiome taxa abundance data (the proposal). The mathematics is the same; only the biological interpretation changes.

---

## 📚 What I Learned

- What **RNA-Seq gene expression data** looks like and why it requires special preprocessing
- Why **log transformation** is essential for right-skewed count data
- How **variance-based feature selection** removes uninformative housekeeping genes
- How **hierarchical clustering** works and how to read a dendrogram
- The difference between **K-Means** (requires K upfront, parallel) and **Hierarchical** (no K needed, sequential)
- How to build and interpret a **clustered heatmap** — the standard genomics visualisation
- What **differential expression analysis** is and how marker genes define cancer subtypes
- How **Davies-Bouldin Index** complements Silhouette Score for cluster evaluation

---

## 🗺️ Part of My ML Learning Roadmap

| # | Project | Status |
|---|---|---|
| 1 | Heart Disease EDA | ✅ Complete |
| 2 | Diabetes Data Cleaning | ✅ Complete |
| 3 | Cancer Risk Classification | ✅ Complete |
| 4 | Survival Analysis | ✅ Complete |
| 5 | Customer Segmentation | ✅ Complete |
| 6 | Gene Expression Clustering | ✅ Complete |
| 7 | Explainable AI with SHAP | 🔜 Next |
| 8 | Counterfactual Explanations | ⏳ Upcoming |
| 9 | Multi-Modal Data Fusion | ⏳ Upcoming |
| 10 | Transfer Learning | ⏳ Upcoming |

---

## 🙋 Author

**Shaflovescoffee19** — building ML skills from scratch toward computational biology research.
