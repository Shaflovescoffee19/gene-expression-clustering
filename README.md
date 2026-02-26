# 🧬 Gene Expression Clustering — Cancer Subtype Discovery

Gene expression data brings unique challenges that standard tabular ML datasets do not — thousands of features per sample, severe right-skew, and the need to find biologically meaningful structure rather than just mathematical clusters. This project builds the complete preprocessing and clustering pipeline used in real genomics research, from raw count normalisation through to the clustered heatmap that has become one of the most recognisable visualisations in molecular biology.

---

## 📌 Project Snapshot

| | |
|---|---|
| **Data** | Simulated RNA-Seq gene expression matrix |
| **Samples** | 150 patients (3 cancer subtypes, 50 each) |
| **Features** | 500 simulated gene expression values |
| **Task** | Discover cancer subtypes without using known labels |
| **Libraries** | `scikit-learn` · `scipy` · `seaborn` · `pandas` · `matplotlib` |

---

## 🗂️ The Data

Real RNA-Seq data consists of per-gene read counts measuring how actively each gene is being transcribed in a cell. The output is a matrix where rows are samples, columns are genes, and values are expression levels. This project simulates a realistic RNA-Seq matrix with three cancer subtypes, each defined by a distinct set of upregulated genes — mimicking the biological reality that different cancer subtypes activate different molecular pathways.

The simulation is intentionally realistic: counts follow a negative binomial distribution (as in real RNA-Seq), subtype signal is added to specific gene sets, and biological noise is present throughout.

---

## 🔧 Preprocessing Pipeline

### 1. Log Transformation
Raw RNA-Seq counts are highly right-skewed — a few highly expressed genes dominate. `log1p` (log(x+1)) transformation compresses the range and produces a more symmetric distribution suitable for distance-based algorithms. The +1 prevents log(0) = undefined for zero-count genes.

### 2. Variance-Based Feature Selection
Of 500 genes, only the 100 with highest variance across samples are retained. Low-variance genes are expressed similarly in all patients regardless of subtype — they add noise without signal. High-variance genes are the ones that discriminate between subtypes.

### 3. Standardisation + PCA
StandardScaler normalises each gene to mean 0, std 1. PCA then reduces 100 genes to the minimum components explaining 80% of total variance — removing remaining noise while preserving the structure needed for clustering.

---

## 🤖 Methods

### Hierarchical Clustering (Ward Linkage)
Bottom-up tree construction — starts with every patient as its own cluster, merges the most similar pair repeatedly. Ward linkage minimises within-cluster variance at each merge, producing compact clusters of roughly equal size. No K needs to be specified upfront — the dendrogram shows the full merge history and the natural cut point emerges from the data.

### K-Means Comparison
K-Means trained with K=3 on the same PCA-reduced features, for direct comparison against hierarchical clustering on the same evaluation metrics.

### Cluster Evaluation
| Method | Silhouette Score | Davies-Bouldin Index |
|--------|-----------------|----------------------|
| Hierarchical (Ward) | *see output* | *see output* |
| K-Means (K=3) | *see output* | *see output* |

### Differential Expression Analysis
For each discovered cluster, genes are ranked by mean expression difference versus all other clusters. The top-ranking genes per cluster are the "marker genes" — the molecular signatures that define each subtype.

---

## 📈 Visualisations Generated

| File | Description |
|------|-------------|
| `plot1_raw_data.png` | Raw count distribution, library sizes, gene variance profile |
| `plot2_log_transform.png` | Before vs after log transformation |
| `plot3_variance_selection.png` | Gene variance profile with selection cutoff |
| `plot4_pca_scree.png` | Variance explained per component + cumulative |
| `plot5_pca_scatter.png` | Patients in PCA space coloured by known subtype |
| `plot6_dendrogram.png` | Hierarchical clustering dendrogram with cut point |
| `plot7_heatmap.png` | Clustered heatmap — patients × genes with cluster annotations |
| `plot8_marker_genes.png` | Top differentially expressed genes per cluster |

---

## 🔍 Key Findings

PCA separates the three subtypes clearly in 2D — PC1 vs PC2 shows well-defined clusters with minimal overlap, confirming that the subtype signal is strong relative to noise. The hierarchical clustering dendrogram shows a natural three-way split at a clear height threshold. The clustered heatmap shows distinct expression blocks per subtype — the red (high expression) and blue (low expression) blocks are biologically interpretable as pathway activations specific to each subtype.

The marker gene analysis correctly recovers the simulated subtype-defining genes, validating that the pipeline finds genuine signal rather than artefactual clusters.

---

## 📂 Repository Structure

```
gene-expression-clustering/
├── gene_expression_clustering.py
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

## ⚙️ Setup

```bash
git clone https://github.com/Shaflovescoffee19/gene-expression-clustering.git
cd gene-expression-clustering
pip3 install scikit-learn scipy seaborn pandas matplotlib numpy
python3 gene_expression_clustering.py
```

---

## 📚 Skills Developed

- Understanding RNA-Seq data structure and why it requires specialised preprocessing
- Log transformation for right-skewed count data — intuition and implementation
- Variance-based feature selection for high-dimensional genomic data
- Hierarchical clustering with Ward linkage — algorithm, dendrogram reading, cut point selection
- The difference between hierarchical and K-Means clustering — when to use each
- Clustered heatmap construction and biological interpretation
- Differential expression analysis — identifying marker genes per cluster
- Davies-Bouldin Index as a complement to Silhouette Score for cluster evaluation

---

## 🗺️ Learning Roadmap

**Project 6 of 10** — a structured series building from data exploration through to advanced ML techniques.

| # | Project | Focus |
|---|---------|-------|
| 1 | Heart Disease EDA | Exploratory analysis, visualisation |
| 2 | Diabetes Data Cleaning | Missing data, outliers, feature engineering |
| 3 | Cancer Risk Classification | Supervised learning, model comparison |
| 4 | Survival Analysis | Time-to-event modelling, Cox regression |
| 5 | Customer Segmentation | Clustering, unsupervised learning |
| 6 | **Gene Expression Clustering** ← | High-dimensional data, heatmaps |
| 7 | Explainable AI with SHAP | Model interpretability |
| 8 | Counterfactual Explanations | Actionable predictions |
| 9 | Multi-Modal Data Fusion | Stacking, ensemble methods |
| 10 | Transfer Learning | Neural networks, domain adaptation |
