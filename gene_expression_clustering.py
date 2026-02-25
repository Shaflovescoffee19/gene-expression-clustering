# ============================================================
# PROJECT 6: Gene Expression Clustering
# ============================================================
# WHAT THIS SCRIPT DOES:
#   1. Generates realistic gene expression data (RNA-Seq style)
#   2. Applies variance-based feature selection
#   3. Normalises expression data (log transformation)
#   4. Reduces dimensions with PCA
#   5. Discovers cancer subtypes with hierarchical clustering
#   6. Visualises with a clustered heatmap + dendrogram
#   7. Evaluates clusters with Silhouette + Davies-Bouldin
#   8. Identifies differentially expressed genes per cluster
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
np.random.seed(42)

# ===========================================================
# STEP 1: GENERATE REALISTIC GENE EXPRESSION DATA
# ===========================================================
# We simulate RNA-Seq data with 3 cancer subtypes.
# Real RNA-Seq data looks exactly like this:
#   - Counts are right-skewed (many low, few very high)
#   - Different subtypes have different gene signatures
#   - Lots of noise from biological and technical variation
#
# This approach is standard in computational biology —
# synthetic benchmarks are used to validate pipelines
# before applying to real patient data.

N_PATIENTS  = 150    # 50 per subtype
N_GENES     = 500    # Simulated gene features
N_SUBTYPES  = 3      # Cancer subtypes to discover

print("=" * 60)
print("STEP 1: GENERATING GENE EXPRESSION DATA")
print("=" * 60)
print(f"  Patients  : {N_PATIENTS} ({N_PATIENTS//N_SUBTYPES} per subtype)")
print(f"  Genes     : {N_GENES}")
print(f"  Subtypes  : {N_SUBTYPES}")
print()

# Base expression — random counts resembling RNA-Seq
base_expr = np.random.negative_binomial(5, 0.3, size=(N_PATIENTS, N_GENES)).astype(float)

# True subtype labels (used only for evaluation, not training)
true_labels = np.array([0]*50 + [1]*50 + [2]*50)

# Add subtype-specific signatures
# Subtype 0: high expression of genes 0-99 (e.g. proliferation pathway)
base_expr[:50, :100]   += np.random.poisson(30, size=(50, 100))

# Subtype 1: high expression of genes 100-199 (e.g. immune pathway)
base_expr[50:100, 100:200] += np.random.poisson(25, size=(50, 100))

# Subtype 2: high expression of genes 200-299 (e.g. metabolism pathway)
base_expr[100:, 200:300]   += np.random.poisson(20, size=(50, 100))

# Gene names — simulate realistic names
gene_names = [f"GENE_{i:04d}" for i in range(N_GENES)]
patient_ids = [f"PAT_{i:03d}" for i in range(N_PATIENTS)]
subtype_names = {0: "Subtype A", 1: "Subtype B", 2: "Subtype C"}

df_expr = pd.DataFrame(base_expr, index=patient_ids, columns=gene_names)

print(f"  Expression matrix shape: {df_expr.shape}")
print(f"  Value range: {df_expr.values.min():.0f} - {df_expr.values.max():.0f}")
print(f"  Mean expression: {df_expr.values.mean():.2f}")
print()

# ===========================================================
# STEP 2: VISUALISE RAW DATA DISTRIBUTION
# ===========================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Raw expression distribution (very right-skewed — typical RNA-Seq)
flat_values = df_expr.values.flatten()
axes[0].hist(flat_values[flat_values < 100], bins=50,
             color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].set_title("Raw Expression Distribution\n(Right-skewed — typical RNA-Seq)",
                  fontweight="bold")
axes[0].set_xlabel("Raw Count")
axes[0].set_ylabel("Frequency")

# Per-patient total counts (library size)
lib_sizes = df_expr.sum(axis=1)
axes[1].hist(lib_sizes, bins=20, color="#DD8452", edgecolor="white", alpha=0.85)
axes[1].set_title("Per-Patient Total Counts\n(Library Size Variation)",
                  fontweight="bold")
axes[1].set_xlabel("Total Counts per Patient")
axes[1].set_ylabel("Number of Patients")

# Gene expression variance distribution
gene_vars = df_expr.var(axis=0)
axes[2].hist(gene_vars, bins=50, color="#55A868", edgecolor="white", alpha=0.85)
axes[2].set_title("Gene Variance Distribution\n(Most genes low variance)",
                  fontweight="bold")
axes[2].set_xlabel("Variance Across Patients")
axes[2].set_ylabel("Number of Genes")

fig.suptitle("Raw Gene Expression Data Overview",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plot1_raw_data.png")
plt.close()
print("Saved: plot1_raw_data.png")

# ===========================================================
# STEP 3: PREPROCESSING — LOG TRANSFORMATION
# ===========================================================
# RNA-Seq counts are highly skewed — a few genes have
# thousands of counts while most have very few.
# Log transformation (log1p = log(x+1)) compresses the
# range and makes the distribution more normal-like,
# which is required for most statistical analyses.
# The +1 prevents log(0) = undefined.

print("=" * 60)
print("STEP 3: LOG TRANSFORMATION")
print("=" * 60)

df_log = np.log1p(df_expr)

print(f"  Before log — range: {df_expr.values.min():.1f} to {df_expr.values.max():.1f}")
print(f"  After  log — range: {df_log.values.min():.3f} to {df_log.values.max():.3f}")
print(f"  Distribution is now more symmetric and normal-like")
print()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df_expr.values.flatten()[df_expr.values.flatten() < 100],
             bins=50, color="#C44E52", edgecolor="white", alpha=0.85)
axes[0].set_title("Before Log Transform (skewed)", fontweight="bold")
axes[0].set_xlabel("Raw Expression")

axes[1].hist(df_log.values.flatten(), bins=50,
             color="#55A868", edgecolor="white", alpha=0.85)
axes[1].set_title("After Log Transform (more normal)", fontweight="bold")
axes[1].set_xlabel("Log(Count + 1)")

fig.suptitle("Effect of Log Transformation on Expression Distribution",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot2_log_transform.png")
plt.close()
print("Saved: plot2_log_transform.png")

# ===========================================================
# STEP 4: VARIANCE-BASED FEATURE SELECTION
# ===========================================================
# Keep only the top 100 most variable genes.
# Low-variance genes are "housekeeping genes" — essential
# for all cells regardless of cancer status — they add noise.
# High-variance genes are more likely to be differentially
# expressed between cancer subtypes.

print("=" * 60)
print("STEP 4: VARIANCE-BASED FEATURE SELECTION")
print("=" * 60)

TOP_GENES = 100
gene_variance = df_log.var(axis=0).sort_values(ascending=False)
top_genes = gene_variance.head(TOP_GENES).index.tolist()

df_selected = df_log[top_genes]

print(f"  Total genes         : {N_GENES}")
print(f"  Selected genes      : {TOP_GENES} (top {TOP_GENES/N_GENES*100:.0f}% most variable)")
print(f"  Removed genes       : {N_GENES - TOP_GENES} (low variance / uninformative)")
print(f"  Variance range kept : {gene_variance[TOP_GENES-1]:.3f} to {gene_variance[0]:.3f}")
print()

# Plot variance cutoff
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, N_GENES + 1), gene_variance.values,
        color="#4C72B0", linewidth=1.5, alpha=0.8)
ax.axvline(x=TOP_GENES, color="red", linestyle="--", linewidth=2,
           label=f"Cutoff: Top {TOP_GENES} genes selected")
ax.fill_between(range(1, TOP_GENES + 1),
                gene_variance.values[:TOP_GENES], alpha=0.3, color="#DD8452",
                label="Selected genes")
ax.set_xlabel("Gene Rank (by variance)", fontsize=12)
ax.set_ylabel("Variance Across Patients", fontsize=12)
ax.set_title("Gene Variance Profile — Selecting Top Variable Genes",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot3_variance_selection.png")
plt.close()
print("Saved: plot3_variance_selection.png")

# ===========================================================
# STEP 5: STANDARDISATION + PCA
# ===========================================================
# Standardise so no single gene dominates by scale.
# Then apply PCA to reduce 100 genes to top components.

print("=" * 60)
print("STEP 5: STANDARDISATION + PCA")
print("=" * 60)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# PCA — keep enough components to explain 80% variance
pca = PCA(n_components=0.80, random_state=42)
pca_coords = pca.fit_transform(df_scaled)

var_explained = pca.explained_variance_ratio_
cumvar = np.cumsum(var_explained)

print(f"  Original dimensions  : {TOP_GENES}")
print(f"  PCA components kept  : {pca.n_components_} (explaining 80% variance)")
print()
print("  Variance per component:")
for i, (v, cv) in enumerate(zip(var_explained[:10], cumvar[:10])):
    print(f"    PC{i+1:2d}: {v*100:5.1f}%  (cumulative: {cv*100:.1f}%)")
if pca.n_components_ > 10:
    print(f"    ... ({pca.n_components_ - 10} more components)")
print()

# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(range(1, min(21, len(var_explained)+1)),
            var_explained[:20] * 100,
            color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("Principal Component", fontsize=12)
axes[0].set_ylabel("Variance Explained (%)", fontsize=12)
axes[0].set_title("Scree Plot — Variance per Component",
                  fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)

axes[1].plot(range(1, len(cumvar)+1), cumvar * 100,
             "bo-", linewidth=2, markersize=5)
axes[1].axhline(y=80, color="red", linestyle="--",
                linewidth=1.5, label="80% threshold")
axes[1].set_xlabel("Number of Components", fontsize=12)
axes[1].set_ylabel("Cumulative Variance Explained (%)", fontsize=12)
axes[1].set_title("Cumulative Variance — How Many PCs to Keep?",
                  fontweight="bold")
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

fig.suptitle("PCA Analysis of Gene Expression Data",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot4_pca_scree.png")
plt.close()
print("Saved: plot4_pca_scree.png")

# ===========================================================
# STEP 6: PCA SCATTER — VISUALISE PATIENT CLUSTERS
# ===========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors_true = ["#4C72B0", "#DD8452", "#55A868"]
subtype_color = [colors_true[l] for l in true_labels]

# PC1 vs PC2 — coloured by TRUE subtype (for validation)
for subtype_id, (name, color) in enumerate(zip(
        ["Subtype A", "Subtype B", "Subtype C"], colors_true)):
    mask = true_labels == subtype_id
    axes[0].scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                    c=color, s=60, alpha=0.8,
                    edgecolors="white", linewidth=0.5,
                    label=f"{name} (n={mask.sum()})")

axes[0].set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", fontsize=11)
axes[0].set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)", fontsize=11)
axes[0].set_title("PCA — True Cancer Subtypes\n(for validation)",
                  fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# PC1 vs PC3
for subtype_id, (name, color) in enumerate(zip(
        ["Subtype A", "Subtype B", "Subtype C"], colors_true)):
    mask = true_labels == subtype_id
    axes[1].scatter(pca_coords[mask, 0], pca_coords[mask, 2],
                    c=color, s=60, alpha=0.8,
                    edgecolors="white", linewidth=0.5,
                    label=f"{name} (n={mask.sum()})")

axes[1].set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", fontsize=11)
axes[1].set_ylabel(f"PC3 ({var_explained[2]*100:.1f}% variance)", fontsize=11)
axes[1].set_title("PCA — PC1 vs PC3", fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

fig.suptitle("PCA Visualisation of Cancer Patient Gene Expression",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot5_pca_scatter.png")
plt.close()
print("Saved: plot5_pca_scatter.png")

# ===========================================================
# STEP 7: HIERARCHICAL CLUSTERING + DENDROGRAM
# ===========================================================

print("=" * 60)
print("STEP 7: HIERARCHICAL CLUSTERING")
print("=" * 60)

# Use Ward linkage on PCA coordinates
Z = linkage(pca_coords[:, :10], method="ward")

# Cut dendrogram at 3 clusters
hc_labels = fcluster(Z, t=3, criterion="maxclust") - 1

sil = silhouette_score(pca_coords[:, :10], hc_labels)
db  = davies_bouldin_score(pca_coords[:, :10], hc_labels)

print(f"  Linkage method     : Ward")
print(f"  Number of clusters : 3")
print(f"  Silhouette Score   : {sil:.4f} (higher = better, max 1.0)")
print(f"  Davies-Bouldin     : {db:.4f} (lower = better, min 0.0)")
print()
print("  Cluster sizes:")
for c in sorted(np.unique(hc_labels)):
    print(f"    Cluster {c}: {(hc_labels==c).sum()} patients")
print()

# Dendrogram
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
           leaf_rotation=90, leaf_font_size=9,
           color_threshold=Z[-2, 2],
           above_threshold_color="gray")
ax.set_xlabel("Patient (or cluster size)", fontsize=12)
ax.set_ylabel("Ward Distance (dissimilarity)", fontsize=12)
ax.set_title("Hierarchical Clustering Dendrogram\n(Cut at 3 clusters — each colour = one subtype)",
             fontsize=13, fontweight="bold")
ax.axhline(y=Z[-2, 2], color="red", linestyle="--",
           linewidth=2, label="Cut point (3 clusters)")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot6_dendrogram.png")
plt.close()
print("Saved: plot6_dendrogram.png")

# ===========================================================
# STEP 8: CLUSTERED HEATMAP
# ===========================================================
# The iconic genomics visualisation — patients x genes
# coloured by expression, with hierarchical clustering
# applied to both rows (patients) and columns (genes).

print("=" * 60)
print("STEP 8: CLUSTERED HEATMAP")
print("=" * 60)

# Use top 40 most variable genes for readable heatmap
heatmap_genes = gene_variance.head(40).index.tolist()
heatmap_data = df_log[heatmap_genes]

# Add cluster labels for row colours
cluster_palette = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}
row_colors = pd.Series(hc_labels, index=patient_ids).map(cluster_palette)

g = sns.clustermap(
    heatmap_data,
    row_colors=row_colors,
    col_cluster=True,
    row_cluster=True,
    method="ward",
    metric="euclidean",
    cmap="RdBu_r",
    center=heatmap_data.values.mean(),
    figsize=(16, 12),
    xticklabels=False,
    yticklabels=False,
    cbar_pos=(0.02, 0.8, 0.03, 0.15)
)

g.fig.suptitle("Clustered Heatmap — Gene Expression Across Cancer Patients\n"
               "(Rows=Patients, Columns=Genes, Colour=Expression Level)",
               fontsize=13, fontweight="bold", y=1.01)

# Add legend for cluster colours
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4C72B0", label="Cluster 0"),
    Patch(facecolor="#DD8452", label="Cluster 1"),
    Patch(facecolor="#55A868", label="Cluster 2")
]
g.ax_heatmap.legend(handles=legend_elements, loc="upper right",
                     bbox_to_anchor=(1.15, 1.1), fontsize=10)

g.savefig("plot7_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved: plot7_heatmap.png")

# ===========================================================
# STEP 9: DIFFERENTIAL EXPRESSION ANALYSIS
# ===========================================================
# For each cluster, find genes that are most highly expressed
# compared to all other patients — these are the "marker genes"
# that define each subtype.

print("=" * 60)
print("STEP 9: MARKER GENES PER CLUSTER")
print("=" * 60)

df_log_copy = df_log[top_genes].copy()
df_log_copy["Cluster"] = hc_labels

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for cluster_id, color in zip([0, 1, 2], colors_true):
    # Mean expression in this cluster vs others
    in_cluster  = df_log_copy[df_log_copy["Cluster"] == cluster_id].drop("Cluster", axis=1).mean()
    out_cluster = df_log_copy[df_log_copy["Cluster"] != cluster_id].drop("Cluster", axis=1).mean()
    fold_change = (in_cluster - out_cluster).sort_values(ascending=False)

    top_markers = fold_change.head(10)
    print(f"  Cluster {cluster_id} top marker genes:")
    for gene, fc in top_markers.items():
        print(f"    {gene}: fold change = {fc:.3f}")
    print()

    top_markers.plot(kind="barh", ax=axes[cluster_id],
                     color=color, edgecolor="white", alpha=0.9)
    axes[cluster_id].invert_yaxis()
    axes[cluster_id].set_title(f"Cluster {cluster_id} — Top Marker Genes",
                               fontweight="bold")
    axes[cluster_id].set_xlabel("Mean Expression Difference vs Other Clusters")
    axes[cluster_id].grid(axis="x", alpha=0.3)

fig.suptitle("Differential Expression — Marker Genes per Cluster",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot8_marker_genes.png")
plt.close()
print("Saved: plot8_marker_genes.png")

# ===========================================================
# STEP 10: COMPARE K-MEANS VS HIERARCHICAL
# ===========================================================

km_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(pca_coords[:, :10])

km_sil  = silhouette_score(pca_coords[:, :10], km_labels)
km_db   = davies_bouldin_score(pca_coords[:, :10], km_labels)
hc_sil  = silhouette_score(pca_coords[:, :10], hc_labels)
hc_db   = davies_bouldin_score(pca_coords[:, :10], hc_labels)

print("=" * 60)
print("STEP 10: K-MEANS vs HIERARCHICAL CLUSTERING")
print("=" * 60)
print(f"  {'Method':<25} {'Silhouette':>12} {'Davies-Bouldin':>15}")
print(f"  {'-'*25} {'-'*12} {'-'*15}")
print(f"  {'K-Means (K=3)':<25} {km_sil:>12.4f} {km_db:>15.4f}")
print(f"  {'Hierarchical (Ward)':<25} {hc_sil:>12.4f} {hc_db:>15.4f}")
print()
print("  Silhouette: higher is better | Davies-Bouldin: lower is better")
print()

# ===========================================================
# FINAL SUMMARY
# ===========================================================

print("=" * 60)
print("PROJECT 6 COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"  Patients analysed       : {N_PATIENTS}")
print(f"  Starting features       : {N_GENES} genes")
print(f"  After variance filter   : {TOP_GENES} genes")
print(f"  After PCA               : {pca.n_components_} components (80% variance)")
print()
print(f"  Hierarchical clustering :")
print(f"    Subtypes discovered   : 3")
print(f"    Silhouette Score      : {hc_sil:.4f}")
print(f"    Davies-Bouldin Index  : {hc_db:.4f}")
print()
print("  Pipeline:")
print("    Raw counts → Log transform → Variance selection")
print("    → Standardise → PCA → Hierarchical cluster → Heatmap")
print()
print("  8 plots saved.")
print("  Ready to push to GitHub!")
print("=" * 60)
