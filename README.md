# Multiplex Immunofluorescence Analysis of Neoadjuvant Pembrolizumab-Treated cSCC

## Background

This repository contains the multiplex immunofluorescence (mIF) analysis code for a single-arm, phase II trial of neoadjuvant pembrolizumab in patients with high-risk resectable cutaneous squamous cell carcinoma (cSCC) ([NCT04808999](https://clinicaltrials.gov/study/NCT04808999)). Pre-treatment core biopsies and post-treatment surgical specimens were profiled with two mIF antibody panels: Panel 1 (CD20, CD68, CD8, PD-1, panCK, PD-L1) capturing lymphocyte, macrophage, and tumor populations, and Panel 2 (CD163, CD68, MHCII, CD11c, panCK, PD-L1) capturing macrophage subtypes and dendritic cells.

The notebooks reproduce the cell typing, univariate and combinatorial density analyses, recurrent cellular neighborhood identification, and spatial cell-to-cell distance analyses to delineate spatial cellular patterns specific to pathologic complete responders (pCR) and non-responders (pNR).

## Notebooks

### `cell_type_heatmaps_and_umaps.ipynb`
Generates UMAP projections and marker intensity heatmaps for cell typing validation across both antibody panels and timepoints.

- **Panel 1** (CD20, CD68, CD8, PD-1, panCK, PD-L1): UMAP colored by cell type and by individual marker intensity; hierarchically clustered heatmap of mean marker expression per cell type.
- **Panel 2** (CD163, CD68, MHCII, CD11c, panCK, PD-L1): Same as above. CK column is dropped from the Panel 2 heatmap to reduce redundancy with Panel 1.
- Generates plots for both **post-treatment** and **pre-treatment** specimens.
- UMAPs are subsampled to 1M cells for visualization; heatmaps use all cells.

### `density_analyses.ipynb`
Univariate and combinatorial density analysis comparing cell type densities between pCR and pNR specimens.

- **Post-treatment analysis**: Computes cell densities (cells/mm²) per WSI, runs one-sided Mann-Whitney U tests, and generates paired boxplots with jittered data points. Performs 3-fold cross-validated logistic regression using (1) all non-tumor cell types and (2) the top 5 features ranked by univariate AUC. Reports ROC curves and confusion matrices.
- **Pre-treatment analysis**: Same density and statistical workflow applied to pre-treatment core biopsies.
- Region area is calculated using a pixel-to-mm conversion factor of 0.4964 µm/px with 1000×1300 px regions.

### `neighborhood_analysis.ipynb`
Recurrent cellular neighborhood (RCN) identification and proportion-based comparison between pCR and pNR specimens.

- Loads pre-computed k-means neighborhood assignments (k=15, 80 µm radius, merged to 9 final RCNs).
- Generates stacked bar plots of cell type composition within each RCN.
- Computes per-WSI neighborhood proportions, excluding pNR-exclusive neighborhoods (RCNs 1, 5, 6) from the proportion denominator.
- Runs one-sided Wilcoxon rank-sum tests for enrichment of each RCN in pCR specimens.
- Produces paired boxplots of neighborhood proportions across response groups.

### `neighborhood_cell_to_cell_distance_analysis.ipynb`
Spatial analysis of cell-cell distances and neighbor counts within and around RCN 0 (lymphoid aggregates).

- Computes minimum distances from each MHCII⁺CD11c⁺ dendritic cell to the nearest CD20⁺ B cell and CD8⁺ T cell across all WSIs.
- Generates per-sample median distance boxplots and a log-distance histogram comparing B cell vs. T cell proximity to dendritic cells.
- Uses Squidpy to build spatial neighbor graphs (100 µm radius) and counts CD20⁺ and CD8⁺ cells within the neighborhood of each MHCII⁺CD11c⁺ cell (restricted to RCN 0).
- Produces per-sample median neighbor count comparisons.

## Data

All notebooks expect data in `h5ad` (AnnData) or `csv` format located at the path specified by `cell_type_dir`. Key input files:

| File | Description |
|------|-------------|
| `panel_1_cell_data.h5ad` | Post-treatment Panel 1 cell data with UMAP coordinates |
| `panel_2_cell_data.h5ad` | Post-treatment Panel 2 cell data with UMAP coordinates |
| `pre_treatment_panel_1_cell_data.h5ad` | Pre-treatment Panel 1 cell data |
| `pre_treatment_panel_2_cell_data.h5ad` | Pre-treatment Panel 2 cell data |
| `registered_post_treatment_cell_types.csv` | Registered post-treatment cell types across both panels |
| `registered_pre_treatment_cell_types.csv` | Registered pre-treatment cell types across both panels |
| `kmeans_radius_10_80_15_neighborhoods.h5ad` | Neighborhood assignments for post-treatment cells |

## Figures Output

All notebooks save figures to a local `figures/` directory in SVG or PNG format.