# Visualize Platform Data

An interactive Streamlit web application for exploring, modeling, and visualizing longitudinal cognitive and motor assessment data. The application integrates platform behavioral scores with questionnaire and registration records to support cross-sectional, longitudinal, psychometric reliability, correlation, and individual-level clustering analyses.

**Live Demo:** [Visualize Platform Data](https://visualize-platform-data-uaoq38joscqjyhgaxkgh7r.streamlit.app/)

## Features

### Data Management and Filtering
- **Flexible Data Ingestion:** Automatically loads the latest files from the local `data/` directory or accepts manual uploads via the sidebar for either a pre-merged CSV (`merged_data*.csv`) or three separate source files:
  - Platform data (`tidy_predicted_results*.csv`)
  - Questionnaire data (`Qdata*.xlsx`)
  - Registration data (`NHRI*.csv`)
- **Automated Preprocessing:** Merges datasets by participant ID (`SID`) and `Session`, calculates exact ages and longitudinal time intervals (`year`, `month`, `week`, or `day`), maps demographic variables (sex, education years/levels), and supports exporting the merged dataset as CSV.
- **Global Filters:** Filter analyses by session (`mean`, `1`, `2`, `3`), age range, and custom age-group cut points.

### Analytical Modules
1. **Overview**
   - Sample composition tables by session and age group, with optional missing-data highlighting.
   - Interactive score distribution histograms and descriptive statistics across the five cognitive/motor domains (`WorkMem`, `EpisMem`, `LangComp`, `LangProd`, `Motor`).
   - Filterable raw data table.
2. **Cross-sectional**
   - Age-related scatter plots for domain scores and raw behavioral metrics.
   - Configurable trend lines: Linear and Quadratic OLS fits (with 95% confidence bands) or LOWESS smoothing.
3. **Longitudinal**
   - Individual trajectory plots over age or session.
   - Linear Mixed-Effects Models (LMEM) estimating baseline age (`FirstAge`) and longitudinal change (`DateDiff`) effects with optional covariates (`IsFemale`, `EduYears`, `MoCA`), random slopes/intercepts, and automatic fallback to random-intercept models when needed.
4. **Reliability**
   - Test-retest reliability via Intraclass Correlation Coefficients (`ICC(C,)`, with optional `ICC(1,)` and `ICC(A,)`).
   - Paired t-tests (2 sessions) or repeated-measures ANOVA (3+ sessions) to evaluate systematic session differences.
   - Bland-Altman plots with Limits of Agreement (LoA) and Oldham's method regression fit.
5. **Correlation**
   - Between-person (Session 1 or session mean) and within-person (repeated-measures) correlation heatmaps for domain or raw scores, with optional MoCA inclusion.
   - Supports Pearson and Spearman methods, partial correlations controlling for `Age`, `IsFemale`, and `EduYears`, age-group stratification, and first principal component (PC1) variance-explained bar charts.
6. **Individual**
   - Unsupervised clustering (`K-Means` or `Gaussian Mixture`) on standardized domain scores projected into 2D space via `PCA` or `t-SNE`, highlighting the selected participant.
   - Z-score radar chart comparing an individual participant against their age-group mean.

## Tech Stack

- **Framework:** Streamlit
- **Data Manipulation:** Pandas, NumPy, python-dateutil
- **Statistical Modeling:** Statsmodels, Pingouin, SciPy
- **Machine Learning:** scikit-learn
- **Visualization:** Plotly
