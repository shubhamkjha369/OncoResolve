# OncoResolve: API Documentation & Usage Guide

`OncoResolve` is a high-hygiene Python bioinformatics library designed for breast cancer molecular subtyping, consensus prognosis risk assessment, and patient-centric uniqueness scoring. It integrates pre-trained machine learning classifiers and survival regression models based on the TCGA-BRCA cohort.

This documentation describes the module architecture, APIs, data prep guidelines, and end-to-end analytical workflows.

---

## Table of Contents
1. [Installation & Setup](#1-installation--setup)
2. [Data Preparation & Harmonization (`utils`)](#2-data-preparation--harmonization-utils)
3. [Molecular Subtyping (`OncoClassifier`)](#3-molecular-subtyping-oncoclassifier)
4. [Prognostic Survival Stratification (`OncoPrognosis`)](#4-prognostic-survival-stratification-oncoprognosis)
5. [Patient Uniqueness Scoring (`compute_cus`)](#5-patient-uniqueness-scoring-compute_cus)
6. [Consensus Feature Selection (`ConsensusSelector`)](#6-consensus-feature-selection-consensusselector)
7. [Complete End-to-End Workflow](#7-complete-end-to-end-workflow)

---

## 1. Installation & Setup

Ensure you have your environment set up and the package installed:
```bash
pip install oncoresolve
```

Dependencies:
* `numpy >= 1.24.0`
* `pandas >= 2.0.0`
* `scikit-learn >= 1.3.0`
* `joblib >= 1.3.0`
* `lifelines >= 0.27.0`

---

## 2. Data Preparation & Harmonization (`utils`)

The data preparation utilities assist in converting Entrez IDs to HUGO symbols, scaling expression profiles, and aligning them with model features.

### `harmonize_namespaces(df, mapping_path)`
Maps Entrez Gene IDs in column names to official HUGO gene symbols.
* **Arguments:**
  * `df` (*pd.DataFrame*): Genomic expression dataframe with genes as columns and patient barcodes as the index.
  * `mapping_path` (*str* or *Path*): Path to the `tcga_entrez_to_hugo.pkl` dictionary mapping file.
* **Returns:**
  * `pd.DataFrame`: Cleaned dataframe with HUGO gene symbol columns. Duplicate columns (mapped from multiple Entrez IDs) are averaged.

### `scale_cohort(df)`
Performs Z-score normalization across all samples per feature (gene).
* **Arguments:**
  * `df` (*pd.DataFrame*): Harmonized expression dataframe.
* **Returns:**
  * `pd.DataFrame`: Normalized dataframe where each column has $\mu = 0$ and $\sigma = 1$.

### `align_features(df, required_features, fill_value=0.0)`
Ensures the columns in the dataframe match the exact sequence of genes expected by the models (sorted alphabetically). Missing genes are filled with a default value.
* **Arguments:**
  * `df` (*pd.DataFrame*): scaled/raw cohort dataframe.
  * `required_features` (*list*): List of required gene symbols.
  * `fill_value` (*float*): Value used to populate columns of missing genes. Defaults to `0.0`.
* **Returns:**
  * `pd.DataFrame`: Aligned dataframe sorted alphabetically by column names.

---

## 3. Molecular Subtyping (`OncoClassifier`)

`OncoClassifier` provides PAM50 intrinsic subtyping ('basal', 'her2', 'luminal_A', 'luminal_B', 'normal') using either a Support Vector Machine (RBF-SVM) or a Logistic Regression (LR) model.

> [!IMPORTANT]
> The pre-trained classifiers bundle an internal `StandardScaler` step. You **must** pass raw (unscaled) log2-transformed expression values (e.g. $\log_2(\text{TPM}+1)$) to these classifiers. Passing pre-scaled data will distort predictions.

### Initialization
```python
from OncoResolve import OncoClassifier

# Load the pre-trained SVM model (Default)
clf_svm = OncoClassifier(model_type="svm")

# Load the pre-trained Logistic Regression model
clf_lr = OncoClassifier(model_type="lr")
```

### Predicting Subtypes
```python
# 1. Predict raw expression (Default)
subtypes = clf_svm.predict(df_aligned_raw)
probs = clf_svm.predict_proba(df_aligned_raw)

# 2. Predict using pre-scaled (Z-score normalized) expression data
# Passing scaled_input=True bypasses the pipeline's StandardScaler
subtypes = clf_svm.predict(df_aligned_scaled, scaled_input=True)
probs = clf_svm.predict_proba(df_aligned_scaled, scaled_input=True)
```

* **Arguments for `predict(X, scaled_input=False)` & `predict_proba(X, scaled_input=False)`:**
  * `X` (*pd.DataFrame* or *np.ndarray*): Input genomic expression matrix.
  * `scaled_input` (*bool*): If set to `True`, the classifier will bypass the internal `StandardScaler` step in the pre-trained pipeline and perform prediction directly. Use this if your dataset is already cohort Z-score scaled. Defaults to `False`.

### Training a Custom Subtyping Model
If you have your own cohort with labeled subtypes, you can train a new classifier from scratch:
```python
# Create an untrained classifier instance
custom_clf = OncoClassifier(model_type="svm", model_path="none", label_encoder_path="none")

# Train the model (X can be scaled/unscaled)
custom_clf.fit(X_train, y_train, model_params={"C": 1.0})
```

---

## 4. Prognostic Survival Stratification (`OncoPrognosis`)

`OncoPrognosis` uses an L2-regularized (Ridge) Cox Proportional Hazards model to predict a Consensus Risk Score (CRS) representing survival risk.

> [!IMPORTANT]
> Unlike the classifier, the prognosis model **does not** contain an internal scaler. You **must** scale your cohort using `scale_cohort` before passing it to `predict_risk()`.

### Initialization
```python
from OncoResolve import OncoPrognosis

prog = OncoPrognosis()
```

### Predicting CRS (Consensus Risk Score)
The risk score corresponds to the hazard ratio factor $\exp(\vec{\beta}^T \vec{x})$ relative to the baseline.
```python
# Predict risk scores
risk_scores = prog.predict_risk(df_aligned_scaled)
```

### Training a Custom Survival Model
To train a regularized Ridge Cox model on custom survival data:
```python
# Prepare survival outcome dataframe containing time and event columns
survival_outcomes = pd.DataFrame({
    "OS_MONTHS": [12.4, 45.1, 8.2],
    "OS_STATUS_BIN": [1, 0, 1]  # 1 = Deceased, 0 = Censored
}, index=["Patient_1", "Patient_2", "Patient_3"])

custom_prog = OncoPrognosis(model_path="none")
custom_prog.fit(df_aligned_scaled, survival_outcomes, event_col="OS_STATUS_BIN", time_col="OS_MONTHS")
```

---

## 5. Patient Uniqueness Scoring (`compute_cus`)

`compute_cus` implements an N-of-1 patient outlier detection algorithm. It calculates a **Composite Uniqueness Score (CUS)** for each patient by combining:
1. **Topological Distance:** Euclidean distance of the patient's profile from the cohort average.
2. **Reconstruction Error:** A Ridge regression model tries to reconstruct the patient's expression profile as a linear combination of all other patients in the cohort. If the reconstruction error ($1 - R^2$) is high, the patient's profile is highly atypical.

### Usage
```python
from OncoResolve import compute_cus

# Expects scaled data
df_cus = compute_cus(df_aligned_scaled, barcodes=df_aligned_scaled.index, alpha=0.001)
# Returns pd.DataFrame:
# | Patient_ID | Topo_Distance | Recon_Error | CUS |
```

A CUS close to $1.0$ flags highly unique or atypical patients that may warrant targeted therapies, whereas a score close to $0.0$ represents typical patient profiles.

---

## 6. Consensus Feature Selection (`ConsensusSelector`)

`ConsensusSelector` is an ensemble method to identify robust biomarkers. It runs three feature selection models and retains features selected by **at least two** of them:
1. **ANOVA F-Test:** Identifies features with strong linear differences between classes.
2. **LASSO (L1 Regularization):** Selects sparse, non-redundant features.
3. **Random Forest (Gini Importance):** Captures non-linear interactions.

### Usage
```python
from OncoResolve import ConsensusSelector

# Select the top 50 consensus biomarkers
selector = ConsensusSelector(top_k=50)
selector.fit(X_train, y_train)

# Get the list of selected consensus genes
selected_genes = selector.consensus_features_

# Transform the expression matrix to keep only selected features
X_selected = selector.transform(X_train)
```

---

## 7. Complete End-to-End Workflow

Below is a complete script demonstrating how to load raw RNA-seq data, clean/harmonize it, select features, subtype patients, run prognosis, and compute uniqueness.

```python
import pandas as pd
import OncoResolve as orr

# 1. Load your raw expression matrix (genes as columns, patients as index)
df_raw = pd.read_csv("raw_expression_data.csv", index_col=0)

# 2. Harmonize Entrez IDs to HUGO symbols
df_clean = orr.harmonize_namespaces(df_raw, "tcga_entrez_to_hugo.pkl")

# 3. Retrieve model-required feature names (alphabetically sorted)
prog = orr.OncoPrognosis()
model_features = sorted(prog.model_.params_.index.tolist())

# 4. Prepare data splits:
# - Raw aligned for the Classifier
df_aligned_raw = orr.align_features(df_clean, model_features)

# - Scaled aligned for Uniqueness and Prognosis
df_scaled = orr.scale_cohort(df_clean)
df_aligned_scaled = orr.align_features(df_scaled, model_features)

# 5. Diagnostic PAM50 Subtyping
clf = orr.OncoClassifier(model_type="svm")
subtypes = clf.predict(df_aligned_raw)
probabilities = clf.predict_proba(df_aligned_raw)

# 6. Prognostic Risk CRS Score
risk_scores = prog.predict_risk(df_aligned_scaled)

# 7. Patient Uniqueness CUS Score (injecting subtypes for visualization)
df_cus = orr.compute_cus(df_aligned_scaled, barcodes=df_aligned_scaled.index, y_subtype=subtypes)

# 8. Consolidate results
results = pd.DataFrame({
    "PAM50_Subtype": subtypes,
    "Consensus_Risk_Score": risk_scores
}, index=df_raw.index)

# Join CUS details
results = results.join(df_cus.set_index("Patient_ID"), how="inner")

print("Analysis Complete! Top 5 atypical patients:")
print(results.sort_values(by="CUS", ascending=False).head())
```

---

## 8. Multi-Cancer & General Disease Applicability

While the default models packaged with `OncoResolve` are pre-trained on Breast Cancer data (TCGA-BRCA), the underlying core classes and mathematical modules are fully generalizable to any cohort (e.g., colorectal, ovarian, lung cancer).

Here are coding examples showing how to adapt the library for custom cancer/disease datasets.

### A. Custom Subtyping on Colorectal Cancer (CMS1-4)
```python
from OncoResolve import OncoClassifier

# 1. Initialize an untrained classifier (passing 'none' to paths bypasses loading breast cancer weights)
clf_crc = OncoClassifier(model_type="svm", model_path="none", label_encoder_path="none")

# 2. Train on colorectal expression data and CMS class labels
clf_crc.fit(X_crc_train, y_crc_subtypes)

# 3. Predict subtypes
predictions = clf_crc.predict(X_crc_test)
```

### B. Custom Prognosis on Ovarian Cancer Survival Outcomes
```python
from OncoResolve import OncoPrognosis

# 1. Initialize an untrained prognostic model
prog_ovarian = OncoPrognosis(model_path="none")

# 2. Train a regularized Cox proportional hazards model
prog_ovarian.fit(
    X_ovarian_scaled, 
    survival_df, 
    event_col="OS_STATUS", 
    time_col="OS_MONTHS"
)

# 3. Predict risk scores
risk_scores = prog_ovarian.predict_risk(X_ovarian_scaled)
```

### C. Cohort Uniqueness and Outlier Detection on Lung Cancer
```python
import OncoResolve as orr

# The uniqueness score is completely mathematical and cohort-independent.
# Pass scaled expression data from any cancer cohort to calculate CUS scores:
df_cus = orr.compute_cus(X_lung_scaled)
```
