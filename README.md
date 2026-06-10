# OncoResolve: Reusable Python Bioinformatics Package

A High-Hygiene Explainable AI and Patient-Centric Uniqueness Framework for Breast Cancer Subtyping. This package allows you to clean custom transcriptomic matrices, run PAM50 predictions using pre-trained TCGA models, compute N-of-1 patient Uniqueness scores (CUS), and predict overall survival risk scores (CRS).

## Installation

### Local development install
From the root of this project folder:
```bash
pip install -e .
```

### Direct install from GitHub
```bash
pip install git+https://github.com/shubhamkjha369/OncoResolve-Breast-Cancer-Transcriptomics.git
```

## Reusable Library Usage

Once installed, you can import the package using capital `OncoResolve`:

```python
import OncoResolve as orr
import pandas as pd

# 1. Prepare and Harmonize your custom RNA-seq expression matrix (genes as columns)
# Pass in the path to the tcga_entrez_to_hugo.pkl mapping file
df_clean = orr.harmonize_namespaces(df_raw, "path/to/tcga_entrez_to_hugo.pkl")
df_scaled = orr.scale_cohort(df_clean)

# Load your consensus gene selection list and align columns alphabetically
consensus_genes = list(pd.read_parquet("path/to/final_consensus_biomarkers.parquet")["gene"])
df_aligned = orr.align_features(df_scaled, consensus_genes)

# 2. Run classification using pre-trained SVM or Logistic Regression models
# Pass in paths to final_probability_svm.pkl and label_encoder_cohort.pkl
clf = orr.OncoClassifier(
    model_type="svm", 
    model_path="path/to/final_probability_svm.pkl", 
    label_encoder_path="path/to/label_encoder_cohort.pkl"
)
predictions = clf.predict(df_aligned)        # Returns PAM50 subtype strings
probabilities = clf.predict_proba(df_aligned)  # Returns class probabilities DataFrame

# 3. Compute Patient Uniqueness Scores (CUS)
df_cus = orr.compute_cus(df_aligned, barcodes=df_aligned.index, alpha=0.001)

# 4. Predict Overall Survival Risk Scores (Consensus Cox CRS)
# Pass in path to survival_crs_ridge_model.pkl
prog = orr.OncoPrognosis(model_path="path/to/survival_crs_ridge_model.pkl")
risk_scores = prog.predict_risk(df_aligned)
```

## Authors
* **Shubham Jha** (shubhamkjha369@gmail.com)
