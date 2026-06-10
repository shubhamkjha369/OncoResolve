import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Ensure the package src directory is in the path if not installed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import OncoResolve as orr
    print("[SUCCESS] Successfully imported OncoResolve package!")
except ImportError as e:
    print(f"[ERROR] Failed to import OncoResolve package: {e}")
    sys.exit(1)

def run_tests():
    print("Starting verification tests...")
    
    # 1. Load the features list from the pre-trained survival model
    package_dir = Path(orr.__file__).resolve().parent
    survival_model_path = package_dir / "models" / "survival_crs_ridge_model.pkl"
    
    if not survival_model_path.exists():
        print(f"[ERROR] Survival model not found at {survival_model_path}")
        sys.exit(1)
        
    survival_model = joblib.load(survival_model_path)
    required_features = sorted(survival_model.params_.index.tolist())
    print(f"Loaded {len(required_features)} required features from survival model.")
    
    # 2. Create mock transcriptomic data
    np.random.seed(42)
    n_samples = 10
    mock_data = np.random.randn(n_samples, len(required_features))
    barcodes = [f"TCGA-BRCA-PATIENT-{i:03d}" for i in range(n_samples)]
    
    df_raw = pd.DataFrame(mock_data, index=barcodes, columns=required_features)
    print(f"Created mock dataframe with shape: {df_raw.shape}")
    
    # 3. Test Utils: scale_cohort and align_features
    print("Testing utils.scale_cohort and utils.align_features...")
    df_scaled = orr.scale_cohort(df_raw)
    df_aligned = orr.align_features(df_scaled, required_features)
    assert df_aligned.shape == (n_samples, len(required_features))
    assert list(df_aligned.columns) == required_features
    print("  Utils tests passed successfully!")
    
    # 4. Test Classifier - SVM
    print("Testing OncoClassifier with SVM model...")
    svm_clf = orr.OncoClassifier(model_type="svm")
    svm_preds = svm_clf.predict(df_aligned)
    svm_probs = svm_clf.predict_proba(df_aligned)
    
    assert len(svm_preds) == n_samples
    assert svm_probs.shape == (n_samples, 5) # LumA, LumB, Her2, Basal, Normal
    print(f"  SVM Predictions: {svm_preds}")
    print("  SVM Classifier tests passed successfully!")
    
    # 5. Test Classifier - Logistic Regression
    print("Testing OncoClassifier with Logistic Regression model...")
    lr_clf = orr.OncoClassifier(model_type="lr")
    lr_preds = lr_clf.predict(df_aligned)
    lr_probs = lr_clf.predict_proba(df_aligned)
    
    assert len(lr_preds) == n_samples
    assert lr_probs.shape == (n_samples, 5)
    print(f"  LR Predictions: {lr_preds}")
    print("  Logistic Regression Classifier tests passed successfully!")
    
    # 6. Test Uniqueness (CUS)
    print("Testing compute_cus...")
    df_cus = orr.compute_cus(df_aligned, barcodes=df_aligned.index, alpha=0.001)
    assert df_cus.shape[0] == n_samples
    assert "CUS" in df_cus.columns
    print(f"  Top CUS scores:\n{df_cus.head(3)}")
    print("  CUS tests passed successfully!")
    
    # 7. Test Prognosis (Cox CRS)
    print("Testing OncoPrognosis...")
    prog = orr.OncoPrognosis()
    risk_scores = prog.predict_risk(df_aligned)
    assert len(risk_scores) == n_samples
    print(f"  Risk scores (CRS): {risk_scores.values}")
    print("  Prognosis tests passed successfully!")
    
    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")

if __name__ == "__main__":
    run_tests()
