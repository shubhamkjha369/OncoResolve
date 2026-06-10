import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class OncoPrognosis:
    """
    OncoResolve prognostic risk model.
    Wraps an L2-regularized (Ridge) Cox Proportional Hazards model
    to predict the Consensus Risk Score (CRS) for survival stratification.
    """
    def __init__(self, model_path=None, penalizer=0.5):
        self.penalizer = penalizer
        self.model_ = None
        
        # Determine default model path relative to package directory if not specified
        package_dir = Path(__file__).resolve().parent
        if model_path is None:
            model_path = package_dir / "models" / "survival_crs_ridge_model.pkl"
            
        if os.path.exists(model_path):
            self.load_pretrained(model_path)

    def load_pretrained(self, model_path):
        """
        Loads a pre-trained CoxPHFitter model.
        """
        self.model_ = joblib.load(model_path)
        return self

    def fit(self, X, survival_df, event_col="OS_STATUS_BIN", time_col="OS_MONTHS"):
        """
        Trains a new regularized Ridge Cox model.
        """
        from lifelines import CoxPHFitter
        
        df_combined = X.join(survival_df[[time_col, event_col]], how="inner")
        
        self.model_ = CoxPHFitter(penalizer=self.penalizer, l1_ratio=0.0)
        self.model_.fit(df_combined, duration_col=time_col, event_col=event_col)
        return self

    def predict_risk(self, X):
        """
        Predicts the Consensus Risk Score (CRS) for the input samples.
        """
        if self.model_ is None:
            raise RuntimeError("Prognostic model is not loaded or trained. Call fit() or load_pretrained() first.")
            
        if not isinstance(X, pd.DataFrame):
            feature_names = self.model_.summary.index.tolist()
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X
            
        crs_scores = self.model_.predict_partial_hazard(X_df)
        return crs_scores
