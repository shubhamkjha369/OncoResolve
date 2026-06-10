import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class OncoClassifier:
    """
    OncoResolve diagnostic classifier.
    Exposes APIs to load the pre-trained models (RBF-SVM or Logistic Regression)
    or train a new model for PAM50 breast cancer subtyping.
    """
    def __init__(self, model_type="svm", model_path=None, label_encoder_path=None):
        self.model_type = model_type.lower()
        self.model_ = None
        self.label_encoder_ = None
        
        # Determine default paths relative to package directory if not specified
        package_dir = Path(__file__).resolve().parent
        
        if model_path is None:
            if self.model_type == "svm":
                model_path = package_dir / "models" / "final_probability_svm.pkl"
            else:
                model_path = package_dir / "models" / "final_logistic_regression_model.pkl"
                
        if label_encoder_path is None:
            label_encoder_path = package_dir / "models" / "label_encoder_cohort.pkl"
            if not os.path.exists(label_encoder_path):
                label_encoder_path = package_dir / "models" / "label_encoder.pkl"
        
        # Attempt to load pre-trained if the files exist
        if os.path.exists(model_path) and os.path.exists(label_encoder_path):
            self.load_pretrained(model_path, label_encoder_path)

    def load_pretrained(self, model_path, label_encoder_path):
        """
        Loads the pre-trained classifier and label encoder.
        """
        self.model_ = joblib.load(model_path)
        self.label_encoder_ = joblib.load(label_encoder_path)
        return self

    def fit(self, X, y, model_params=None):
        """
        Trains a new classification model from scratch.
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression

        if isinstance(y[0], str) or isinstance(y.iloc[0] if hasattr(y, 'iloc') else y[0], str):
            self.label_encoder_ = LabelEncoder()
            y_encoded = self.label_encoder_.fit_transform(y)
        else:
            y_encoded = y

        params = model_params or {}
        if self.model_type == "svm":
            self.model_ = SVC(kernel="rbf", probability=True, random_state=42, **params)
        elif self.model_type == "logistic_regression" or self.model_type == "lr":
            self.model_ = LogisticRegression(max_iter=3000, random_state=42, **params)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Select 'svm' or 'logistic_regression'.")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self.model_.fit(X_arr, y_encoded)
        return self

    def predict(self, X):
        """
        Predicts PAM50 intrinsic subtypes for the input samples.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not loaded or trained. Call fit() or load_pretrained() first.")
            
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        preds = self.model_.predict(X_arr)
        
        if self.label_encoder_ is not None:
            return self.label_encoder_.inverse_transform(preds)
        return preds

    def predict_proba(self, X):
        """
        Predicts classification probabilities for each PAM50 subtype.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not loaded or trained. Call fit() or load_pretrained() first.")
            
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        probs = self.model_.predict_proba(X_arr)
        
        if self.label_encoder_ is not None:
            classes = self.label_encoder_.classes_
            return pd.DataFrame(probs, columns=classes)
        return probs
