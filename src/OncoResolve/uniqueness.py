import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed

def _process_sample(i, X_full, alpha):
    y_i = X_full[i, :]
    X_minus_i = np.delete(X_full, i, axis=0).T

    scaler = StandardScaler()
    X_minus_i_sc = scaler.fit_transform(X_minus_i)

    y_i_mean = y_i.mean()
    y_i_centered = y_i - y_i_mean

    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X_minus_i_sc, y_i_centered)
    
    y_pred_centered = ridge.predict(X_minus_i_sc)
    y_pred_vector = y_pred_centered + y_i_mean
    
    r2 = r2_score(y_i, y_pred_vector)
    return r2

def compute_cus(X, barcodes=None, y_subtype=None, alpha=0.001, n_jobs=-1):
    """
    Computes the Composite Uniqueness Score (CUS) for each patient.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
        if barcodes is None:
            barcodes = X.index.tolist()
    else:
        X_arr = X
        if barcodes is None:
            barcodes = [f"Patient_{i}" for i in range(X_arr.shape[0])]
            
    n_samples = X_arr.shape[0]
    
    dist_matrix = euclidean_distances(X_arr, X_arr)
    mean_distances = dist_matrix.mean(axis=1)
    
    scaler = MinMaxScaler()
    norm_distances = scaler.fit_transform(mean_distances.reshape(-1, 1)).flatten()
    
    r2_scores = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_sample)(i, X_arr, alpha) for i in range(n_samples)
    )
    r2_scores = np.array(r2_scores)
    
    reconstruction_errors = 1.0 - r2_scores
    norm_recon_errors = scaler.fit_transform(reconstruction_errors.reshape(-1, 1)).flatten()
    
    cus_scores = 0.5 * norm_distances + 0.5 * norm_recon_errors
    
    df_cus = pd.DataFrame({
        "Patient_ID": barcodes,
        "Topo_Distance": norm_distances,
        "Recon_Error": norm_recon_errors,
        "CUS": cus_scores
    })
    
    if y_subtype is not None:
        df_cus.insert(1, "Subtype", y_subtype)
        
    df_cus = df_cus.sort_values(by="CUS", ascending=False).reset_index(drop=True)
    return df_cus
