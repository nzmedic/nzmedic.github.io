# V1: global explainability with feature importance
# TODO: test for pesence of SHAP then use for local explainability and reason codes 

# projects/lendy-graduation-retention/explain.py
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .config import BASE_FEATURE_COLS
from .feature_utils import one_hot_base_features

def explain_risk_model_global_local(
    hazard_model_bundle: Dict,
    df_for_explain: pd.DataFrame,
    model_kind: str = "graduation_risk",
    top_k: int = 8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate global and local explanations for the risk model.

    Args:
        hazard_model_bundle: Fitted hazard model bundle.
        df_for_explain: Data to explain at a snapshot.
        model_kind: Label for the output rows.
        top_k: Number of top features per record.

    Returns:
        Tuple of (global_df, local_df) explanation DataFrames.
    """
    X = one_hot_base_features(df_for_explain).reindex(columns=hazard_model_bundle["columns"], fill_value=0)

    sample = df_for_explain.sample(n=min(200, len(df_for_explain)), random_state=11).copy()
    Xs = one_hot_base_features(sample).reindex(columns=hazard_model_bundle["columns"], fill_value=0)

    # Try SHAP for tree
    use_shap = False
    shap_values = None
    try:
        shap = importlib.import_module("shap")  # type: ignore
        if hazard_model_bundle["type"] == "gbm":
            explainer = shap.TreeExplainer(hazard_model_bundle["model"])
            shap_values = explainer.shap_values(Xs)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            use_shap = True
    except Exception:
        use_shap = False

    if use_shap:
        g = np.mean(np.abs(shap_values), axis=0)
        global_df = pd.DataFrame({
            "model_kind": model_kind,
            "feature": Xs.columns,
            "importance": g
        }).sort_values("importance", ascending=False).head(30)

        local_rows = []
        for i in range(len(sample)):
            vals = shap_values[i]
            top_idx = np.argsort(np.abs(vals))[::-1][:top_k]
            for rank, j in enumerate(top_idx, start=1):
                local_rows.append({
                    "model_kind": model_kind,
                    "loan_id": int(sample.iloc[i]["loan_id"]),
                    "month_asof": int(sample.iloc[i]["month_asof"]),
                    "rank": rank,
                    "feature": Xs.columns[j],
                    "contribution": float(vals[j])
                })
        local_df = pd.DataFrame(local_rows)
        return global_df, local_df

    # Fallback: permutation importance on a proxy regression of model probs
    if hazard_model_bundle["type"] == "logit":
        X_scaled = hazard_model_bundle["scaler"].transform(X)
        y_proxy = hazard_model_bundle["model"].predict_proba(X_scaled)[:, 1]
    else:
        y_proxy = hazard_model_bundle["model"].predict_proba(X)[:, 1]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y_proxy)
    r = permutation_importance(ridge, X, y_proxy, n_repeats=3, random_state=10)
    global_df = pd.DataFrame({
        "model_kind": model_kind,
        "feature": X.columns,
        "importance": r.importances_mean
    }).sort_values("importance", ascending=False).head(30)

    # Local attribution approximation
    Xs_mat = Xs.values.astype(float)
    Xs_std = (Xs_mat - Xs_mat.mean(axis=0)) / (Xs_mat.std(axis=0) + 1e-6)

    if hazard_model_bundle["type"] == "logit":
        coefs = hazard_model_bundle["model"].coef_.reshape(-1)
    else:
        w = global_df.set_index("feature")["importance"]
        coefs = np.array([float(w.get(c, 0.0)) for c in Xs.columns])

    contribs = Xs_std * coefs.reshape(1, -1)
    local_rows = []
    for i in range(len(sample)):
        row = contribs[i]
        top_idx = np.argsort(np.abs(row))[::-1][:top_k]
        for rank, j in enumerate(top_idx, start=1):
            local_rows.append({
                "model_kind": model_kind,
                "loan_id": int(sample.iloc[i]["loan_id"]),
                "month_asof": int(sample.iloc[i]["month_asof"]),
                "rank": rank,
                "feature": Xs.columns[j],
                "contribution": float(row[j])
            })
    local_df = pd.DataFrame(local_rows)
    return global_df, local_df

def explain_uplift_via_surrogate(uplift_df: pd.DataFrame, top_k: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Explain uplift using a surrogate model for stable reason codes.

    Args:
        uplift_df: Uplift dataset with ITE estimates.
        top_k: Number of top features per record.

    Returns:
        Tuple of (global_df, local_df) explanation DataFrames.
    """
    d = uplift_df.copy()
    X = one_hot_base_features(d)
    y = d["ite_retention"].values

    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)

    surrogate = Ridge(alpha=2.0)
    surrogate.fit(Xs, y)

    global_df = pd.DataFrame({
        "model_kind": "uplift_surrogate",
        "feature": X.columns,
        "importance": np.abs(surrogate.coef_)
    }).sort_values("importance", ascending=False).head(30)

    Xs_dense = Xs.toarray() if hasattr(Xs, "toarray") else Xs
    contrib = Xs_dense * surrogate.coef_.reshape(1, -1)

    rng = np.random.default_rng(11)
    sample_idx = rng.choice(np.arange(len(d)), size=min(200, len(d)), replace=False)

    local_rows = []
    for i in sample_idx:
        row = contrib[i]
        top_idx = np.argsort(np.abs(row))[::-1][:top_k]
        for rank, j in enumerate(top_idx, start=1):
            local_rows.append({
                "model_kind": "uplift_surrogate",
                "loan_id": int(d.iloc[i]["loan_id"]),
                "month_asof": int(d.iloc[i]["month_asof"]),
                "rank": rank,
                "feature": X.columns[j],
                "contribution": float(row[j])
            })
    local_df = pd.DataFrame(local_rows)
    return global_df, local_df
