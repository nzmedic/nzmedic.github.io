#train/predict hazard, propensity, outcome models; ITE + segmentation; metrics/frontier

# projects/lendy-graduation-retention/models.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .config import BASE_FEATURE_COLS

def _one_hot(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    X = df[columns].copy()
    X = pd.get_dummies(X, columns=["introducer"], drop_first=True)
    return X

def fit_hazard_models(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> Dict[str, Dict]:
    X_train = _one_hot(train_df, BASE_FEATURE_COLS)
    y_train = train_df["event"].values.astype(int)
    X_valid = _one_hot(valid_df, BASE_FEATURE_COLS).reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler(with_mean=False)
    Xtr_s = scaler.fit_transform(X_train)
    Xva_s = scaler.transform(X_valid)

    baseline = LogisticRegression(penalty="l2", C=0.8, solver="liblinear", max_iter=500)
    baseline.fit(Xtr_s, y_train)

    gbm = GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8)
    gbm.fit(X_train, y_train)

    return {
        "hazard_logit": {"model": baseline, "scaler": scaler, "columns": X_train.columns.tolist(), "type": "logit"},
        "hazard_gbm": {"model": gbm, "scaler": None, "columns": X_train.columns.tolist(), "type": "gbm"},
    }

def predict_hazard(model_bundle: Dict, df: pd.DataFrame) -> np.ndarray:
    X = _one_hot(df, BASE_FEATURE_COLS).reindex(columns=model_bundle["columns"], fill_value=0)
    if model_bundle["type"] == "logit":
        Xs = model_bundle["scaler"].transform(X)
        p = model_bundle["model"].predict_proba(Xs)[:, 1]
    else:
        p = model_bundle["model"].predict_proba(X)[:, 1]
    return np.clip(p, 1e-6, 1-1e-6)

# get the loan level summaries including balances and probability the customer graduates to a prime loan (churns)
def loan_level_survival_summary(
    perf_df: pd.DataFrame,
    hazard_model_bundle: Dict,
    asof_month: int,
    horizons=(3, 6, 12)
) -> pd.DataFrame:
    snap = perf_df[(perf_df["month_asof"] == asof_month) & (perf_df["balance"] > 0)].copy()
    max_forward = max(24, max(horizons))

    surv = np.ones(len(snap))
    cum_event_prob = np.zeros(len(snap))
    expected_t = np.zeros(len(snap))
    probs = {}

    current = snap.copy()
    for t in range(1, max_forward + 1):
        current = current.copy()
        current["loan_age_month"] = current["loan_age_month"] + 1
        current["month_asof"] = current["month_asof"] + 1

        ht = predict_hazard(hazard_model_bundle, current)
        event_t = surv * ht
        cum_event_prob += event_t
        expected_t += t * event_t
        surv *= (1 - ht)

        if t in horizons:
            probs[t] = cum_event_prob.copy()

    cum24 = cum_event_prob.copy()
    expected_time = np.where(cum24 > 1e-6, expected_t / cum24, np.nan)

    out = pd.DataFrame({
        "scenario_name": snap["scenario_name"].values,
        "loan_id": snap["loan_id"].values,
        "product": snap["product"].values,
        "month_asof": snap["month_asof"].values,
        "balance": snap["balance"].values,
        "prob_graduate_3m": probs.get(3, np.nan),
        "prob_graduate_6m": probs.get(6, np.nan),
        "prob_graduate_12m": probs.get(12, np.nan),
        "expected_time_to_graduate_months": expected_time
    })

    # For horizons less than 12 months prob_graduate_12m is NaN which breaks qcut. In such cases, use longest horizon available
    decile_col = None
    for cand in ["prob_graduate_12m", "prob_graduate_6m", "prob_graduate_3m"]:
        if cand in out.columns and out[cand].notna().any():
            decile_col = cand
            break

    if decile_col is None:
        out["risk_decile"] = 1
    else:
        s = out[decile_col]
        # Null horizon issue resolved in calling functions but with flat distributions (identical, or nearly) qcut can still fail so manage here
        if s.nunique(dropna=True) < 10:
            out["risk_decile"] = pd.cut(
                s.rank(method="first"),
                bins=min(10, max(1, s.notna().sum())),
                labels=False,
                include_lowest=True
            )
            out["risk_decile"] = out["risk_decile"].fillna(0) + 1
        else:
            out["risk_decile"] = pd.qcut(
                s.rank(method="first"),
                10,
                labels=False,
                duplicates="drop"
            ) + 1

    out["risk_decile"] = out["risk_decile"].astype(int)

    return out

# --- uplift / causal ---

def fit_propensity_model(train_df: pd.DataFrame) -> Dict:
    X = _one_hot(train_df, BASE_FEATURE_COLS)
    y = train_df["treated"].values.astype(int)
    m = GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8)
    m.fit(X, y)
    return {"model": m, "columns": X.columns.tolist()}

def predict_propensity(bundle: Dict, df: pd.DataFrame) -> np.ndarray:
    X = _one_hot(df, BASE_FEATURE_COLS).reindex(columns=bundle["columns"], fill_value=0)
    p = bundle["model"].predict_proba(X)[:, 1]
    return np.clip(p, 0.02, 0.98)

def fit_t_learner_outcome_models(train_df: pd.DataFrame, outcome_col: str = "retained_within_h") -> Dict:
    X = _one_hot(train_df, BASE_FEATURE_COLS)
    y = train_df[outcome_col].values.astype(int)
    t = train_df["treated"].values.astype(int)

    X_t, y_t = X[t == 1], y[t == 1]
    X_c, y_c = X[t == 0], y[t == 0]

    mt = GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8)
    mc = GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8)
    mt.fit(X_t, y_t)
    mc.fit(X_c, y_c)

    return {"mt": mt, "mc": mc, "columns": X.columns.tolist()}

def predict_outcome_models(bundle: Dict, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = _one_hot(df, BASE_FEATURE_COLS).reindex(columns=bundle["columns"], fill_value=0)
    mu1 = bundle["mt"].predict_proba(X)[:, 1]
    mu0 = bundle["mc"].predict_proba(X)[:, 1]
    return np.clip(mu1, 1e-3, 1-1e-3), np.clip(mu0, 1e-3, 1-1e-3)

def doubly_robust_ite_retention(df: pd.DataFrame, prop: Dict, outm: Dict, outcome_col: str = "retained_within_h") -> pd.DataFrame:
    d = df.copy()
    e = predict_propensity(prop, d)
    mu1, mu0 = predict_outcome_models(outm, d)

    T = d["treated"].values.astype(int)
    Y = d[outcome_col].values.astype(float)
    muT = np.where(T == 1, mu1, mu0)

    dr = (mu1 - mu0) + (T - e) * (Y - muT) / (e * (1 - e))
    dr = np.clip(dr, -0.25, 0.40)

    d["propensity"] = e
    d["mu1_retention"] = mu1
    d["mu0_retention"] = mu0
    d["ite_retention"] = dr
    return d

def segment_customers(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    base_ret = d["mu0_retention"].values
    ite = d["ite_retention"].values

    seg = []
    for tau, br in zip(ite, base_ret):
        if tau < -0.01:
            seg.append("do_not_disturb")
        elif tau >= 0.05:
            seg.append("persuadable")
        elif br >= 0.90 and tau < 0.03:
            seg.append("sure_thing")
        elif br < 0.75 and tau < 0.03:
            seg.append("lost_cause")
        else:
            seg.append("mixed")
    d["segment"] = seg
    return d

# --- evaluation ---

def survival_eval_metrics(perf_df: pd.DataFrame, hazard_bundle: Dict, asof_month: int) -> pd.DataFrame:
    horizons = [3, 6, 12]
    snap = perf_df[(perf_df["month_asof"] == asof_month) & (perf_df["balance"] > 1_000)].copy()
    df = perf_df.sort_values(["loan_id", "month_asof"]).copy()

    metrics = []
    for h in horizons:
        future = df[(df["month_asof"] > asof_month) & (df["month_asof"] <= asof_month + h)]
        y = future.groupby("loan_id")["graduated_this_month"].max().rename("y").reset_index()
        tmp = snap.merge(y, on="loan_id", how="left")
        tmp["y"] = tmp["y"].fillna(0).astype(int)

        summ = loan_level_survival_summary(perf_df, hazard_bundle, asof_month, horizons=(3, 6, 12))
        tmp = tmp.merge(summ[["loan_id", f"prob_graduate_{h}m"]], on="loan_id", how="left")

        y_true = tmp["y"].values
        y_pred = tmp[f"prob_graduate_{h}m"].values
        auc = roc_auc_score(y_true, y_pred) if (y_true.sum() > 10 and y_true.sum() < len(y_true) - 10) else np.nan
        metrics.append(("hazard_model", f"AUC_{h}m", float(auc), f"asof_month={asof_month}"))

        tmp["decile"] = pd.qcut(pd.Series(y_pred).rank(method="first"), 10, labels=False) + 1
        cal = tmp.groupby("decile").agg(pred_mean=(f"prob_graduate_{h}m", "mean"),
                                        obs_rate=("y", "mean")).reset_index()
        mce = float(np.mean(np.abs(cal["pred_mean"] - cal["obs_rate"])))
        metrics.append(("hazard_model", f"Calib_MAE_{h}m", mce, f"decile-calibration asof_month={asof_month}"))

    return pd.DataFrame(metrics, columns=["model_name", "metric_name", "metric_value", "notes"])

def naive_vs_adjusted_treatment_effect(uplift_df: pd.DataFrame) -> Tuple[float, float]:
    d = uplift_df.copy()
    naive = d.loc[d["treated"] == 1, "retained_within_h"].mean() - d.loc[d["treated"] == 0, "retained_within_h"].mean()
    adjusted = d["ite_retention"].mean()
    return float(naive), float(adjusted)

def compute_uplift_curve(uplift_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    d = uplift_df.sort_values("ite_retention", ascending=False).reset_index(drop=True).copy()
    d["inc"] = d["ite_retention"].clip(-0.25, 0.40)
    d["cum_inc"] = d["inc"].cumsum()
    d["n"] = np.arange(1, len(d) + 1)

    mean_inc = d["inc"].mean()
    d["random_cum"] = d["n"] * mean_inc
    d["uplift_over_random"] = d["cum_inc"] - d["random_cum"]
    auuc = float(d["uplift_over_random"].mean())
    return d[["n", "cum_inc", "random_cum", "uplift_over_random"]], auuc

def uplift_frontier(uplift_df: pd.DataFrame, budget_type: str, budget_values: List[float], horizon_months: int = 12):
    d = uplift_df.copy()
    d["incremental_retained_balance"] = d["balance"] * d["ite_retention"].clip(0, 1)

    funding_proxy = (d["market_rate"] - 0.02).clip(0.01, 0.12)
    net_margin = (d["apr"] - funding_proxy).clip(0.01, 0.25)

    discount = (d["treatment_bps"].fillna(0) / 10_000.0)
    horizon_years = horizon_months / 12.0
    d["incremental_nii"] = d["incremental_retained_balance"] * (net_margin - discount).clip(-0.10, 0.30) * horizon_years

    eligible = d[(d["segment"] != "do_not_disturb") & (d["ite_retention"] > 0.0)].copy()
    eligible["offer_cost"] = eligible["balance"] * (eligible["treatment_bps"]/10_000.0) * horizon_years
    eligible = eligible.sort_values("incremental_nii", ascending=False)

    rows = []
    for b in budget_values:
        if budget_type == "count":
            chosen = eligible.head(int(b))
        else:
            chosen = eligible[eligible["offer_cost"].cumsum() <= b]

        rows.append({
            "budget_type": budget_type,
            "budget_value": float(b),
            "retained_aum": float(chosen["incremental_retained_balance"].sum()),
            "incremental_nii": float(chosen["incremental_nii"].sum()),
            "offers_made": int(len(chosen))
        })
    return pd.DataFrame(rows), d
