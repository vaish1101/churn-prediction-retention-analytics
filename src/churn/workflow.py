"""Training-only development, frozen test evaluation, and export generation."""

import hashlib
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .analytics import (
    bootstrap_intervals, calibration_bins, capacity_table, classification_metrics,
    scored_customers, subgroup_diagnostics,
)
from .config import EXPORTS, MODEL_VERSION, RAW_DATA, REPORTS, ROOT, SEED
from .data import CATEGORICAL, NUMERIC, FEATURES, load_customer_data, model_matrix


def logistic_estimator():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("numeric", numeric, list(NUMERIC)),
                                      ("categorical", categorical, list(CATEGORICAL))])
    return Pipeline([("preprocess", preprocessor),
                     ("model", LogisticRegression(max_iter=2000, random_state=SEED))])


def catboost_estimator():
    return CatBoostClassifier(
        loss_function="Logloss", verbose=False, allow_writing_files=False,
        random_seed=SEED, thread_count=2, cat_features=list(CATEGORICAL),
        learning_rate=0.04, l2_leaf_reg=3,
    )


def _tune(x, y):
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    specs = {
        "Logistic Regression": (logistic_estimator(), {"model__C": [0.1, 1.0, 10.0]}),
        "CatBoost": (catboost_estimator(), {"depth": [4, 6], "iterations": [350]}),
    }
    fitted = {}
    cv_rows = []
    for name, (estimator, grid) in specs.items():
        search = GridSearchCV(estimator, grid, cv=cv, scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
                              refit="pr_auc", n_jobs=1, error_score="raise")
        search.fit(x, y)
        fitted[name] = search.best_estimator_
        for i, params in enumerate(search.cv_results_["params"]):
            cv_rows.append({"model": name, "params": json.dumps(params, sort_keys=True),
                            "cv_roc_auc_mean": float(search.cv_results_["mean_test_roc_auc"][i]),
                            "cv_roc_auc_sd": float(search.cv_results_["std_test_roc_auc"][i]),
                            "cv_pr_auc_mean": float(search.cv_results_["mean_test_pr_auc"][i]),
                            "cv_pr_auc_sd": float(search.cv_results_["std_test_pr_auc"][i])})
    return fitted, pd.DataFrame(cv_rows)


def _choose(fitted, x_train, y_train, x_valid, y_valid):
    """Validation-only choice. Calibrate only for a meaningful Brier improvement."""
    candidates = []
    for name, raw_model in fitted.items():
        raw_probability = raw_model.predict_proba(x_valid)[:, 1]
        raw = classification_metrics(y_valid, raw_probability)
        raw_top = capacity_table(y_valid, raw_probability, (.10,)).iloc[0]
        candidates.append({"model": name, "calibration": "none", "recall_at_10pct": raw_top.recall_at_k,
                           "lift_at_10pct": raw_top.lift_at_k, **raw})
        calibrated = CalibratedClassifierCV(raw_model, method="sigmoid", cv=3, ensemble=False)
        calibrated.fit(x_train, y_train)
        probability = calibrated.predict_proba(x_valid)[:, 1]
        met = classification_metrics(y_valid, probability)
        top = capacity_table(y_valid, probability, (.10,)).iloc[0]
        candidates.append({"model": name, "calibration": "sigmoid", "recall_at_10pct": top.recall_at_k,
                           "lift_at_10pct": top.lift_at_k, **met})
    table = pd.DataFrame(candidates)
    eligible = []
    for name in fitted:
        sub = table.loc[table.model.eq(name)]
        base = sub.loc[sub.calibration.eq("none")].iloc[0]
        cal = sub.loc[sub.calibration.eq("sigmoid")].iloc[0]
        eligible.append(cal if base.brier_score - cal.brier_score >= .002 else base)
    eligible = pd.DataFrame(eligible)
    # PR AUC is primary. Differences under 0.01 are treated as close; Top-10%
    # capture then Brier break ties, then the interpretable baseline if close.
    best_pr = eligible.pr_auc.max()
    close = eligible.loc[eligible.pr_auc.ge(best_pr - .01)]
    best_top = close.recall_at_10pct.max()
    close = close.loc[close.recall_at_10pct.ge(best_top - .02)]
    best_brier = close.brier_score.min()
    near = close.loc[close.brier_score.le(best_brier + .003)]
    chosen = near.loc[near.model.eq("Logistic Regression")].iloc[0] if near.model.eq("Logistic Regression").any() else near.sort_values("brier_score").iloc[0]
    return str(chosen.model), str(chosen.calibration), table


def _threshold(y, probability, target_recall=.80):
    """Highest validation-score threshold meeting a stated 80% recall scenario."""
    positives = np.asarray(probability)[np.asarray(y) == 1]
    if not len(positives):
        raise ValueError("Validation has no churn cases")
    rank = max(1, int(np.ceil(target_recall * len(positives))))
    return float(np.sort(positives)[::-1][rank - 1])


def _curve_figure(y, probability, destination, kind):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if kind == "roc":
        RocCurveDisplay.from_predictions(y, probability, ax=ax)
        ax.plot([0, 1], [0, 1], "--", color="gray")
    else:
        PrecisionRecallDisplay.from_predictions(y, probability, ax=ax)
        ax.axhline(float(np.mean(y)), linestyle="--", color="gray", label="Prevalence")
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _calibration_figure(bins, destination):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(bins.mean_predicted_risk, bins.observed_churn_rate, "o-", label="Test bins")
    ax.set(xlabel="Mean predicted risk", ylabel="Observed churn rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _explanations(model, model_name, x, scored, reference):
    """Reference contrasts show associations, not causal effects or SHAP values."""
    base_model = model.calibrated_classifiers_[0].estimator if isinstance(model, CalibratedClassifierCV) else model
    if model_name == "Logistic Regression":
        names = base_model.named_steps["preprocess"].get_feature_names_out()
        coeff = base_model.named_steps["model"].coef_[0]
        importance = pd.DataFrame({"feature": names, "importance": np.abs(coeff), "coefficient_log_odds": coeff})
    else:
        importance = pd.DataFrame({"feature": base_model.feature_names_, "importance": base_model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    refs = {column: (reference[column].median() if column in NUMERIC else reference[column].mode().iloc[0]) for column in FEATURES}
    # Only the four most globally influential raw fields are contrasted per customer.
    selected = []
    for label in importance.feature:
        raw = label.split("__", 1)[-1]
        match = next((f for f in FEATURES if raw == f or raw.startswith(f + "_")), None)
        if match and match not in selected:
            selected.append(match)
        if len(selected) == 4:
            break
    effects = []
    actual = scored.churn_probability.to_numpy()
    for column in selected:
        counter = x.copy()
        counter[column] = refs[column]
        contrast = actual - model.predict_proba(counter)[:, 1]
        effects.append((column, contrast))
    positive, negative = [], []
    for i in range(len(x)):
        ranked = sorted(((column, effect[i]) for column, effect in effects), key=lambda pair: pair[1], reverse=True)
        positive.append(", ".join(column for column, effect in ranked if effect > .005) or "None among checked features")
        negative.append(", ".join(column for column, effect in reversed(ranked) if effect < -.005) or "None among checked features")
    scored["strongest_risk_increasing_associations"] = positive
    scored["strongest_risk_reducing_associations"] = negative
    return importance


def train_and_export():
    data = load_customer_data(RAW_DATA)
    x, y = model_matrix(data), data.observed_churn
    development_idx, test_idx = train_test_split(np.arange(len(data)), test_size=.20, stratify=y, random_state=SEED)
    train_idx, valid_idx = train_test_split(development_idx, test_size=.25, stratify=y.iloc[development_idx], random_state=SEED)
    assert not (set(train_idx) & set(valid_idx) or set(train_idx) & set(test_idx) or set(valid_idx) & set(test_idx))
    fitted, cv_table = _tune(x.iloc[train_idx], y.iloc[train_idx])
    name, calibration, validation_table = _choose(fitted, x.iloc[train_idx], y.iloc[train_idx], x.iloc[valid_idx], y.iloc[valid_idx])
    validation_model = fitted[name] if calibration == "none" else CalibratedClassifierCV(fitted[name], method="sigmoid", cv=3, ensemble=False).fit(x.iloc[train_idx], y.iloc[train_idx])
    validation_probability = validation_model.predict_proba(x.iloc[valid_idx])[:, 1]
    threshold = _threshold(y.iloc[valid_idx], validation_probability)
    risk_cutoffs = {"high_cutoff": float(np.quantile(validation_probability, .90)),
                    "medium_cutoff": float(np.quantile(validation_probability, .70))}
    priority_cutoff = float(np.quantile(validation_probability * data.MonthlyCharges.iloc[valid_idx].to_numpy(), .90))
    # Analytical choices are frozen here. Only now refit on development data and score test once.
    final = fitted[name].set_params()  # best parameters are already frozen by training-only CV
    final = CalibratedClassifierCV(final, method="sigmoid", cv=3, ensemble=False) if calibration == "sigmoid" else final
    final.fit(x.iloc[development_idx], y.iloc[development_idx])
    test_probability = final.predict_proba(x.iloc[test_idx])[:, 1]
    test_data = data.iloc[test_idx].reset_index(drop=True)
    test_x = x.iloc[test_idx].reset_index(drop=True)
    test_y = y.iloc[test_idx].to_numpy()
    metrics = classification_metrics(test_y, test_probability, threshold)
    intervals = bootstrap_intervals(test_y, test_probability, threshold=threshold)
    capacity = capacity_table(test_y, test_probability)
    bins = calibration_bins(test_y, test_probability)
    scored = scored_customers(test_data, test_probability, MODEL_VERSION,
                              risk_cutoffs=risk_cutoffs, priority_cutoff=priority_cutoff)
    subgroup = subgroup_diagnostics(test_data, test_probability, threshold)
    importance = _explanations(final, name, test_x, scored, x.iloc[development_idx])
    segments = []
    for column in ("Contract", "InternetService", "PaymentMethod"):
        tmp = pd.DataFrame({"category": test_data[column], "observed_churn": test_y, "churn_probability": test_probability,
                            "monthly_charge_exposure_proxy": test_data.MonthlyCharges})
        summary = tmp.groupby("category", observed=True).agg(customer_count=("observed_churn", "size"),
            observed_churn_rate=("observed_churn", "mean"), mean_predicted_risk=("churn_probability", "mean"),
            monthly_charge_exposure_proxy_total=("monthly_charge_exposure_proxy", "sum")).reset_index()
        summary.insert(0, "segment", column)
        segments.append(summary)
    segment = pd.concat(segments, ignore_index=True)
    for folder in (EXPORTS, REPORTS / "metrics", REPORTS / "figures"):
        folder.mkdir(parents=True, exist_ok=True)
    output_tables = {
        "customers_scored": scored, "segment_summary": segment,
        "campaign_capacity": capacity, "calibration_bins": bins,
        "subgroup_diagnostics": subgroup,
    }
    model_metrics = validation_table.assign(
        model_version=MODEL_VERSION, split="validation",
        selection_status=np.where(validation_table.model.eq(name) & validation_table.calibration.eq(calibration),
                                  "selected", "candidate"),
    )
    test_row = pd.DataFrame([{"model_version": MODEL_VERSION, "split": "untouched_test",
                              "model": name, "calibration": calibration,
                              "selection_status": "selected", **metrics}])
    model_metrics = pd.concat([model_metrics, test_row], ignore_index=True)
    output_tables["model_metrics"] = model_metrics
    for key, table in output_tables.items():
        if "model_version" not in table:
            table.insert(0, "model_version", MODEL_VERSION)
        table.to_csv(EXPORTS / f"{key}.csv", index=False)
    cv_table.to_csv(REPORTS / "metrics" / "training_cv.csv", index=False)
    validation_table.to_csv(REPORTS / "metrics" / "validation_candidates.csv", index=False)
    importance.to_csv(REPORTS / "metrics" / "global_importance.csv", index=False)
    quality = {"rows": len(data), "columns_in_raw": 21, "unique_customer_ids": int(data.customerID.nunique()),
               "source_churn_rate": float(data.observed_churn.mean()),
               "missing_after_cleaning": int(data[list(FEATURES)].isna().sum().sum()),
               "zero_tenure_total_charges_zero": int(data.tenure.eq(0).sum()),
               "feature_summary": {
                   "tenure": data.tenure.describe().to_dict(),
                   "MonthlyCharges": data.MonthlyCharges.describe().to_dict(),
                   "TotalCharges": data.TotalCharges.describe().to_dict(),
                   "contract_counts": data.Contract.value_counts().to_dict(),
                   "internet_service_counts": data.InternetService.value_counts().to_dict(),
               }}
    (REPORTS / "metrics" / "data_quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    pd.Series(test_probability, name="churn_probability").describe().to_csv(REPORTS / "metrics" / "test_score_distribution.csv")
    _curve_figure(test_y, test_probability, REPORTS / "figures" / "roc.png", "roc")
    _curve_figure(test_y, test_probability, REPORTS / "figures" / "precision_recall.png", "pr")
    _calibration_figure(bins, REPORTS / "figures" / "calibration.png")
    artifact = {"model": final, "model_name": name, "calibration": calibration,
                "threshold": threshold, "model_version": MODEL_VERSION, "feature_columns": list(FEATURES),
                "risk_cutoffs": risk_cutoffs, "priority_cutoff": priority_cutoff}
    joblib.dump(artifact, REPORTS / "metrics" / "final_model.joblib")
    manifest = {"model_version": MODEL_VERSION, "raw_sha256": hashlib.sha256(RAW_DATA.read_bytes()).hexdigest(),
                "seed": SEED, "train_n": len(train_idx), "validation_n": len(valid_idx), "test_n": len(test_idx),
                "selected_model": name, "calibration": calibration, "validation_threshold_for_80pct_recall": threshold,
                "risk_cutoffs_from_validation": risk_cutoffs, "priority_cutoff_from_validation": priority_cutoff,
                "test_metrics": metrics, "bootstrap_95pct": intervals,
                "scoring_scope": "Untouched test customers only; observed labels are for evaluation, not model features."}
    (REPORTS / "metrics" / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
