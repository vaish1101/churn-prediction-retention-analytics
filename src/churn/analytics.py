"""Metrics, calibration, targeting and deterministic business suggestions."""

import math
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve,
)

from .config import CAPACITIES


def classification_metrics(y, probability, threshold=0.5):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y) != len(probability) or len(y) == 0 or not np.isfinite(probability).all():
        raise ValueError("Labels and finite probabilities must align")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be in [0, 1]")
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y), "observed_churn_rate": float(y.mean()),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "roc_auc": roc_auc_score(y, probability) if len(np.unique(y)) == 2 else None,
        "pr_auc": average_precision_score(y, probability) if y.sum() else None,
        "brier_score": brier_score_loss(y, probability),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


def capacity_table(y, probability, capacities=CAPACITIES):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y) != len(probability) or not len(y):
        raise ValueError("Labels and scores must align")
    ranked = np.argsort(-probability, kind="stable")
    prevalence = float(y.mean())
    rows = []
    for fraction in capacities:
        if not 0 < fraction <= 1:
            raise ValueError("Capacity fractions must be in (0,1]")
        k = min(len(y), max(1, math.ceil(len(y) * fraction)))
        captured = int(y[ranked[:k]].sum())
        precision = captured / k
        recall = captured / int(y.sum()) if y.sum() else 0.0
        rows.append({
            "capacity_pct": float(fraction), "capacity_n": k,
            "captured_churners": captured,
            "precision_at_k": precision, "recall_at_k": recall,
            "lift_at_k": precision / prevalence if prevalence else None,
            "cumulative_gain": recall,
            "random_expected_churners": float(k * prevalence),
        })
    return pd.DataFrame(rows)


def calibration_bins(y, probability, n_bins=10):
    frame = pd.DataFrame({"observed_churn": np.asarray(y, dtype=int), "probability": np.asarray(probability, dtype=float)})
    if not frame.probability.between(0, 1).all():
        raise ValueError("Probabilities must be in [0, 1]")
    frame["bin"] = np.minimum((frame.probability * n_bins).astype(int), n_bins - 1)
    grouped = frame.groupby("bin", observed=True).agg(
        n=("observed_churn", "size"), mean_predicted_risk=("probability", "mean"),
        observed_churn_rate=("observed_churn", "mean"),
    ).reset_index()
    grouped["bin_lower"] = grouped["bin"] / n_bins
    grouped["bin_upper"] = (grouped["bin"] + 1) / n_bins
    assert grouped.n.sum() == len(frame)
    return grouped


def bootstrap_intervals(y, probability, *, threshold=0.5, fraction=0.10, seed=42, iterations=1000):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    generator = np.random.default_rng(seed)
    values = {key: [] for key in ("roc_auc", "pr_auc", "recall", "recall_at_10pct")}
    for _ in range(iterations):
        idx = generator.integers(0, len(y), len(y))
        yi, pi = y[idx], probability[idx]
        if len(np.unique(yi)) < 2:
            continue
        metrics = classification_metrics(yi, pi, threshold)
        top = capacity_table(yi, pi, (fraction,)).iloc[0]
        for key in values:
            values[key].append(top.recall_at_k if key == "recall_at_10pct" else metrics[key])
    return {key: {"lower": float(np.quantile(v, .025)), "upper": float(np.quantile(v, .975))} for key, v in values.items()}


def assign_bands(probability, *, high_cutoff=None, medium_cutoff=None):
    """Apply frozen validation percentile cutoffs, or relative ranks for diagnostics."""
    scores = np.asarray(probability, dtype=float)
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(len(scores), dtype=int)
    rank[order] = np.arange(1, len(scores) + 1)
    high = math.ceil(.10 * len(scores))
    medium = math.ceil(.30 * len(scores))
    if (high_cutoff is None) != (medium_cutoff is None):
        raise ValueError("Both frozen risk cutoffs are required together")
    if high_cutoff is None:
        bands = np.where(rank <= high, "High", np.where(rank <= medium, "Medium", "Low"))
    else:
        if not 0 <= medium_cutoff <= high_cutoff <= 1:
            raise ValueError("Risk cutoffs must satisfy 0 <= medium <= high <= 1")
        bands = np.where(scores >= high_cutoff, "High", np.where(scores >= medium_cutoff, "Medium", "Low"))
    return rank, bands


def suggested_action(row):
    """Associative suggestions only; no treatment-effect claim."""
    if row["risk_band"] != "High" and row.get("priority_band") != "Top 10%":
        return "No priority outreach under the 10% capacity scenario"
    if row["Contract"] == "Month-to-month":
        return "Consider contract-options outreach"
    if row["TechSupport"] == "No" or row["OnlineSecurity"] == "No":
        return "Consider proactive service review"
    if row["MonthlyCharges"] >= 80:
        return "Consider plan or bundle review"
    return "Consider a retention check-in"


def scored_customers(frame, probability, model_version, *, risk_cutoffs=None, priority_cutoff=None):
    """Score an evaluation cohort; priority is expected monthly-charge exposure."""
    if len(frame) != len(probability):
        raise ValueError("Scores must align with customer rows")
    out = frame.loc[:, ["customerID", "Contract", "TechSupport", "OnlineSecurity", "MonthlyCharges", "tenure"]].copy()
    out.insert(0, "model_version", model_version)
    out = out.rename(columns={"customerID": "customer_id"})
    if "observed_churn" in frame:
        out["observed_churn"] = frame.observed_churn.to_numpy()
    out["churn_probability"] = np.asarray(probability, dtype=float)
    out["risk_rank"], out["risk_band"] = assign_bands(probability, **(risk_cutoffs or {}))
    out["monthly_charge_exposure_proxy"] = out.MonthlyCharges
    out["priority_score"] = out.churn_probability * out.monthly_charge_exposure_proxy
    priority_order = np.argsort(-out.priority_score.to_numpy(), kind="stable")
    priority_rank = np.empty(len(out), dtype=int)
    priority_rank[priority_order] = np.arange(1, len(out) + 1)
    out["priority_rank"] = priority_rank
    if priority_cutoff is None:
        out["priority_band"] = np.where(priority_rank <= math.ceil(.10 * len(out)), "Top 10%", "Other")
    else:
        out["priority_band"] = np.where(out.priority_score >= priority_cutoff, "Top 10%", "Other")
    out["suggested_retention_action"] = out.apply(suggested_action, axis=1)
    return out


def optional_cost_scenario(scored, *, contact_cost=None, intervention_cost=None,
                           assumed_save_rate=None, value_horizon_months=None,
                           capacity_pct=.10):
    """Disabled until every explicitly labeled scenario assumption is supplied.

    This is an illustrative arithmetic scenario, not observed campaign ROI or
    a claim that any action causes retention.
    """
    assumptions = (contact_cost, intervention_cost, assumed_save_rate, value_horizon_months)
    if all(value is None for value in assumptions):
        return None
    if any(value is None for value in assumptions):
        raise ValueError("All SCENARIO ASSUMPTIONS are required to enable the scenario")
    if contact_cost < 0 or intervention_cost < 0 or not 0 <= assumed_save_rate <= 1 or value_horizon_months <= 0:
        raise ValueError("Invalid SCENARIO ASSUMPTIONS")
    if not 0 < capacity_pct <= 1:
        raise ValueError("Invalid campaign capacity")
    k = math.ceil(len(scored) * capacity_pct)
    targeted = scored.nsmallest(k, "priority_rank")
    gross_proxy = float((targeted.churn_probability * targeted.monthly_charge_exposure_proxy).sum()
                        * assumed_save_rate * value_horizon_months)
    expense = float(k * (contact_cost + intervention_cost))
    return {"label": "SCENARIO ASSUMPTIONS: not observed company economics",
            "capacity_pct": capacity_pct, "contacted_customers": k,
            "contact_cost_assumption": contact_cost, "intervention_cost_assumption": intervention_cost,
            "assumed_save_rate": assumed_save_rate, "value_horizon_months": value_horizon_months,
            "gross_charge_exposure_proxy": gross_proxy, "assumed_expense": expense,
            "net_scenario_proxy": gross_proxy - expense}


def subgroup_diagnostics(frame, probability, threshold=0.5):
    rows = []
    groups = {
        "Contract": frame.Contract.astype(str),
        "InternetService": frame.InternetService.astype(str),
        "PaymentMethod": frame.PaymentMethod.astype(str),
        "SeniorCitizen": frame.SeniorCitizen.astype(str),
        "tenure_band": pd.cut(frame.tenure, [-1, 12, 36, 72], labels=["0-12 months", "13-36 months", "37-72 months"]).astype(str),
    }
    y = frame.observed_churn.to_numpy()
    for name, categories in groups.items():
        for category in sorted(categories.unique()):
            mask = categories.eq(category).to_numpy()
            metrics = classification_metrics(y[mask], np.asarray(probability)[mask], threshold)
            rows.append({"group": name, "category": category, "n": int(mask.sum()),
                         "prevalence": metrics["observed_churn_rate"],
                         "recall": metrics["recall"], "precision": metrics["precision"],
                         "roc_auc": metrics["roc_auc"] if mask.sum() >= 50 and len(np.unique(y[mask])) == 2 else None,
                         "brier_score": metrics["brier_score"]})
    return pd.DataFrame(rows)
