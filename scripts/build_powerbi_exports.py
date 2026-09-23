"""Validate that training has generated the versioned Power BI contract."""

import sys
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from churn.config import EXPORTS, MODEL_VERSION

required = {
    "customers_scored": {"customer_id", "model_version", "churn_probability", "risk_band", "priority_rank", "suggested_retention_action"},
    "segment_summary": {"segment", "category", "customer_count", "observed_churn_rate"},
    "campaign_capacity": {"capacity_pct", "capacity_n", "captured_churners", "precision_at_k", "recall_at_k", "lift_at_k", "cumulative_gain"},
    "model_metrics": {"model_version", "roc_auc", "pr_auc", "brier_score"},
    "calibration_bins": {"bin", "n", "mean_predicted_risk", "observed_churn_rate"},
    "subgroup_diagnostics": {"group", "category", "n", "prevalence"},
}
for name, columns in required.items():
    path = EXPORTS / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}; first run scripts/run_training.py")
    data = pd.read_csv(path)
    if data.empty or not (columns | {"model_version"}) <= set(data.columns):
        raise SystemExit(f"Invalid export {path}")
    if not data.model_version.eq(MODEL_VERSION).all():
        raise SystemExit(f"Model version mismatch in {path}")
    print(f"{name}: {len(data)} rows")
dictionary = root / "powerbi" / "metric_dictionary.csv"
if not dictionary.exists():
    raise SystemExit("Missing metric dictionary")
print(f"Power BI contract verified for {MODEL_VERSION}")
