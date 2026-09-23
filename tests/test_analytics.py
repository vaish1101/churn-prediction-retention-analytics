import numpy as np
import pandas as pd
import pytest

from churn.analytics import (
    assign_bands, bootstrap_intervals, calibration_bins, capacity_table,
    classification_metrics, optional_cost_scenario, scored_customers, subgroup_diagnostics, suggested_action,
)
from churn.config import RAW_DATA
from churn.data import load_customer_data


def test_metric_probabilities_not_hard_labels():
    y = [0, 0, 1, 1]
    p = [.1, .4, .6, .9]
    m = classification_metrics(y, p, .5)
    assert m["roc_auc"] == 1
    assert m["pr_auc"] == 1
    assert m["tp"] == 2 and m["tn"] == 2
    assert m["specificity"] == 1
    with pytest.raises(ValueError):
        classification_metrics(y, [0, 0, 0, 1.1])


def test_top_k_and_lift_hand_calculation():
    y = [1, 0, 1, 0]
    p = [.9, .8, .7, .1]
    table = capacity_table(y, p, (.25, .5))
    first = table.iloc[0]
    assert first.capacity_n == 1
    assert first.captured_churners == 1
    assert first.precision_at_k == 1
    assert first.recall_at_k == .5
    assert first.lift_at_k == 2
    assert table.iloc[1].recall_at_k == .5


def test_calibration_bins_reconcile():
    y = [0, 1, 0, 1, 1]
    p = [.05, .22, .22, .9, 1.0]
    bins = calibration_bins(y, p, 10)
    assert bins.n.sum() == len(y)
    assert bins.observed_churn_rate.between(0, 1).all()
    assert bins.mean_predicted_risk.between(0, 1).all()


def test_capacity_bands_are_deterministic():
    scores = [.8, .1, .8, .3] * 25
    rank_a, bands_a = assign_bands(scores)
    rank_b, bands_b = assign_bands(scores)
    np.testing.assert_array_equal(rank_a, rank_b)
    np.testing.assert_array_equal(bands_a, bands_b)
    assert sum(bands_a == "High") == 10
    assert sum(bands_a == "Medium") == 20


def test_frozen_validation_cutoffs_do_not_depend_on_batch_size():
    rank, band = assign_bands([.61], high_cutoff=.70, medium_cutoff=.40)
    assert rank[0] == 1
    assert band[0] == "Medium"


def test_scored_customer_export_and_priority():
    data = load_customer_data(RAW_DATA).iloc[:100].copy()
    probability = np.linspace(.01, .99, 100)
    scored = scored_customers(data, probability, "test-version")
    required = {"customer_id", "model_version", "observed_churn", "churn_probability",
                "risk_rank", "risk_band", "monthly_charge_exposure_proxy", "priority_score",
                "priority_rank", "priority_band", "suggested_retention_action"}
    assert required <= set(scored.columns)
    assert scored[list(required)].notna().all().all()
    assert scored.customer_id.is_unique
    assert scored.priority_rank.is_unique
    assert scored.priority_score.between(0, 200).all()
    assert scored.loc[scored.risk_rank.eq(1), "risk_band"].iloc[0] == "High"


def test_action_is_suggestion_not_causal_claim():
    row = pd.Series({"risk_band": "High", "Contract": "Month-to-month", "TechSupport": "Yes",
                     "OnlineSecurity": "Yes", "MonthlyCharges": 80})
    text = suggested_action(row)
    assert "Consider" in text
    assert "prevent" not in text.lower()


def test_bootstrap_intervals_are_bounded_and_deterministic():
    y = np.array([0, 1] * 50)
    p = np.linspace(.05, .95, 100)
    a = bootstrap_intervals(y, p, seed=7, iterations=50)
    b = bootstrap_intervals(y, p, seed=7, iterations=50)
    assert a == b
    assert all(0 <= interval["lower"] <= interval["upper"] <= 1 for interval in a.values())


def test_subgroups_reconcile():
    data = load_customer_data(RAW_DATA).iloc[:200].copy()
    probability = np.full(len(data), .25)
    groups = subgroup_diagnostics(data, probability)
    assert groups.loc[groups.group.eq("Contract"), "n"].sum() == len(data)
    assert groups.n.gt(0).all()


def test_cost_scenario_is_disabled_without_assumptions():
    data = load_customer_data(RAW_DATA).iloc[:100]
    scored = scored_customers(data, np.full(len(data), .3), "test")
    assert optional_cost_scenario(scored) is None
    with pytest.raises(ValueError, match="SCENARIO ASSUMPTIONS"):
        optional_cost_scenario(scored, contact_cost=1)
    scenario = optional_cost_scenario(scored, contact_cost=1, intervention_cost=2,
                                      assumed_save_rate=.1, value_horizon_months=6)
    assert scenario["contacted_customers"] == 10
    assert "SCENARIO ASSUMPTIONS" in scenario["label"]
