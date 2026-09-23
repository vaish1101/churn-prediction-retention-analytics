"""Score a separate customer cohort with the frozen fitted model."""

import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churn.analytics import scored_customers
from churn.config import REPORTS
from churn.data import load_customer_data, model_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path, help="CSV with customerID and all model features")
    parser.add_argument("output_csv", type=Path, help="Destination CSV; never overwrites source")
    args = parser.parse_args()
    if args.input_csv.resolve() == args.output_csv.resolve() or args.output_csv.exists():
        parser.error("Output must be a new path distinct from input")
    artifact = joblib.load(REPORTS / "metrics" / "final_model.joblib")
    data = load_customer_data(args.input_csv, require_target=False)
    probability = artifact["model"].predict_proba(model_matrix(data))[:, 1]
    output = scored_customers(data, probability, artifact["model_version"],
                              risk_cutoffs=artifact["risk_cutoffs"],
                              priority_cutoff=artifact["priority_cutoff"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Scored {len(output)} customers to {args.output_csv}")


if __name__ == "__main__":
    main()
