"""Strict validation and deterministic cleaning of the static Telco sample."""

from pathlib import Path

import pandas as pd

NUMERIC = ("tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL = (
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
)
FEATURES = NUMERIC + CATEGORICAL
ALLOWED = {
    "gender": {"Female", "Male"}, "SeniorCitizen": {"0", "1"},
    "Partner": {"Yes", "No"}, "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaymentMethod": {"Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"},
    "PaperlessBilling": {"Yes", "No"},
}
for _field in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"):
    ALLOWED[_field] = {"Yes", "No", "No internet service"}


def load_customer_data(path, *, require_target=True):
    """Read values unchanged, then validate and clean in memory; never overwrite raw CSV."""
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Raw customer CSV not found at {path}. Obtain the IBM Telco sample, "
            "place it at data/raw/telco_customer_churn.csv, and see data/README.md. "
            "The project does not download data automatically."
        )
    frame = pd.read_csv(path, dtype={"customerID": "string", "SeniorCitizen": "string", "TotalCharges": "string"})
    required = set(FEATURES) | {"customerID"} | ({"Churn"} if require_target else set())
    missing = required - set(frame.columns)
    extra = set(frame.columns) - required
    if missing or extra:
        raise ValueError(f"Schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    minimum = 100 if require_target else 1
    if len(frame) < minimum:
        raise ValueError(f"Row count below the documented sanity minimum ({minimum})")
    if frame.customerID.isna().any() or frame.customerID.duplicated().any():
        raise ValueError("customerID must be non-null and unique")
    if frame[list(FEATURES)].isna().any().any():
        raise ValueError("Unexpected missing feature values")
    for column, allowed in ALLOWED.items():
        values = set(frame[column].astype(str).unique())
        if not values <= allowed:
            raise ValueError(f"Unexpected {column} values: {sorted(values - allowed)}")
    for column in ("tenure", "MonthlyCharges"):
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any() or (converted < 0).any():
            raise ValueError(f"Invalid {column}")
        frame[column] = converted
    if not (frame.tenure % 1 == 0).all():
        raise ValueError("tenure must be an integer month count")
    frame["tenure"] = frame.tenure.astype(int)
    raw_total = frame.TotalCharges.astype("string").str.strip()
    blank = raw_total.eq("")
    if (blank & frame.tenure.ne(0)).any():
        raise ValueError("Blank TotalCharges is only permitted when tenure is zero")
    parsed = pd.to_numeric(raw_total.mask(blank, "0"), errors="coerce")
    if parsed.isna().any() or (parsed < 0).any():
        raise ValueError("Unparseable or negative TotalCharges")
    frame["TotalCharges"] = parsed.astype(float)
    frame["SeniorCitizen"] = frame.SeniorCitizen.astype(str)
    for column in CATEGORICAL:
        frame[column] = frame[column].astype(str)
    if require_target:
        if frame.Churn.isna().any() or set(frame.Churn.unique()) != {"Yes", "No"}:
            raise ValueError("Churn must contain only Yes and No")
        frame["observed_churn"] = frame.Churn.map({"No": 0, "Yes": 1}).astype(int)
    return frame


def model_matrix(frame):
    x = frame.loc[:, FEATURES].copy()
    assert "customerID" not in x and "Churn" not in x and "observed_churn" not in x
    return x
