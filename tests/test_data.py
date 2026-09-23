import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from churn.config import RAW_DATA, SEED
from churn.data import FEATURES, load_customer_data, model_matrix
from churn.workflow import logistic_estimator


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(RAW_DATA, dtype={"SeniorCitizen": "string", "TotalCharges": "string"})


def write_bad(tmp_path, frame):
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    return path


def test_source_profile_and_total_charges():
    data = load_customer_data(RAW_DATA)
    assert data.shape[0] == 7043
    assert data.observed_churn.sum() == 1869
    assert data.customerID.is_unique
    assert data.loc[data.tenure.eq(0), "TotalCharges"].eq(0).all()
    assert data.TotalCharges.dtype.kind == "f"


def test_missing_raw_file_has_setup_guidance(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_customer_data(tmp_path / "missing.csv")


def test_schema_missing_column(tmp_path, raw):
    with pytest.raises(ValueError, match="Schema mismatch"):
        load_customer_data(write_bad(tmp_path, raw.drop(columns="Contract")))


def test_duplicate_customer_id(tmp_path, raw):
    bad = raw.copy()
    bad.loc[1, "customerID"] = bad.loc[0, "customerID"]
    with pytest.raises(ValueError, match="customerID"):
        load_customer_data(write_bad(tmp_path, bad))


def test_unexpected_category(tmp_path, raw):
    bad = raw.copy()
    bad.loc[0, "Contract"] = "Unknown contract"
    with pytest.raises(ValueError, match="Unexpected Contract"):
        load_customer_data(write_bad(tmp_path, bad))


def test_total_charges_parse_failure_is_not_zero(tmp_path, raw):
    bad = raw.copy()
    bad.loc[0, "TotalCharges"] = "garbled"
    with pytest.raises(ValueError, match="TotalCharges"):
        load_customer_data(write_bad(tmp_path, bad))


def test_total_charges_blank_only_at_zero_tenure(tmp_path, raw):
    bad = raw.copy()
    bad.loc[0, "TotalCharges"] = " "
    with pytest.raises(ValueError, match="Blank TotalCharges"):
        load_customer_data(write_bad(tmp_path, bad))


def test_no_id_or_target_in_features():
    data = load_customer_data(RAW_DATA)
    x = model_matrix(data)
    assert list(x.columns) == list(FEATURES)
    assert not {"customerID", "Churn", "observed_churn"} & set(x.columns)


def test_deterministic_stratified_nonoverlap():
    data = load_customer_data(RAW_DATA)
    index = list(range(len(data)))
    a, b = train_test_split(index, test_size=.2, random_state=SEED, stratify=data.observed_churn)
    c, d = train_test_split(index, test_size=.2, random_state=SEED, stratify=data.observed_churn)
    assert a == c and b == d
    assert not set(a) & set(b)
    assert len(set(a) | set(b)) == len(data)


def test_logistic_pipeline_ignores_unseen_category():
    data = load_customer_data(RAW_DATA)
    x = model_matrix(data.iloc[:500])
    y = data.observed_churn.iloc[:500]
    model = logistic_estimator().fit(x, y)
    novel = x.iloc[:1].copy()
    novel.loc[novel.index[0], "Contract"] = "novel at scoring"
    score = model.predict_proba(novel)[0, 1]
    assert 0 <= score <= 1


def test_scoring_input_without_target(tmp_path, raw):
    path = write_bad(tmp_path, raw.head(1).drop(columns="Churn"))
    data = load_customer_data(path, require_target=False)
    assert "observed_churn" not in data
    assert len(model_matrix(data)) == 1
