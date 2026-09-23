# Churn Prediction & Retention Analytics

Predictive modeling and retention analytics system that identifies customers at churn risk, explains key signals, and converts model outputs into prioritized retention decisions.

![Retention Overview Power BI dashboard](reports/screenshots/retention_overview.png)

*Retention Overview connects predicted risk, customer segments, charge exposure, and outreach priorities.*

## Business Problem

Which customers are most likely to churn, why are they at risk, and who should be prioritized when retention capacity is limited?

This decision problem appears across telecom, banking, insurance, utilities, SaaS, automotive services, and B2B businesses wherever companies need to identify attrition risk and prioritize limited retention capacity.

This project uses telecom as the demonstration domain. The trained model itself is not assumed to transfer across sectors; the reusable part is the analytical workflow for risk estimation, prioritization, capacity planning, and explanation.

A churn probability alone is not enough. The workflow identifies high risk customers, examines signals associated with their scores, ranks customers by risk and charge exposure, and shows how campaign capacity changes the target group.

## Key Results

Final performance is measured on an untouched test cohort.

| Measure | Result |
| --- | ---: |
| Held out customers | 1,409 |
| ROC AUC | 0.842 |
| Average Precision | 0.634 |
| Recall | 81.0% |
| Churners captured at 10% capacity | 105 of 374 |
| Lift at 10% capacity | 2.81x |

At the 10% campaign scenario, 141 customers capture 28.1% of observed churners with 74.5% precision.

## Key Insights

- At 10% campaign capacity, 141 customers capture 105 of 374 observed churners with 2.81x lift.
- Month to month customers have 42.6% observed churn prevalence, compared with 2.7% for two year contracts.
- Customers in their first 12 months have 47.9% observed churn prevalence, compared with 10.0% among customers with 37 to 72 months of tenure.
- Fiber optic customers have 41.1% observed churn prevalence, while customers with no internet service have 8.0%.

## How I Solved It

- Validated the IBM Telco sample of 7,043 customers and excluded Customer ID from model features.
- Split the data into 4,225 training, 1,409 validation, and 1,409 untouched test customers.
- Compared a Logistic Regression baseline with a CatBoost challenger using training cross validation and validation data.
- Froze the selected model, threshold, calibration decision, and prioritization cutoffs before final test evaluation.
- Evaluated ranking, probability quality, campaign capacity, customer segments, and uncertainty.
- Converted model outputs into retention priorities, aggregate reporting exports, and a two page Power BI dashboard.

## System Architecture

![System architecture from customer data to retention dashboard](docs/architecture.svg)

## Model Performance & Decision Logic

| Candidate | Validation ROC AUC | Validation Average Precision |
| --- | ---: | ---: |
| Logistic Regression | 0.836 | 0.642 |
| CatBoost | 0.840 | 0.651 |

CatBoost performed slightly better on validation discrimination. The differences in Average Precision, Top 10% recall, and Brier Score were within predefined practical equivalence ranges. Logistic Regression was retained for comparable predictive performance and simpler interpretation.

The 0.251 classification threshold was selected on validation data for an 80% recall scenario. The final test cohort remained untouched until the model, calibration decision, threshold, and targeting rules were frozen.

Probability quality was assessed with reliability bins and Brier Score. Sigmoid calibration did not materially improve validation Brier Score, so the original Logistic Regression probabilities were retained. The highlighted 10% capacity is a planning scenario, not a mathematically proven optimum.

Supporting results: [run manifest](reports/metrics/run_manifest.json), [candidate comparison](reports/metrics/validation_candidates.csv), and [campaign capacity table](powerbi/exports/campaign_capacity.csv).

## Retention Prioritization

```text
Priority Score = Churn Probability × MonthlyCharges
```

`MonthlyCharges` is labeled **Monthly Charge Exposure Proxy**. It is not customer lifetime value, profit, or expected revenue saved.

- **High Risk customers: 158.** Risk Band groups customers by predicted churn probability using validation cutoffs.
- **Top 10% Priority Band customers: 129.** Priority Band combines churn probability with monthly charge exposure using a validation cutoff.
- **10% campaign capacity: 141 customers.** Campaign Capacity selects the highest risk customers for a fixed outreach limit.

These counts represent different decision concepts and are intentionally not interchangeable. Model coefficients, segment summaries, and customer level contrasts describe attributes associated with predicted churn risk. Suggested retention actions are discussion prompts, not interventions with measured causal effects.

## Power BI Dashboard

### Retention Overview

The hero page shows risk bands, customer segments, priority customers, charge exposure, model signals, and the retention priority framework. Slicers support review by contract, internet service, and risk band.

### Model & Campaign Performance

![Model and Campaign Performance Power BI dashboard](reports/screenshots/model_campaign_performance.png)

The second page presents ROC AUC, Average Precision, calibration, campaign capacity, lift, the validation model comparison, and the selected operating point.

The repository includes both screenshots, the [dashboard contract](powerbi/README.md), aggregate reporting exports, and the [metric dictionary](powerbi/metric_dictionary.csv).

## Tech Stack

Python 3.12 · pandas · scikit-learn · CatBoost · matplotlib · pytest · Power BI

## Run Locally

Create a Python 3.12 environment and install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Obtain the IBM sample as described in [data/README.md](data/README.md) and place it at `data/raw/telco_customer_churn.csv`.

```bash
python scripts/run_training.py
python scripts/build_powerbi_exports.py
python -m pytest -q
```

The current release passes all 21 automated tests.

## Limitations

- The fictional dataset is a static snapshot without a rigorous future prediction horizon.
- No treatment outcome data supports causal intervention or savings claims.
- Monthly charge exposure is a prioritization proxy, not customer lifetime value.

The raw source dataset is not included because redistribution rights remain unclear. Repository code and documentation use the [MIT License](LICENSE); dataset rights remain separate.
