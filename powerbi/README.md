# Power BI dashboard contract

The two-page Power BI report uses validated analytical exports from `powerbi/exports/`:

- **Retention Overview** presents customer risk, priority, segments, exposure, and suggested retention actions.
- **Model & Campaign Performance** presents final test metrics, discrimination, probability quality, validation model comparison, and campaign capacity results.

The final page images are published under `reports/screenshots/`. The local PBIX is intentionally excluded from Git because its embedded `DataModel` contains customer-level records derived from the source dataset. The standalone PBIP descriptor is also excluded because its referenced report and semantic-model project folders were not supplied. This preserves a useful public dashboard record without redistributing the source data or publishing an incomplete Power BI project.

## Export contract

Run `python scripts/run_training.py` to generate analytical outputs, then `python scripts/build_powerbi_exports.py` to validate the dashboard contract. All exports include `model_version`.

Public aggregate tables:

- `segment_summary.csv`
- `campaign_capacity.csv`
- `model_metrics.csv`
- `calibration_bins.csv`
- `subgroup_diagnostics.csv`

`customers_scored.csv` is required locally for the customer-level report page but is excluded from Git. It contains the untouched test cohort, customer identifiers, and observed labels used for retrospective evaluation. It must not be treated as a live campaign list.

Risk cutoffs and the priority cutoff were frozen from validation scores. The priority score is churn probability multiplied by the Monthly Charge Exposure Proxy. It is not customer lifetime value, profit, or expected revenue saved. Suggested retention actions are deterministic discussion prompts, not interventions with measured causal effects.

Field definitions and limitations are documented in `metric_dictionary.csv`.
