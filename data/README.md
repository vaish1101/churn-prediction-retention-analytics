# Data provenance

The local raw file is `data/raw/telco_customer_churn.csv`, moved without changing its bytes from `data 12.23.40 12.23.40.csv`.

- SHA-256: `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`
- 7,043 customer rows; 21 source columns; target `Churn` (1,869 Yes; 5,174 No).
- The schema and example records match the IBM Telco Customer Churn sample. IBM describes its Telco sample as data for a fictional telecom company: <https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113>.
- Exact redistribution terms for this particular local CSV have **not** been established. The raw CSV is excluded from Git by default pending a source/license decision. No license is asserted.
- For a fresh local setup, IBM hosts a corresponding 7,043-row CSV in its [watsonx-ai-samples repository](https://github.com/IBM/watsonx-ai-samples/blob/master/cpd4.5/data/customer_churn/WA_FnUseC_TelcoCustomerChurn.csv). Obtain it from IBM, place it at `data/raw/telco_customer_churn.csv`, and verify schema, row count, and checksum against this project's expected source. A different checksum must be investigated before treating a run as a reproduction.
- The 11 whitespace-only `TotalCharges` values all have `tenure = 0`. Validation replaces those with zero in memory; any other parse error fails loudly. The raw file remains unchanged.

This is a static snapshot. It does not establish a prospective observation window or a 30/90-day forecasting horizon.
