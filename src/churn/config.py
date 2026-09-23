"""Project paths and analytical policy fixed before test evaluation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw" / "telco_customer_churn.csv"
REPORTS = ROOT / "reports"
EXPORTS = ROOT / "powerbi" / "exports"
SEED = 42
CAPACITIES = (0.05, 0.10, 0.20, 0.30)
MODEL_VERSION = "churn-0.1.0"

