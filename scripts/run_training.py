"""Rebuild analytical outputs from the raw local CSV."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churn.workflow import train_and_export


if __name__ == "__main__":
    print(json.dumps(train_and_export(), indent=2))

