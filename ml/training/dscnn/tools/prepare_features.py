"""Prepare development log-Mel only; never assign labels to candidate intervals."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import write_json
from dscnn_baseline.data import FeatureStore, development_identity, development_splits


def main():
    rows = development_splits()[0]
    store = FeatureStore()
    started = time.monotonic()
    windows = 0
    for number, row in enumerate(rows, 1):
        values, _ = store.load(row)
        windows += len(values)
        if number % 50 == 0 or number == len(rows):
            print(f"DEVELOPMENT_FEATURES {number}/{len(rows)} windows={windows}", flush=True)
    write_json(ROOT / "reports/real_training_20261114/feature_preparation.json", {
        "complete": True, "kind": "development_feature_preparation_not_training",
        "development_identity": development_identity(), "clips": len(rows), "windows": windows,
        "cache": str(store.cache), "test_audio_decoded": 0,
        "human_review_applied": False, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
