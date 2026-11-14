"""Verify protected YAMNet, raw data, splits and environment inventories."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

from protect_baseline import snapshot


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(report):
    report = Path(report).resolve()
    output = report / "integrity.json"
    after_path = report / "protected_after.json"
    if output.exists() or after_path.exists():
        raise FileExistsError("Preservation receipts already exist; do not overwrite them")
    before = json.loads((report / "protected_before.json").read_text())
    after = snapshot()
    after_path.write_text(json.dumps(after, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    changed = [p for p, value in before["files"].items() if after["files"].get(p) != value]
    added = sorted(set(after["files"]) - set(before["files"]))
    raw_before = json.loads((report / "data_before.json").read_text())
    raw_changed = [r["path"] for r in raw_before
                   if not Path(r["path"]).is_file() or digest(Path(r["path"])) != r["sha256"]]
    packages_before = json.loads((report / "dscnn_packages_before.json").read_text())
    packages_after = sorted([{"name": d.metadata["Name"], "version": d.version}
                             for d in distributions()], key=lambda d: d["name"].lower())
    result = {"kind": "post_training_preservation_audit",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "protected_yamnet_files": len(after["files"]),
              "raw_and_metadata_files": len(raw_before),
              "protected_environments": list(after["packages"]),
              "yamnet_files_and_environments_unchanged": before == after,
              "raw_and_metadata_unchanged": not raw_changed,
              "dscnn_packages_unchanged": packages_before == packages_after,
              "changed_or_missing_yamnet": changed, "added_yamnet": added,
              "changed_or_missing_raw_and_metadata": raw_changed}
    result["unchanged"] = all(result[k] for k in (
        "yamnet_files_and_environments_unchanged", "raw_and_metadata_unchanged", "dscnn_packages_unchanged"))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    verify(parser.parse_args().report)
