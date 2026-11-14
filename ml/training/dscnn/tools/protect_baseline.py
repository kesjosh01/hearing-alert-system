"""Record/compare existing YAMNet files and all three environment distributions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YAMNET = ROOT.parent / "yamnet"


def snapshot():
    files = {}
    for directory in ("data", "runs", "config", "sound_baseline", "tools", "reports"):
        for path in sorted((YAMNET / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                files[str(path.relative_to(YAMNET))] = digest.hexdigest()
    packages = {}
    for environment in sorted(YAMNET.glob(".venv*")):
        distributions = []
        for metadata in environment.glob("lib/python*/site-packages/*.dist-info/METADATA"):
            values = metadata.read_text(encoding="utf-8").splitlines()
            distributions.append([next(v[6:] for v in values if v.startswith("Name: ")),
                                  next(v[9:] for v in values if v.startswith("Version: "))])
        packages[environment.name] = sorted(distributions)
    return {"files": files, "packages": packages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    result = snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Every snapshot is append-only; a repeated command must choose a new name.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    comparison = {"protected_files": len(result["files"]), "environments": list(result["packages"])}
    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        comparison["unchanged"] = before == result
        comparison["changed_or_missing"] = [p for p, digest in before["files"].items() if result["files"].get(p) != digest]
        comparison["added"] = sorted(set(result["files"]) - set(before["files"]))
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if comparison.get("unchanged") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
