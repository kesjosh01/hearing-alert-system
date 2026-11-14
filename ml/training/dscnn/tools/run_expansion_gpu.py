"""Use the already installed DS-CNN CUDA libraries for one child process only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/expansion_20261114"
GPU = "GPU-10f7f4b0-eee7-9d43-26de-7bbefdbfc7ce"


def main(mode):
    python = ROOT / ".venv/bin/python"
    site = ROOT / ".venv/lib/python3.12/site-packages"
    libraries = sorted(str(p) for p in (site / "nvidia").glob("*/lib") if p.is_dir())
    if len(libraries) != 12 or not python.is_file():
        raise ValueError("Existing validated DS-CNN environment is incomplete; do not install automatically")
    visible = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
    if GPU not in visible.splitlines() or active.strip():
        raise RuntimeError("Selected GPU unavailable or has existing compute process; do not share/terminate it")
    env = dict(os.environ)
    env.update(VIRTUAL_ENV=str(ROOT / ".venv"), CUDA_VISIBLE_DEVICES=GPU,
               PATH=os.pathsep.join([str(site / "nvidia/cuda_nvcc/bin"), str(python.parent), env.get("PATH", "")]),
               LD_LIBRARY_PATH=os.pathsep.join(libraries + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])),
               CUDA_CACHE_PATH=str(ROOT / ".cache/runtime/cuda"), TF_USE_LEGACY_KERAS="1",
               TF_FORCE_GPU_ALLOW_GROWTH="true", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               TF_XLA_FLAGS="--tf_xla_auto_jit=0", WANDB_MODE="disabled", PYTHONUNBUFFERED="1")
    env.pop("CUDA_FORCE_PTX_JIT", None)
    env.pop("XLA_FLAGS", None)
    command = [str(python), "-B", "-m", "expansion.runtime", mode] if mode in ("preflight", "prepare") else \
              [str(python), "-B", "-m", "expansion.study"] if mode == "study" else \
              [str(python), "-B", str(ROOT / "tools/refit_expanded_with_coverage.py")] if mode == "coverage" else \
              [str(python), "-B", str(ROOT / "tools/evaluate_expanded_model.py")]
    receipt_path = REPORT / f"{mode}_process.json"
    if receipt_path.exists():
        raise FileExistsError("Inspect prior process identity before repeating a stage")
    started = time.monotonic()
    child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    receipt = {"pid": child.pid, "boot_id": Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               "start_ticks": Path(f'/proc/{child.pid}/stat').read_text().split()[21],
               "command": command, "cwd": str(ROOT), "gpu_uuid": GPU,
               "selected_cuda_libraries": libraries, "process_environment_only": True,
               "package_or_driver_changes": False, "existing_compute_processes_before": active.strip(), "exit_code": None}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    receipt.update(exit_code=child.wait(), elapsed_seconds=time.monotonic() - started)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    sys.exit(receipt["exit_code"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "prepare", "study", "coverage", "evaluate"))
    main(parser.parse_args().mode)
