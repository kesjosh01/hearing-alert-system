"""Opt-in W&B scalar recording; never upload training files or persist a key."""
from __future__ import annotations

import importlib
import os

from .common import WANDB_MODES

EPOCH_FIELDS = (
    "epoch", "train_loss", "domestic_target_recall", "overseas_other_fpr",
    "domestic_doorlock_fp", "validation_guards_passed", "saved",
)
CONFIG_FIELDS = (
    "epochs", "patience", "batch_size", "steps_per_epoch", "learning_rate", "domestic_weight",
)


def validate_options(args):
    """Reject incomplete online requests before reading audio or creating a run."""
    if args.wandb_mode not in WANDB_MODES:
        raise ValueError(f"Invalid W&B mode: {args.wandb_mode}")
    if args.wandb_mode == "disabled":
        return
    if not args.wandb_project or not args.wandb_project.strip():
        raise ValueError("W&B recording requires a nonempty --wandb-project")
    if args.wandb_mode == "online":
        if not args.wandb_entity or not args.wandb_entity.strip():
            raise ValueError("Online W&B requires an explicit --wandb-entity")
        if not os.environ.get("WANDB_API_KEY", "").strip():
            raise RuntimeError("Set WANDB_API_KEY in this terminal before online W&B recording")


def start_run(args, options, directory):
    """Start recording after the existing annotation/data/model checks pass."""
    validate_options(args)
    if args.wandb_mode == "disabled":
        return None
    run = None
    try:
        wandb = importlib.import_module("wandb")
        if wandb.run is not None:
            raise RuntimeError("An existing W&B run must be finished before DS-CNN recording")
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            mode=args.wandb_mode,
            dir=str(directory),
            config={
                "model": "DS-CNN", "fold": args.fold, "seed": options["seed"] + args.fold,
                **{name: options[name] for name in CONFIG_FIELDS},
            },
            settings=wandb.Settings(
                console="off", capture_loggers={}, disable_code=True, save_code=False,
                disable_git=True, disable_job_creation=True, config_paths=(),
                # These internal fields are verified with the installed wandb 0.31.0.
                x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True,
                x_save_requirements=False,
                finish_timeout=30.0, finish_timeout_raises=True,
            ),
        )
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")
        return run
    except Exception as error:
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass  # Keep the initialization error as the first fault.
        # Do not copy credentials or arbitrary SDK exception text into our logs.
        raise RuntimeError(f"W&B initialization failed ({type(error).__name__})") from None


def log_epoch(run, row):
    if run is None:
        return
    payload = {name: row[name] for name in EPOCH_FIELDS}
    try:
        run.log(payload, step=int(row["epoch"]), commit=True)
    except Exception as error:
        raise RuntimeError(f"W&B epoch logging failed ({type(error).__name__})") from None


def finish_run(run, status):
    """Record sync failure separately without replacing the training first fault."""
    if run is None:
        return
    tracking = status["wandb"]
    tracking["finished"] = False
    try:
        run.finish(exit_code=0 if status["exit_code"] == 0 else 1)
    except Exception as error:
        tracking["finish_error_type"] = type(error).__name__
    else:
        tracking["finished"] = True
