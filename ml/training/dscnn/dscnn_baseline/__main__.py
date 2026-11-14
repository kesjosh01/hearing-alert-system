"""No implicit training, downloads, hardware access or test evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import WANDB_MODES, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="Validate frozen labels/splits using metadata only")
    smoke = sub.add_parser("selfcheck", help="Synthetic CPU pipeline check, never real-data accuracy")
    smoke.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train", help="Future annotated development-only training; no fixed test/refit")
    train.add_argument("--annotations", type=Path)
    train.add_argument("--supervision", choices=("event_intervals", "file_labels"), default="event_intervals")
    train.add_argument("--file-label-approval", type=Path)
    train.add_argument("--fold", type=int, choices=range(4), required=True)
    train.add_argument("--run-name", required=True)
    train.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--domestic-weight", type=int, choices=(1, 2, 4), default=None)
    train.add_argument("--skip-export", action="store_true", help="CV float selection only; export the frozen final model later")
    train.add_argument("--wandb-mode", choices=WANDB_MODES, default="online",
                       help="online (default): send selected scalar metrics; offline: local only; disabled: no W&B")
    train.add_argument("--wandb-project", default="hearing_alert_system_CNN",
                       help="W&B project name (default: hearing_alert_system_CNN)")
    train.add_argument("--wandb-entity", default="jhh0415-korea-university",
                       help="W&B project owner (default: jhh0415-korea-university)")
    refit = sub.add_parser("refit", help="Fixed-epoch refit from a completed DS-CNN four-fold summary")
    refit.add_argument("--cv-summary", type=Path, required=True)
    refit.add_argument("--annotations", type=Path)
    refit.add_argument("--supervision", choices=("event_intervals", "file_labels"), default="event_intervals")
    refit.add_argument("--file-label-approval", type=Path)
    refit.add_argument("--run-name", required=True)
    refit.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    refit.add_argument("--wandb-mode", choices=WANDB_MODES, default="disabled")
    refit.add_argument("--wandb-project", default="hearing_alert_system_CNN")
    refit.add_argument("--wandb-entity", default="jhh0415-korea-university")
    args = parser.parse_args()
    if args.command == "audit":
        from .data import audit
        print(dump(audit()))
    elif args.command == "selfcheck":
        from .selfcheck import run
        run(args.output)
    elif args.command == "train":
        from .training import train_fold
        train_fold(args)
    elif args.command == "refit":
        from .training import train_refit
        train_refit(args)


if __name__ == "__main__":
    main()
