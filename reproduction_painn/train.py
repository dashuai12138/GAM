"""Command-line entry point for formal PaiNN ablation training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reproduction_painn.training import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--steps", type=int, help="override steps for a smoke/benchmark run"
    )
    parser.add_argument(
        "--epochs", type=int, help="override the configured epoch target"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="checkpoint path; omit the value to use OUTPUT_DIR/latest.pt",
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config["output_dir"])
    resume = output_dir / "latest.pt" if args.resume == "latest" else args.resume
    train_from_config(
        config,
        device_name=args.device,
        steps_override=args.steps,
        epochs_override=args.epochs,
        output_dir_override=output_dir,
        resume=resume,
    )


if __name__ == "__main__":
    main()
