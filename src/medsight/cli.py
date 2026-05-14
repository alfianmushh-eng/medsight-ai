"""medsight command-line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medsight import __version__, config, datasets


def _cmd_inspect(args: argparse.Namespace) -> int:
    ds = datasets.ImageFolderDataset(args.root)
    info = {
        "root": str(args.root),
        "classes": ds.classes,
        "total": len(ds),
        "class_counts": ds.class_counts(),
    }
    print(json.dumps(info, indent=2))
    return 0


def _cmd_validate_config(args: argparse.Namespace) -> int:
    cfg = config.load_config(args.path)
    print(f"OK: {cfg.name}")
    print(f"  data.train_root = {cfg.data.train_root}")
    print(f"  data.batch_size = {cfg.data.batch_size}")
    print(f"  model.name      = {cfg.model.name}")
    print(f"  train.epochs    = {cfg.train.epochs}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="medsight", description="Medical imaging research toolkit")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    insp = sub.add_parser("inspect", help="Inspect an ImageFolder-style dataset directory")
    insp.add_argument("root", type=Path)
    insp.set_defaults(func=_cmd_inspect)

    val = sub.add_parser("validate-config", help="Validate an experiment YAML")
    val.add_argument("path", type=Path)
    val.set_defaults(func=_cmd_validate_config)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
