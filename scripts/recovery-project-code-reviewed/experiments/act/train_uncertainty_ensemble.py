"""Подготовка трёх обычных ACT для оценки неопределённости."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from act_recovery.ensemble.config import runtime_report
from act_recovery.ensemble.source import context
from act_recovery.ensemble.train import run_members
from act_recovery.settings import CODE_ROOT


def main():
    """Вход: без аргументов.
    Выход: выполнение выбранной команды терминала."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=("inspect", "smoke", "train"))
    parser.add_argument("--project-root", type=Path, default=CODE_ROOT)
    parser.add_argument("--initial-policy", type=Path)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    ctx = context(root, args.initial_policy)
    print(json.dumps(runtime_report(ctx), ensure_ascii=False, indent=2), flush=True)
    if args.mode != "inspect":
        run_members(ctx, smoke=args.mode == "smoke")
    else:
        print("INSPECT OK")


if __name__ == "__main__":
    main()
