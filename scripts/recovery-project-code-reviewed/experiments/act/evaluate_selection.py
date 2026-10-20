"""Независимая проверка recovery и обычной задачи."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from act_recovery.evaluation.clean import clean
from act_recovery.evaluation.clean_worker import clean_worker
from act_recovery.evaluation.collect import collect
from act_recovery.evaluation.recovery import recovery
from act_recovery.evaluation.worker import recovery_worker
from act_recovery.files import StageNotReady, read_json, write_json
from act_recovery.settings import CODE_ROOT


def main():
    """Вход: без аргументов.
    Выход: выполнение выбранной команды терминала."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project-root", type=Path, default=CODE_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    collection = commands.add_parser("collect", help="Новые TEST starts; без expert и uncertainty scoring.")
    collection.add_argument("--init-ids", type=int, nargs="+", default=[15, 45])
    collection.add_argument("--models", type=Path)
    for name in ("recovery", "clean"):
        command = commands.add_parser(name)
        command.add_argument("--pool", type=Path)
        command.add_argument("--resume", type=Path)
    worker = commands.add_parser("_clean-worker", help=argparse.SUPPRESS)
    worker.add_argument("--request", type=Path, required=True)
    recovery_process = commands.add_parser("_recovery-worker", help=argparse.SUPPRESS)
    recovery_process.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    try:
        if args.command == "_recovery-worker":
            recovery_worker(args.request)
        elif args.command == "_clean-worker":
            request = read_json(args.request)
            try:
                result = clean_worker(request)
            except BaseException as error:
                write_json(
                    Path(request["output"]) / "worker_result.json",
                    {"status": "error", "error": str(error), "traceback": traceback.format_exc()},
                )
                raise
            write_json(Path(request["output"]) / "worker_result.json", result)
        elif args.command == "collect":
            collect(args, root)
        elif args.command == "recovery":
            recovery(args, root)
        elif args.command == "clean":
            clean(args, root)
    except (StageNotReady, FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"\nERROR: {error}", file=sys.stderr, flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
