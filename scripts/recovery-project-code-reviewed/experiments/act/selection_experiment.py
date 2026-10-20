"""Экспертные демонстрации, датасеты и дообучение выбранных политик."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from act_recovery.cli import positive_int
from act_recovery.expert.attempt import expert_one
from act_recovery.expert.collect import expert
from act_recovery.expert.prepare import prepare
from act_recovery.files import read_json, write_json
from act_recovery.settings import CODE_ROOT
from act_recovery.training.launch import train
from act_recovery.training.loop import train_one


def internal_worker(args):
    """Вход: args — аргументы команды терминала.
    Выход: сохранённый результат экспертной попытки или обучения."""
    request = read_json(args.request)
    os.chdir(request["project_root"])
    output = Path(request["output"])
    try:
        if args.command == "_expert-worker":
            result = {"status": "complete", "attempt": expert_one(request)}
        else:
            result = {"status": "complete", "model": train_one(request)}
        write_json(output / "worker_result.json", result)
    except BaseException as error:
        write_json(
            output / "worker_result.json",
            {"status": "error", "error": str(error), "traceback": traceback.format_exc()},
        )
        raise


def main():
    """Вход: без аргументов.
    Выход: выполнение выбранной команды терминала."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project-root", type=Path, default=CODE_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    collection = commands.add_parser(
        "expert", help="Expert continuation для selected candidates; одинаковый budget."
    )
    collection.add_argument("--plan", type=Path)
    collection.add_argument("--max-expert-steps", type=positive_int, default=280)
    preparation = commands.add_parser(
        "prepare", help="Отдельный dataset только выбранных demos для каждого метода."
    )
    preparation.add_argument("--collection", type=Path)
    training = commands.add_parser("train", help="Три копии исходной ACT, один общий training recipe.")
    training.add_argument("--datasets", type=Path)
    training.add_argument("--steps", type=positive_int, default=1000)
    training.add_argument("--seed", type=int, default=0)
    for name in ("_expert-worker", "_train-worker"):
        command = commands.add_parser(name, help=argparse.SUPPRESS)
        command.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    os.chdir(root)
    if args.command.startswith("_"):
        internal_worker(args)
    elif args.command == "expert":
        expert(args, root)
    elif args.command == "prepare":
        prepare(args, root)
    elif args.command == "train":
        train(args, root)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
