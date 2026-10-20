"""Сбор кандидатов, оценка расхождения ACT и план отбора."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from act_recovery.pool.collect import collect
from act_recovery.pool.worker import worker
from act_recovery.selection.score import score
from act_recovery.selection.select import select
from act_recovery.settings import CODE_ROOT


def main():
    """Вход: команда из терминала. Выход: выполнение collect, score или select."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project-root", type=Path, default=CODE_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    collection = commands.add_parser("collect", help="Собрать состояния после промаха и падения.")
    collection.add_argument("--init-ids", type=int, nargs="+", required=True)
    collection.add_argument("--ensemble", type=Path)
    scoring = commands.add_parser("score", help="Вычислить расхождение предсказаний трёх ACT.")
    scoring.add_argument("--pool", type=Path)
    scoring.add_argument("--ensemble", type=Path)
    selection = commands.add_parser("select", help="Сохранить планы отбора U, F_ref и Random.")
    selection.add_argument("--scores", type=Path)
    selection.add_argument("--budget", type=int, required=True)
    selection.add_argument("--seed", type=int, default=0)
    internal = commands.add_parser("_worker", help=argparse.SUPPRESS)
    internal.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    os.chdir(root)
    if args.command == "_worker":
        worker(args.request)
    elif args.command == "collect":
        collect(args, root)
    elif args.command == "score":
        score(args, root)
    elif args.command == "select":
        select(args, root)


if __name__ == "__main__":
    main()
