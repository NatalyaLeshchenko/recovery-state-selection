"""Чтение JSON, хеши файлов и пути к результатам."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


class StageNotReady(RuntimeError):
    pass


def read_json(path):

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Вход: путь и данные. Выход: записанный JSON"""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    """Вход: путь к файлу. Выход: строка SHA-256 для проверки."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        block = stream.read(1024 * 1024)
        while block:
            digest.update(block)
            block = stream.read(1024 * 1024)
    return digest.hexdigest()


def local_path(root, path):
    
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def resolve_pointer(root, filename):
    
    pointer = root / "pilot_runs" / filename
    return local_path(root, pointer.read_text(encoding="utf-8").strip())


def write_pointer(root, filename, path):
    
    target = root / "pilot_runs" / filename
    temporary = target.with_suffix(".tmp")
    temporary.write_text(str(path) + "\n", encoding="utf-8")
    temporary.replace(target)


def new_folder(root, prefix):
    
    path = root / "pilot_runs/selection" / datetime.now().strftime(prefix + "_%Y%m%d_%H%M%S_%f")
    path.mkdir(parents=True, exist_ok=False)
    return path
