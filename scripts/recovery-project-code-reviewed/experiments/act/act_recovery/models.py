"""Сведения о файлах ACT, версии библиотек и проверка неизменности весов."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

from act_recovery.files import file_hash
from act_recovery.settings import POLICY_IGNORE, REQUIRED


def policy_settings(config):

    settings = {}
    for key, value in config.items():
        if key not in POLICY_IGNORE:
            settings[key] = value
    return settings


def identity(path):
    
    names = set(REQUIRED) #названия нужных файлов
    for pattern in ("policy_preprocessor*", "policy_postprocessor*"):
        names.update((p.name for p in path.glob(pattern) if p.is_file()))
    for name in names:
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    hashes = {}
    for name in sorted(names):
        hashes[name] = file_hash(path / name)
    return {"path": str(path), "sha256": hashes}


def verify_identity(record):
    
    path = Path(record["path"])
    if any((file_hash(path / name) != expected for name, expected in record["sha256"].items())):
        raise RuntimeError("Файлы исходной модели изменились во время подготовки ensemble.")


def array_hash(value):
   
    value = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(value.tobytes()).hexdigest()


def model_identity(path):

    path = path.resolve()
    required = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
    for name in required:
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    config = json.loads((path / "config.json").read_text())
    if config.get("type") != "act":
        raise RuntimeError(f"Ожидается ACT, получено {config.get('type')}: {path}")
    names = set(required)
    for pattern in ("policy_preprocessor*", "policy_postprocessor*"):
        names.update((p.name for p in path.glob(pattern) if p.is_file()))
    hashes = {}
    for name in sorted(names):
        hashes[name] = file_hash(path / name)
    return {"path": str(path), "sha256": hashes}


def verify_model(identity):
   
    path = Path(identity["path"])
    for name, expected in identity["sha256"].items():
        if file_hash(path / name) != expected:
            raise RuntimeError(f"Модель или processors изменились после фиксации пула: {path / name}")


def runtime_versions():
    
    result = {"python": sys.version.split()[0]}
    for package in ("lerobot", "hf-libero", "gymnasium", "torch", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result
