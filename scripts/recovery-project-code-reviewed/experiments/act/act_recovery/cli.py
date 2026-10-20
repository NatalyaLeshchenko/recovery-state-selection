"""Общие типы аргументов командной строки."""

from __future__ import annotations

import argparse


def positive_int(value):
    
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Нужно положительное целое число.")
    return value
