from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import re
from typing import Any, Sequence

import numpy as np


WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").split())


def word_tokens(value: str) -> tuple[str, ...]:
    return tuple(WORD_RE.findall(value.casefold()))


def word_f1(left: str, right: str) -> float:
    left_counts = Counter(word_tokens(left))
    right_counts = Counter(word_tokens(right))
    if not left_counts or not right_counts:
        return 0.0
    overlap = sum((left_counts & right_counts).values())
    return 2.0 * overlap / (sum(left_counts.values()) + sum(right_counts.values()))


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embedding matrix must have two dimensions")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def cosine_matrix(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    left_values = l2_normalize(left)
    right_values = left_values if right is None else l2_normalize(right)
    if left_values.shape[1] != right_values.shape[1]:
        raise ValueError("embedding dimensions do not match")
    return left_values @ right_values.T


def stable_id(prefix: str, *values: object) -> str:
    raw = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return prefix + sha256(raw.encode("utf-8")).hexdigest()


def extract_json_array(text: str) -> tuple[list[Any] | None, bool]:
    """Parse an exact JSON array or the first top-level array in surrounding text."""

    try:
        exact = json.loads(text)
    except json.JSONDecodeError:
        exact = None
    else:
        return (exact, False) if isinstance(exact, list) else (None, False)

    decoder = json.JSONDecoder()
    in_string = False
    escaped = False
    depth = 0
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character in "{[":
            if character == "[" and depth == 0:
                try:
                    value, _ = decoder.raw_decode(text, index)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(value, list):
                        return value, True
            depth += 1
        elif character in "}]" and depth:
            depth -= 1
    return None, False


def batched(values: Sequence[Any], size: int):
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]
