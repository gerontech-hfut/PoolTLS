from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .text import l2_normalize


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class LocalTextEncoder:
    """Frozen encoder covering complete texts with overlapping token windows.

    Each window uses the pretrained encoder's attention-mask mean pooling.
    ``encode_chunks`` retains window vectors for passage retrieval; ``encode``
    returns one normalized vector per text for existing short-text consumers.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda:0",
        batch_size: int = 32,
        max_length: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if type(chunk_overlap) is not int or chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.requested_max_length = max_length
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, use_fast=True
        )
        self.model = AutoModel.from_pretrained(
            str(model_path), local_files_only=True
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        # Some tokenizers use a huge sentinel for an unspecified limit.  The
        # model's positional embeddings, not that sentinel, bound each window.
        limits = [
            limit
            for limit in (
                getattr(self.model.config, "max_position_embeddings", None),
                getattr(self.tokenizer, "model_max_length", None),
            )
            if type(limit) is int and 0 < limit < 10_000_000
        ]
        if not limits:
            raise ValueError("cannot determine the encoder's maximum sequence length")
        self.model_max_length = min(limits)
        self.max_length = min(max_length, self.model_max_length)
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        self.content_length = self.max_length - special_tokens
        if self.content_length <= 0:
            raise ValueError("max_length leaves no room for text tokens")
        if chunk_overlap >= self.content_length:
            raise ValueError("chunk_overlap must be smaller than the window's text capacity")
        self._cache: dict[str, np.ndarray] = {}
        self._chunk_cache: dict[str, np.ndarray] = {}

    def _encode_missing(self, texts: Sequence[str]) -> None:
        torch = self._torch
        collected: dict[str, list[np.ndarray]] = {text: [] for text in texts}
        windows = []
        owners: list[str] = []

        def encode_batch() -> None:
            # Only this bounded batch of windows is materialized on the GPU.
            encoded = self.tokenizer.pad(windows, padding=True, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            vectors = l2_normalize(pooled.float().cpu().numpy())
            for text, vector in zip(owners, vectors, strict=True):
                collected[text].append(vector)
            windows.clear()
            owners.clear()

        step = self.content_length - self.chunk_overlap
        for text in texts:
            token_ids = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            # Empty strings still have one special-token-only window, matching
            # the original mean-pooling behavior rather than producing NaNs.
            for start in range(0, max(1, len(token_ids)), step):
                stop = min(start + self.content_length, len(token_ids))
                windows.append(self.tokenizer.prepare_for_model(
                    token_ids[start:stop],
                    add_special_tokens=True,
                    padding=False,
                    truncation=False,
                    return_attention_mask=True,
                ))
                owners.append(text)
                if len(windows) == self.batch_size:
                    encode_batch()
                if stop == len(token_ids):
                    break
        if windows:
            encode_batch()
        for text, vectors in collected.items():
            chunks = np.stack(vectors).astype(np.float32)
            self._chunk_cache[text] = chunks
            self._cache[text] = (
                chunks[0]
                if len(chunks) == 1
                else l2_normalize(chunks.mean(axis=0, keepdims=True))[0]
            )

    def encode_chunks(self, texts: Sequence[str]) -> tuple[np.ndarray, ...]:
        """Return normalized ``(window_count, hidden_size)`` vectors per text."""
        values = tuple(str(text) for text in texts)
        missing = tuple(dict.fromkeys(text for text in values if text not in self._chunk_cache))
        if missing:
            self._encode_missing(missing)
        return tuple(self._chunk_cache[text] for text in values)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = tuple(str(text) for text in texts)
        if not values:
            hidden_size = int(getattr(self.model.config, "hidden_size", 0))
            return np.empty((0, hidden_size), dtype=np.float32)
        self.encode_chunks(values)
        return np.stack([self._cache[text] for text in values]).astype(np.float32)
