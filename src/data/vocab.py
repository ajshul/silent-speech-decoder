"""Vocabulary utilities for CTC-style character tokenization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class Vocab:
    tokens: List[str]
    token_to_id: Dict[str, int]
    pad_id: int
    blank_id: int
    unk_id: int

    @classmethod
    def from_json(cls, path: Path) -> "Vocab":
        data = json.loads(Path(path).read_text())
        tokens: List[str] = data["tokens"]
        token_to_id = {tok: i for i, tok in enumerate(tokens)}
        return cls(
            tokens=tokens,
            token_to_id=token_to_id,
            pad_id=int(data.get("pad_id", 0)),
            blank_id=int(data.get("blank_id", 1)),
            unk_id=int(data.get("unk_id", 2)),
        )

    def to_json(self, path: Path) -> None:
        payload = {
            "tokens": self.tokens,
            "pad_id": self.pad_id,
            "blank_id": self.blank_id,
            "unk_id": self.unk_id,
        }
        path.write_text(json.dumps(payload, indent=2))

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id.get(ch, self.unk_id) for ch in text.lower()]

    def decode(self, ids: Iterable[int], skip_blank: bool = True) -> str:
        chars: List[str] = []
        for i in ids:
            if i == self.blank_id and skip_blank:
                continue
            if i == self.pad_id:
                continue
            if 0 <= i < len(self.tokens):
                chars.append(self.tokens[i])
        return "".join(chars)

    @property
    def size(self) -> int:
        return len(self.tokens)
