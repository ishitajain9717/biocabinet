from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
@dataclass
class NodeResult:
    name:str
    ok: bool
    message:str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
