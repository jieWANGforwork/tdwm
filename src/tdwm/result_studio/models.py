"""Shared, UI-independent result contract. No inferred success labels."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Trajectory:
    name: str
    kind: str  # reference / executed / predicted
    success: bool | None = None
    states: np.ndarray | None = None
    actions: np.ndarray | None = None
    embeddings: np.ndarray | None = None
    frames: list[str] = field(default_factory=list)
    video: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        if self.frames:
            return len(self.frames)
        return len(self.states) if self.states is not None else 0


@dataclass
class Trial:
    index: int
    episode: int
    offset: int
    start: int
    goal: int
    reference: Trajectory
    methods: dict[str, dict[str, Trajectory]]
    source: str = ""
    demo: bool = False
