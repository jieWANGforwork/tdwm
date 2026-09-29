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
    method_specs: dict[str, "MethodSpec"] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodSpec:
    """A trained model variant and an independently chosen inference/search rule."""

    training_method: str
    search_method: str

    @property
    def label(self) -> str:
        if self.search_method == "未指定":
            return self.training_method
        return f"{self.training_method} · {self.search_method}"


def method_catalog(trials: list[Trial]) -> dict[str, MethodSpec]:
    catalog = {}
    for trial in trials:
        for key in trial.methods:
            spec = trial.method_specs.get(key, MethodSpec(key, "未指定"))
            if key in catalog and catalog[key] != spec:
                raise ValueError(f"同一运行编号的训练/搜索方法不一致：{key}")
            catalog[key] = spec
    return catalog


def select_runs(catalog, training_methods, search_methods) -> list[str]:
    return [
        key
        for key, spec in catalog.items()
        if spec.training_method in training_methods
        and spec.search_method in search_methods
    ]
