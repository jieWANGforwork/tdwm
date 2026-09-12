"""Auditable Eff optimizer/checkpoint primitives shared by full training.

No world-model module can enter this optimizer. The dataset and deployment
launchers must seal the protocol/cache/terminal provenance into ``identity``
before constructing the trainer. Resume rejects identity mismatches.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tdwm.methods.eff import EffModel, eff_loss, efficiency_weights
from tdwm.training.eff_data import EffReplayBatch

CHECKPOINT_FORMAT = "tdwm-eff-training-v1"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _batch_on_device(batch: EffReplayBatch, device: torch.device) -> dict:
    # dataclasses.asdict deep-copies tensors; avoid copying a full minibatch twice.
    return {
        name: getattr(batch, name).to(device) for name in batch.__dataclass_fields__
    }


def loss_inputs(
    batch: EffReplayBatch,
    *,
    device: torch.device,
    beta: float,
    gamma_g: float,
    critic_coefficient: float,
    include_goal_boundary: bool,
) -> dict:
    """Attach path weights and optional supervised zero-cost goal endpoints.

    When enabled, each endpoint is a second V sample of its SAME path. Its G
    loss is masked out; no false zero successor label is introduced.
    """
    data = _batch_on_device(batch, device)
    size = len(batch.state)
    weights = efficiency_weights(
        data["efficiency"], data["efficiency_known"], data["offset_chunks"], beta=beta
    )
    names = (
        "state",
        "next_state",
        "bootstrap_state",
        "goal",
        "terminal_after_transition",
        "observed_cost",
        "goal_reached",
        "continuation_valid",
        "vector_valid",
    )
    inputs = {key: data[key] for key in names}
    path_ids = torch.arange(size, device=device)
    if include_goal_boundary:
        for key in names:
            if key in ("state", "next_state", "bootstrap_state", "goal"):
                endpoint = data["goal"]
            elif key == "goal_reached":
                endpoint = torch.ones_like(data[key])
            else:
                endpoint = torch.zeros_like(data[key])
            inputs[key] = torch.cat((inputs[key], endpoint))
        path_ids = path_ids.repeat(2)
    return dict(
        inputs,
        path_ids=path_ids,
        path_weights=weights,
        gamma_g=gamma_g,
        critic_coefficient=critic_coefficient,
    )


class EffTrainer:
    """One optimizer step is one sampled path minibatch, plus boundary labels."""

    def __init__(
        self,
        model: EffModel,
        *,
        identity: dict,
        seed: int,
        learning_rate: float,
        weight_decay: float,
        gamma_g: float,
        beta: float,
        critic_coefficient: float,
        ema_rate: float,
        gradient_clip: float,
        include_goal_boundary: bool,
        device: str | torch.device,
    ) -> None:
        numeric = (
            learning_rate,
            weight_decay,
            gamma_g,
            beta,
            critic_coefficient,
            ema_rate,
            gradient_clip,
        )
        if not all(math.isfinite(x) for x in numeric):
            raise ValueError("Training hyperparameters must be finite.")
        if learning_rate <= 0 or weight_decay < 0 or gradient_clip <= 0:
            raise ValueError("Invalid optimizer settings.")
        if not 0 <= gamma_g <= 1 or beta < 0 or critic_coefficient < 0:
            raise ValueError("Invalid loss settings.")
        if not 0 < ema_rate <= 1:
            raise ValueError("EMA rate must be in (0, 1].")
        if not identity:
            raise ValueError("An explicit run identity is required.")
        self.device = torch.device(device)
        self.model = model.to(self.device).train()
        self.identity = copy.deepcopy(identity)
        self.settings = dict(
            seed=seed,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gamma_g=gamma_g,
            beta=beta,
            critic_coefficient=critic_coefficient,
            ema_rate=ema_rate,
            gradient_clip=gradient_clip,
            include_goal_boundary=include_goal_boundary,
            g_hidden_dim=model.g.hidden_dim,
            v_hidden_dim=model.v.hidden_dim,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.online_parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.rng = np.random.default_rng(seed)
        self.global_step = 0

    def step(
        self, batch: EffReplayBatch, *, learning_rate: float | None = None
    ) -> dict:
        if learning_rate is not None:
            if not math.isfinite(learning_rate) or learning_rate < 0:
                raise ValueError("Scheduled learning rate must be finite >= 0.")
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
        self.optimizer.zero_grad(set_to_none=True)
        inputs = loss_inputs(
            batch,
            device=self.device,
            beta=self.settings["beta"],
            gamma_g=self.settings["gamma_g"],
            critic_coefficient=self.settings["critic_coefficient"],
            include_goal_boundary=self.settings["include_goal_boundary"],
        )
        loss = eff_loss(self.model, **inputs)
        if not bool(torch.isfinite(loss.total)):
            raise FloatingPointError("Nonfinite Eff loss; optimizer step not applied.")
        loss.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.online_parameters(),
            self.settings["gradient_clip"],
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.model.update_targets(rate=self.settings["ema_rate"])
        self.global_step += 1
        return {
            "global_step": self.global_step,
            "total_loss": loss.total.item(),
            "g_vector_loss": loss.vector.item(),
            "v_movement_loss": loss.critic.item(),
            "gradient_norm": gradient_norm.item(),
            "v_valid_samples": int(loss.critic_valid.sum()),
            "known_efficiency_paths": int(batch.efficiency_known.sum()),
            "direct_samples": int(batch.goal_reached.sum()),
            "td_samples": int((~batch.goal_reached & batch.continuation_valid).sum()),
            "failed_terminal_samples": int(
                (~batch.goal_reached & ~batch.continuation_valid).sum()
            ),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def validate(self, batch: EffReplayBatch) -> dict:
        """No updates, no sampler draws, no EMA changes, no train RNG mutation."""
        previous_mode = self.model.training
        self.model.eval()
        try:
            inputs = loss_inputs(
                batch,
                device=self.device,
                beta=self.settings["beta"],
                gamma_g=self.settings["gamma_g"],
                critic_coefficient=self.settings["critic_coefficient"],
                include_goal_boundary=self.settings["include_goal_boundary"],
            )
            losses = eff_loss(self.model, **inputs)
            inputs["path_weights"] = torch.ones_like(inputs["path_weights"])
            unweighted = eff_loss(self.model, **inputs)
        finally:
            self.model.train(previous_mode)
        return {
            "total_loss": losses.total.item(),
            "g_vector_loss": losses.vector.item(),
            "v_movement_loss": losses.critic.item(),
            "unweighted_total_loss": unweighted.total.item(),
            "unweighted_g_vector_loss": unweighted.vector.item(),
            "unweighted_v_movement_loss": unweighted.critic.item(),
        }

    def save(self, path: str | Path, *, epoch: int) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "method": "Eff",
            "epoch": epoch,
            "global_step": self.global_step,
            "identity": self.identity,
            "identity_sha256": canonical_sha256(self.identity),
            "settings": self.settings,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "numpy_rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if self.device.type == "cuda"
            else None,
        }
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".eff-", suffix=".pt", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return hashlib.sha256(destination.read_bytes()).hexdigest()

    def resume(self, path: str | Path) -> int:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Not an Eff training checkpoint.")
        if payload.get("identity_sha256") != canonical_sha256(payload.get("identity")):
            raise ValueError("Checkpoint identity hash is invalid.")
        if payload["identity"] != self.identity or payload["settings"] != self.settings:
            raise ValueError("Resume protocol, artifact identity or settings differ.")
        if self.device.type == "cuda" and payload["cuda_rng"] is None:
            raise ValueError("CUDA resume requires saved CUDA RNG state.")
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.rng.bit_generator.state = payload["numpy_rng"]
        torch.set_rng_state(payload["torch_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        self.global_step = int(payload["global_step"])
        return int(payload["epoch"])


def load_eff_model(
    path: str | Path,
    *,
    expected_identity: dict,
    expected_global_step: int,
    device: str | torch.device,
) -> tuple[EffModel, dict]:
    """Restore a frozen deployment model with explicit provenance/update gates."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("method") != "Eff":
        raise ValueError("Not an Eff checkpoint.")
    if payload.get("identity") != expected_identity or payload.get(
        "identity_sha256"
    ) != canonical_sha256(expected_identity):
        raise ValueError("Eff checkpoint identity differs from evaluation protocol.")
    if payload.get("global_step") != expected_global_step:
        raise ValueError("Eff checkpoint optimizer updates differ from protocol.")
    settings = payload["settings"]
    model = EffModel(
        g_hidden_dim=settings["g_hidden_dim"], v_hidden_dim=settings["v_hidden_dim"]
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, payload
