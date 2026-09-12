"""Two-stage state-planner training using frozen Eff and public SWM CEM.

Stage 1 learns real midpoint structure. Stage 2 alternates CEM matching with
learned state updates and differentiates the resulting trajectory/efficiency/
dynamics losses into P only. Candidate-state gradients remain enabled even
though the G/V/F parameters and each action-search result are frozen.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tdwm.adapters.effplan import EffPlanTrackingCost
from tdwm.methods.eff import EffModel
from tdwm.methods.effplan import (
    StatePlanner,
    generate_state_path,
    planner_loss,
    refine_state_path,
)
from tdwm.training.eff_data import EffPlanReplayBatch
from tdwm.training.eff_runtime import canonical_sha256

CHECKPOINT_FORMAT = "tdwm-effplan-training-v1"


@dataclass(frozen=True)
class EffPlanTrainSettings:
    phase: str
    seed: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    epsilon: float
    target_readout: bool
    trajectory_coefficient: float
    efficiency_coefficient: float
    dynamics_coefficient: float
    search_iterations: tuple[int, ...]
    cem_candidates: int
    cem_elites: int
    cem_batch_size: int
    supervision: str

    def __post_init__(self) -> None:
        if self.phase not in {"generation", "refinement"}:
            raise ValueError("Unknown EffPlan training phase.")
        if self.supervision not in {"final", "mean_rounds"}:
            raise ValueError(
                "Planner supervision must be explicitly final or mean_rounds."
            )
        values = (
            self.learning_rate,
            self.weight_decay,
            self.gradient_clip,
            self.epsilon,
            self.trajectory_coefficient,
            self.efficiency_coefficient,
            self.dynamics_coefficient,
        )
        if any(not math.isfinite(x) for x in values):
            raise ValueError("Planner settings must be finite.")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.epsilon <= 0:
            raise ValueError(
                "Learning rate, gradient clip and epsilon must be positive."
            )
        if any(
            x < 0
            for x in (
                self.weight_decay,
                self.trajectory_coefficient,
                self.efficiency_coefficient,
                self.dynamics_coefficient,
            )
        ):
            raise ValueError(
                "Planner loss coefficients and weight decay must be nonnegative."
            )
        if self.phase == "generation":
            if (
                self.efficiency_coefficient
                or self.dynamics_coefficient
                or self.search_iterations
            ):
                raise ValueError(
                    "Generation phase has trajectory loss only and no CEM."
                )
            if self.trajectory_coefficient <= 0:
                raise ValueError("Generation phase must supervise real state nodes.")
        elif not self.search_iterations or any(x < 1 for x in self.search_iterations):
            raise ValueError("Refinement requires explicit positive CEM rounds.")
        if self.cem_candidates < 2 or not 2 <= self.cem_elites <= self.cem_candidates:
            raise ValueError("CEM needs >=2 elites for a defined sample variance.")
        if self.cem_batch_size < 1:
            raise ValueError("CEM batch size must be positive.")


def latent_planning_context(start: torch.Tensor, goal: torch.Tensor) -> dict:
    """SWM cached-latent rollout; dummy pixels convey history shape only.

    Both observation and goal embeddings are already provided. No generated
    pixels or real-image decoding is involved; tests assert that F receives
    the cached real anchor continuously throughout its five-block rollout.
    """
    if start.shape != goal.shape or start.ndim != 2 or start.shape[-1] != 192:
        raise ValueError("Expected matching [batch, 192] start/goal states.")
    return {
        "pixels": start.new_zeros(len(start), 1, 3, 1, 1),
        "emb": start.detach()[:, None],
        "goal_emb": goal.detach()[:, None],
    }


class EffPlanTrainer:
    def __init__(
        self,
        *,
        planner: StatePlanner,
        eff: EffModel,
        settings: EffPlanTrainSettings,
        identity: dict,
        device: str,
        tracking_model: EffPlanTrackingCost | None = None,
    ) -> None:
        if not identity:
            raise ValueError("Planner training requires explicit source identity.")
        if any(p.requires_grad for p in eff.parameters()):
            raise ValueError("G/V must be frozen before training P.")
        if not all(p.requires_grad for p in planner.parameters()):
            raise ValueError("All online P parameters must be trainable.")
        if settings.phase == "refinement" and tracking_model is None:
            raise ValueError(
                "Refinement requires frozen F and the CEM tracking adapter."
            )
        if tracking_model is not None:
            if tracking_model.target_critic.eff is not eff:
                raise ValueError(
                    "CEM and planner feedback must share the same frozen Eff."
                )
            if tracking_model.target_critic.target != settings.target_readout:
                raise ValueError("Planner and CEM readout selection differs.")
            if any(p.requires_grad for p in tracking_model.parameters()):
                raise ValueError("All G/V/F parameters must remain frozen.")
        self.settings = settings
        self.identity = copy.deepcopy(identity)
        self.device = torch.device(device)
        self.planner = planner.to(device).train()
        self.eff = eff.to(device).eval()
        self.tracking_model = tracking_model
        self.optimizer = torch.optim.AdamW(
            self.planner.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        self.global_step = 0
        self.rng = np.random.default_rng(settings.seed)
        self.solver = None
        if settings.phase == "refinement":
            import stable_worldmodel as swm

            self.tracking_model.to(device).eval()
            self.solver = swm.solver.CEMSolver(
                model=self.tracking_model,
                batch_size=settings.cem_batch_size,
                num_samples=settings.cem_candidates,
                topk=settings.cem_elites,
                n_steps=settings.search_iterations[0],
                var_scale=1.0,
                device=device,
                seed=settings.seed,
            )

    def value(self, left, right):
        return self.eff.value(left, right, target=self.settings.target_readout)

    def _loss(self, batch: EffPlanReplayBatch):
        import stable_worldmodel as swm
        from gymnasium.spaces import Box

        states = batch.real_states.to(self.device).detach()
        valid = batch.trajectory_valid.to(self.device)
        if states.ndim != 3 or states.shape[1:] != (6, 192) or batch.state_stride != 5:
            raise ValueError(
                "Planner training needs six nodes at primitive stride five."
            )
        if self.settings.phase == "generation" and not bool(valid.all()):
            raise ValueError(
                "Generation phase requires real connected midpoint labels."
            )
        nodes = generate_state_path(
            self.planner,
            states[:, 0],
            states[:, -1],
            self.value,
            horizon=5,
            epsilon=self.settings.epsilon,
        )
        loss_args = dict(
            real_states=states,
            value=self.value,
            epsilon=self.settings.epsilon,
            trajectory_coefficient=self.settings.trajectory_coefficient,
            efficiency_coefficient=self.settings.efficiency_coefficient,
            dynamics_coefficient=self.settings.dynamics_coefficient,
            trajectory_valid=valid,
        )
        losses = []
        if self.solver is None:
            losses.append(planner_loss(nodes, predicted_future=None, **loss_args))
        else:
            self.solver.configure(
                action_space=Box(-1.0, 1.0, shape=(len(states), 5), dtype=np.float32),
                n_envs=len(states),
                config=swm.PlanConfig(
                    horizon=5,
                    receding_horizon=5,
                    history_len=1,
                    action_block=5,
                ),
            )
            info = latent_planning_context(states[:, 0], states[:, -1])
            actions = None
            for iterations in self.settings.search_iterations:
                self.solver.n_steps = iterations
                search_info = dict(info, effplan_nodes=nodes.detach())
                result = self.solver.solve(search_info, init_action=actions)
                # Clone outside SWM's inference context before gradient-based
                # state feedback; the action search itself is not differentiated.
                actions = result["actions"].to(self.device).clone().detach()
                expanded = {key: val[:, None] for key, val in info.items()}
                with torch.no_grad():
                    future = self.tracking_model.future_states(
                        expanded, actions[:, None]
                    )[:, 0]
                nodes = refine_state_path(
                    self.planner,
                    nodes,
                    self.value,
                    predicted_future=future,
                    epsilon=self.settings.epsilon,
                    dynamics_coefficient=self.settings.dynamics_coefficient,
                )
                losses.append(planner_loss(nodes, predicted_future=future, **loss_args))
        selected = losses if self.settings.supervision == "mean_rounds" else losses[-1:]
        loss = torch.stack([x.total for x in selected]).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite planner loss; optimizer not updated.")
        return loss, selected, valid

    def step(self, batch: EffPlanReplayBatch) -> dict:
        self.optimizer.zero_grad(set_to_none=True)
        loss, selected, valid = self._loss(batch)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.planner.parameters(),
            self.settings.gradient_clip,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.global_step += 1
        return {
            "global_step": self.global_step,
            "phase": self.settings.phase,
            "total_loss": loss.item(),
            "gradient_norm": norm.item(),
            "trajectory_loss": torch.stack([x.trajectory for x in selected])
            .mean()
            .item(),
            "efficiency_loss": torch.stack([x.efficiency for x in selected])
            .mean()
            .item(),
            "dynamics_loss": torch.stack([x.dynamics for x in selected]).mean().item(),
            "trajectory_labelled_paths": int(valid.sum()),
            "cem_candidate_rollouts": len(valid)
            * self.settings.cem_candidates
            * sum(self.settings.search_iterations),
            "returned_action_rerolls": len(valid)
            * len(self.settings.search_iterations),
        }

    @torch.no_grad()
    def validate(self, batch: EffPlanReplayBatch) -> dict:
        mode = self.planner.training
        self.planner.eval()
        cem_rng = None if self.solver is None else self.solver.torch_gen.get_state()
        try:
            loss, selected, valid = self._loss(batch)
            return {
                "total_loss": loss.item(),
                "trajectory_loss": torch.stack([x.trajectory for x in selected])
                .mean()
                .item(),
                "efficiency_loss": torch.stack([x.efficiency for x in selected])
                .mean()
                .item(),
                "dynamics_loss": torch.stack([x.dynamics for x in selected])
                .mean()
                .item(),
                "trajectory_labelled_paths": int(valid.sum()),
            }
        finally:
            self.planner.train(mode)
            if cem_rng is not None:
                self.solver.torch_gen.set_state(cem_rng)

    def save(self, path: str | Path, *, epoch: int) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "method": "EffPlan",
            "settings": dataclasses.asdict(self.settings),
            "identity": self.identity,
            "identity_sha256": canonical_sha256(self.identity),
            "planner_hidden_dim": self.planner.hidden_dim,
            "epoch": epoch,
            "global_step": self.global_step,
            "planner": self.planner.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "numpy_rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cem_rng": None
            if self.solver is None
            else self.solver.torch_gen.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if self.device.type == "cuda"
            else None,
        }
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".effplan-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def resume(self, path: str | Path) -> int:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("format") != CHECKPOINT_FORMAT
            or payload.get("method") != "EffPlan"
        ):
            raise ValueError("Not an EffPlan training checkpoint.")
        if (
            payload.get("identity") != self.identity
            or payload.get("identity_sha256") != canonical_sha256(self.identity)
            or payload.get("settings") != dataclasses.asdict(self.settings)
            or payload.get("planner_hidden_dim") != self.planner.hidden_dim
        ):
            raise ValueError("EffPlan resume identity, phase or settings differ.")
        if (self.solver is None) != (payload.get("cem_rng") is None):
            raise ValueError("EffPlan resume CEM RNG is incompatible with the phase.")
        if self.device.type == "cuda" and payload.get("cuda_rng") is None:
            raise ValueError("CUDA resume requires CUDA RNG state.")
        self.planner.load_state_dict(payload["planner"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.rng.bit_generator.state = payload["numpy_rng"]
        torch.set_rng_state(payload["torch_rng"])
        if self.solver is not None:
            self.solver.torch_gen.set_state(payload["cem_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        self.global_step = int(payload["global_step"])
        return int(payload["epoch"])


def load_effplan_planner(
    path: str | Path,
    *,
    expected_identity: dict,
    expected_global_step: int,
    expected_phase: str,
    device: str,
) -> tuple[StatePlanner, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("method") != "EffPlan":
        raise ValueError("Not an EffPlan checkpoint.")
    if (
        payload.get("identity") != expected_identity
        or payload.get("identity_sha256") != canonical_sha256(expected_identity)
        or payload.get("global_step") != expected_global_step
        or payload["settings"]["phase"] != expected_phase
    ):
        raise ValueError("EffPlan deployment identity, updates or phase differs.")
    planner = StatePlanner(hidden_dim=payload["planner_hidden_dim"])
    planner.load_state_dict(payload["planner"], strict=True)
    return planner.to(device).eval().requires_grad_(False), payload
