"""Deployment support for Actor-Free TD-LeWM V2 coupled-Hybrid methods.

V2 checkpoints contain jointly fine-tuned online/EMA LeWM models and the
online/EMA copy of the single V1-shaped TD-JEPA predictor.  Deployment uses
the *online* world model and predictor only.  The target copies are restored
strictly to audit checkpoint completeness, then discarded from the planner.

The planner intentionally reuses V1's three score topologies.  This does not
make a V2 artifact a V1 artifact: family, method, variant, implementation,
coupled-training metadata, and V1 initialization provenance are all validated
against the explicit V2 contract before either online module is returned.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tdwm.adapters.frozen_actor_free_td_v1_common import (
    ACTION_BLOCK_STEPS,
    LEWM_HISTORY_SIZE,
    SCORE_MODES,
    ActorFreeTDLeWMV1,
    _resolve_g_first_weight,
    require_exact_values,
    require_positive_float,
)
from tdwm.methods.actor_free_td_lewm_v1 import (
    ActorFreeTDJEPAPredictorV1,
    project_tasks_to_sphere_v1,
)
from tdwm.methods.actor_free_td_lewm_v2 import (
    V2_ACTION_DIM,
    V2_ACTION_EMBEDDING_DIM,
    V2_OUTPUT_DIM,
    V2_RAW_ACTION_DIM,
    V2_STATE_DIM,
    V2_TASK_DIM,
    validate_lewm_action_encoder_v2,
)

METHOD_FAMILY = "actor_free_td_lewm_v2"
IMPLEMENTATION_VERSION = "v2"
OBJECTIVE_VERSION = 0
DEPLOYMENT_CHECKPOINT_VERSION = 1
V2_SCORE_MODES = SCORE_MODES | frozenset({"g_only_f_rollout_mean"})
SOURCE_V1_FAMILY = "actor_free_td_lewm_v1"
SOURCE_V1_IMPLEMENTATION_VERSION = "v1"
SOURCE_V1_CODE_REVISION = "3c4e62ef2ab72387536433f27ef11bce75477e7e"
SOURCE_V1_EPOCH = 10
SOURCE_V1_GLOBAL_STEP = 127_960
SOURCE_V1_SHA256 = {
    "c": "88bd65c48a6c701852f50552ec8f9109d6ae8ac57c467de207aa2c652c0f59a3",
    "d": "3115fffeb83ba6ae7e0c272913fe7a1ba16d42953b2185f6a3f7b168899d819a",
    "f": "b4de1b511075d763194ad1e332d127cbe390553738162f3a402ef8847bb74fd0",
    "g1": "c224d18fcd8390247f115239c4b2db013479a062438cca92003674c739f3e24b",
    "g2": "1c290f91772b42fdf6824d92832c6fff4e2d8ca3ea08089ff1a41016ea1c2ebe",
    "g3": "b279a85b1dd0816bd5fb9724da490810d470755880639297aa13699c86c2d8fb",
}
SOURCE_ARTIFACTS = {
    "split_file_sha256": (
        "4594afb3603b4258431ff9076c82acbe3ddcaccb277940b825a99017ce83d830"
    ),
    "train_indices_sha256": (
        "a1665554b6f5dc1c4aa37768cd7008fdc96f6a55ec5e8e12d9a93afa99880561"
    ),
    "validation_indices_sha256": (
        "e5aed8baa556f3f868ed471c511488df2117332837303ba958df278b34a61a6c"
    ),
    "column_normalization_sha256": (
        "7fd14e6a72841a36abd8f1d4aedf4f17f4f71ca508cacefe331e989664954818"
    ),
    "frozen_latent_store_manifest_sha256": (
        "fc80bcc4187a7fd98ff7bbfcfa1d5a4c3a76b467af2f5f22fed601855c573c7e"
    ),
    "g1_neighbor_index_manifest_sha256": (
        "3b2d785790d86c4c45bc10f1cf706f9fc186a02071fb4f8b586eca75a2af76f2"
    ),
}


@dataclass(frozen=True)
class ActorFreeTDV2MethodSpec:
    """Deployment identity and objective validation for one V2-family method."""

    method: str
    variant: str
    display_name: str
    objective_keys: tuple[str, ...]
    validate_method_config: Callable[[Mapping[str, Any]], None]
    method_family: str = METHOD_FAMILY
    implementation_version: str = IMPLEMENTATION_VERSION
    evaluation_stage: str = "planner_evaluation"
    initialization: str = "corresponding_v1_deployment_finetune"
    local_prediction: str = "original_lewm_one_step_mse"
    local_prediction_target: str | None = None
    local_prediction_target_gradient: str | None = None
    inference_g_score: str = "negative_goal_projection_of_v2_online_predictor"
    deployed_world_model: str = "online_v2_world_model"
    deployed_predictor: str = "online_v2_predictor"
    training_stage: str | None = None
    initialization_contract: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.variant not in SOURCE_V1_SHA256:
            raise ValueError(f"Unsupported V2 variant {self.variant!r}.")
        if self.method != f"{self.method_family}_{self.variant}":
            raise ValueError("V2 method names must end in their exact variant.")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.method_family,
                self.implementation_version,
                self.evaluation_stage,
                self.initialization,
                self.local_prediction,
                self.inference_g_score,
                self.deployed_world_model,
                self.deployed_predictor,
            )
        ):
            raise ValueError("V2 deployment identity fields must be non-empty.")
        optional_target = (
            self.local_prediction_target,
            self.local_prediction_target_gradient,
        )
        if (optional_target[0] is None) != (optional_target[1] is None):
            raise ValueError(
                "V2 local prediction target and gradient must be specified together."
            )
        if any(value is not None and not value for value in optional_target):
            raise ValueError("V2 local prediction target fields must be non-empty.")
        if self.training_stage is not None and not self.training_stage:
            raise ValueError("V2 training_stage must be non-empty when specified.")
        if self.initialization_contract is not None and not isinstance(
            self.initialization_contract, Mapping
        ):
            raise ValueError("V2 initialization_contract must be a mapping.")


def _positive_integer(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool):
        raise ValueError(f"predictor_config.{key} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"predictor_config.{key} must be a positive integer."
        ) from error
    if integer <= 0 or integer != value:
        raise ValueError(f"predictor_config.{key} must be a positive integer.")
    return integer


def validate_source_v1_v2(
    source: Mapping[str, Any], *, spec: ActorFreeTDV2MethodSpec
) -> None:
    require_exact_values(
        source,
        {
            "method": f"{SOURCE_V1_FAMILY}_{spec.variant}",
            "method_family": SOURCE_V1_FAMILY,
            "variant": spec.variant,
            "implementation_version": SOURCE_V1_IMPLEMENTATION_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "deployment_checkpoint_version": 1,
            "source_seed": 3072,
            "source_epoch": SOURCE_V1_EPOCH,
            "source_global_step": SOURCE_V1_GLOBAL_STEP,
            "checkpoint_sha256": SOURCE_V1_SHA256[spec.variant],
            "source_code_revision": SOURCE_V1_CODE_REVISION,
            "optimizer_state": "reset",
        },
        label="predictor_config.source_v1",
    )


def validate_actor_free_td_v2_payload(
    payload: Mapping[str, Any],
    *,
    spec: ActorFreeTDV2MethodSpec,
) -> dict[str, Any]:
    """Validate a V2 deployment payload without constructing its modules."""

    checkpoint_identity: dict[str, Any] = {
        "method": spec.method,
        "method_family": spec.method_family,
        "variant": spec.variant,
        "implementation_version": spec.implementation_version,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
    }
    if spec.training_stage is not None:
        checkpoint_identity.update(
            {
                "stage": spec.training_stage,
                "initialization": spec.initialization,
            }
        )
    require_exact_values(
        payload,
        checkpoint_identity,
        label="checkpoint",
    )
    if spec.initialization_contract is not None:
        initialization_contract = payload.get("initialization_contract")
        if not isinstance(initialization_contract, Mapping) or dict(
            initialization_contract
        ) != dict(spec.initialization_contract):
            raise ValueError(
                "checkpoint.initialization_contract must exactly match the "
                "EMA-SG V1-fresh initialization contract."
            )
    required = {
        "world_model_state_dict",
        "target_world_model_state_dict",
        "world_model_config",
        "predictor_state_dict",
        "target_predictor_state_dict",
        "predictor_config",
        "source_v1_provenance",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(
            f"{spec.display_name} checkpoint is missing {sorted(missing)}."
        )
    for forbidden, explanation in (
        ("actor_state_dict", "V2 is actor-free."),
        ("successor_state_dict", "V2 stores G as predictor_state_dict."),
        ("action_encoder_state_dict", "V2 stores E_A inside its world model."),
    ):
        if forbidden in payload:
            raise ValueError(f"{explanation} Unexpected {forbidden}.")

    config_value = payload["predictor_config"]
    if not isinstance(config_value, Mapping):
        raise ValueError("checkpoint.predictor_config must be a mapping.")
    config = dict(config_value)
    required_config = {
        "method",
        "method_family",
        "variant",
        "implementation_version",
        "objective_version",
        "deployment_checkpoint_version",
        "architecture",
        "state_dim",
        "raw_action_dim",
        "action_dim",
        "action_embedding_dim",
        "task_dim",
        "output_dim",
        "hidden_dim",
        "hidden_layers",
        "embedding_layers",
        "gamma",
        "target_ema_decay",
        "target_world_ema_decay",
        "task_sampling",
        "joint_objective",
        "source_v1",
        "source_artifacts",
    }
    missing_config = required_config - config.keys()
    if missing_config:
        raise ValueError(f"predictor_config is missing {sorted(missing_config)}.")
    predictor_identity: dict[str, Any] = {
        "method": spec.method,
        "method_family": spec.method_family,
        "variant": spec.variant,
        "implementation_version": spec.implementation_version,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        "architecture": "td_jepa_forward_map_v1",
        "state_dim": V2_STATE_DIM,
        "raw_action_dim": V2_RAW_ACTION_DIM,
        "action_dim": V2_ACTION_DIM,
        "action_embedding_dim": V2_ACTION_EMBEDDING_DIM,
        "task_dim": V2_TASK_DIM,
        "output_dim": V2_OUTPUT_DIM,
        "hidden_dim": 256,
        "hidden_layers": 1,
        "embedding_layers": 2,
        "num_parallel": 1,
        "action_processing": "online_shared_lewm_action_encoder",
        "shared_lewm_action_encoder": True,
        "action_encoder_trainable": True,
        "action_encoder_source": "world_model.action_encoder",
        "state_parameterization": "coupled_online_lewm_latent",
        "goal_conditioning": "task_input",
        "bootstrap_action": "ema_dataset_next_action_embedding",
        "actor": "none",
        "reward": "none",
        "gamma": 0.95,
        "target_ema_decay": 0.995,
        "target_world_ema_decay": 0.995,
        "loss_warmup_fraction": 0.05,
    }
    if spec.training_stage is not None:
        predictor_identity.update(
            {
                "stage": spec.training_stage,
                "initialization": spec.initialization,
            }
        )
    require_exact_values(
        config,
        predictor_identity,
        label="predictor_config",
    )
    if spec.initialization_contract is not None:
        initialization_contract = config.get("initialization_contract")
        if not isinstance(initialization_contract, Mapping) or dict(
            initialization_contract
        ) != dict(spec.initialization_contract):
            raise ValueError(
                "predictor_config.initialization_contract must exactly match "
                "the EMA-SG V1-fresh initialization contract."
            )
    for key in (
        "state_dim",
        "raw_action_dim",
        "action_dim",
        "action_embedding_dim",
        "task_dim",
        "output_dim",
        "hidden_dim",
    ):
        _positive_integer(config, key)
    for key in ("hidden_layers", "embedding_layers"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"predictor_config.{key} must be non-negative.")
    if config["embedding_layers"] < 2:
        raise ValueError("predictor_config.embedding_layers must be at least two.")
    for key in ("gamma", "target_ema_decay", "target_world_ema_decay"):
        try:
            value = float(config[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"predictor_config.{key} must lie in [0, 1).") from error
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(f"predictor_config.{key} must lie in [0, 1).")

    task_sampling = config["task_sampling"]
    if not isinstance(task_sampling, Mapping):
        raise ValueError("predictor_config.task_sampling must be a mapping.")
    require_exact_values(
        task_sampling,
        {
            "sampling": "per_transition_bernoulli",
            "goal_probability": 0.5,
            "random_source": "isotropic_gaussian_sphere",
            "goal_source": "uniform_reachable_future_ema_latent_same_clip",
            "normalization": "sqrt_dim_l2_sphere",
            "mix_unit": "flattened_transition_minibatch",
        },
        label="predictor_config.task_sampling",
    )
    objective = config["joint_objective"]
    if not isinstance(objective, Mapping):
        raise ValueError("predictor_config.joint_objective must be a mapping.")
    local_prediction_contract: dict[str, Any] = {
        "local_prediction": spec.local_prediction,
    }
    if spec.local_prediction_target is not None:
        local_prediction_contract.update(
            {
                "local_prediction_target": spec.local_prediction_target,
                "local_prediction_target_gradient": (
                    spec.local_prediction_target_gradient
                ),
            }
        )
    require_exact_values(
        objective,
        {
            **local_prediction_contract,
            "local_prediction_weight": 1.0,
            "regularization": "original_lewm_sigreg",
            "target_encoder": "ema_world_model",
            "td_target": "ema_next_latent_plus_ema_predictor_dataset_next_action",
            "bootstrap_action": "ema_world_model_action_encoder",
            "real_td_weight": 1.0,
            "predicted_td_weight": 1.0,
            "predicted_context_detach": False,
            "hybrid_reduction": "sum",
            "per_transition_td_reduction": "feature_sum",
            "batch_td_reduction": "transition_mean",
            "base_td_population": "all_transitions",
            "random_task_weight": 1.0,
            "goal_subset": "goal_derived_tasks_only",
            "final_weight_normalization": "mean_one_over_all_transitions",
            "weight_gradient": "stop_gradient",
            "candidate_td_targets": "none",
            "actor": "none",
            "reward": "none",
        },
        label="predictor_config.joint_objective",
    )
    missing_objective = set(spec.objective_keys) - objective.keys()
    if missing_objective:
        raise ValueError(f"joint_objective is missing {sorted(missing_objective)}.")
    spec.validate_method_config(config)

    source = config["source_v1"]
    if not isinstance(source, Mapping):
        raise ValueError("predictor_config.source_v1 must be a mapping.")
    validate_source_v1_v2(source, spec=spec)
    source_artifacts = config["source_artifacts"]
    if not isinstance(source_artifacts, Mapping):
        raise ValueError("predictor_config.source_artifacts must be a mapping.")
    require_exact_values(
        source_artifacts,
        SOURCE_ARTIFACTS,
        label="predictor_config.source_artifacts",
    )

    world_config = payload["world_model_config"]
    if not isinstance(world_config, Mapping):
        raise ValueError("checkpoint.world_model_config must be a mapping.")
    action_encoder_config = world_config.get("action_encoder")
    if not isinstance(action_encoder_config, Mapping):
        raise ValueError("V2 world_model_config must contain action_encoder.")
    require_exact_values(
        action_encoder_config,
        {"input_dim": V2_RAW_ACTION_DIM, "emb_dim": V2_ACTION_EMBEDDING_DIM},
        label="world_model_config.action_encoder",
    )

    provenance = payload["source_v1_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint.source_v1_provenance must be a mapping.")
    require_exact_values(
        provenance,
        {
            "checkpoint_sha256": SOURCE_V1_SHA256[spec.variant],
            "source_epoch": SOURCE_V1_EPOCH,
            "source_global_step": SOURCE_V1_GLOBAL_STEP,
            "optimizer_state_loaded": False,
            "target_world_initialization": "copy_of_v1_online_world_model",
        },
        label="source_v1_provenance",
    )
    if (
        not isinstance(provenance.get("checkpoint_path"), str)
        or not provenance["checkpoint_path"]
    ):
        raise ValueError("source_v1_provenance.checkpoint_path must be non-empty.")
    return config


def load_actor_free_td_v2_checkpoint(
    checkpoint_path: str | Path,
    *,
    spec: ActorFreeTDV2MethodSpec,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, ActorFreeTDJEPAPredictorV1, dict[str, Any], dict[str, Any]]:
    """Restore and freeze the online V2 world model and shared predictor."""

    payload_value = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(payload_value, Mapping):
        raise ValueError("Deployment checkpoint must contain a mapping payload.")
    payload = dict(payload_value)
    config = validate_actor_free_td_v2_payload(payload, spec=spec)

    predictor = ActorFreeTDJEPAPredictorV1(
        hidden_dim=int(config["hidden_dim"]),
        hidden_layers=int(config["hidden_layers"]),
        embedding_layers=int(config["embedding_layers"]),
    )
    if any("action_encoder" in key for key in payload["predictor_state_dict"]):
        raise ValueError("V2 predictor_state_dict must not duplicate action_encoder.")
    predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    target_predictor = predictor.make_target()
    if any("action_encoder" in key for key in payload["target_predictor_state_dict"]):
        raise ValueError(
            "V2 target_predictor_state_dict must not duplicate action_encoder."
        )
    target_predictor.load_state_dict(
        payload["target_predictor_state_dict"], strict=True
    )

    import hydra
    from omegaconf import OmegaConf

    world_config = OmegaConf.create(payload["world_model_config"])
    world_model = hydra.utils.instantiate(world_config)
    target_world_model = hydra.utils.instantiate(world_config)
    world_model.load_state_dict(payload["world_model_state_dict"], strict=True)
    target_world_model.load_state_dict(
        payload["target_world_model_state_dict"], strict=True
    )
    for module in (world_model, target_world_model, predictor, target_predictor):
        module.eval().requires_grad_(False)
    action_encoder = getattr(world_model, "action_encoder", None)
    validate_lewm_action_encoder_v2(action_encoder)
    validate_lewm_action_encoder_v2(getattr(target_world_model, "action_encoder", None))
    return world_model, predictor, config, payload


class ActorFreeTDLeWMV2(ActorFreeTDLeWMV1):
    """V2 deployment wrapper with V1's audited CEM scoring topology."""

    implementation_version = IMPLEMENTATION_VERSION
    method_family = METHOD_FAMILY
    supported_score_modes = V2_SCORE_MODES

    def get_cost(
        self,
        info_dict: dict[str, Any],
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        if self.score_mode != "g_only_f_rollout_mean":
            return super().get_cost(info_dict, action_candidates)
        if action_candidates.ndim != 4:
            raise ValueError(
                "action_candidates must have shape (batch, samples, horizon, 25)."
            )
        if action_candidates.shape[2] != 5:
            raise ValueError(
                "V2 g_only_f_rollout_mean requires CEM planning horizon=5."
            )
        return self._g_only_f_rollout_mean_cost(info_dict, action_candidates)

    def _g_only_f_rollout_mean_cost(
        self,
        info_dict: dict[str, Any],
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Score aligned ``(z_(k-1), A_k)`` pairs for an arbitrary horizon."""

        if "goal" not in info_dict and "goal_emb" not in info_dict:
            raise AssertionError("goal not in info_dict")
        if action_candidates.ndim != 4:
            raise ValueError(
                "action_candidates must have shape (batch, samples, horizon, 25)."
            )
        if action_candidates.shape[-1] != V2_RAW_ACTION_DIM:
            raise ValueError("V2 expects normalized raw 25D action blocks.")
        if not action_candidates.is_floating_point():
            raise TypeError("action_candidates must have a floating-point dtype.")
        if not bool(torch.isfinite(action_candidates).all()):
            raise ValueError("action_candidates must contain only finite values.")
        batch, samples, horizon = action_candidates.shape[:3]
        if horizon <= 0:
            raise ValueError("The planning horizon must be positive.")

        current = self._current_state_for_samples(
            dict(info_dict),
            batch=batch,
            samples=samples,
            reference=action_candidates,
        )
        goal = self._goal_for_samples(
            info_dict,
            batch=batch,
            samples=samples,
            reference=action_candidates,
        )
        task = project_tasks_to_sphere_v1(goal)
        future = self._rollout_future(
            info_dict,
            action_candidates,
            batch=batch,
            samples=samples,
            horizon=horizon,
        )
        predecessors = torch.cat(
            (current.unsqueeze(-2), future[..., :-1, :]),
            dim=-2,
        )
        expected_shape = (batch, samples, horizon, V2_STATE_DIM)
        if predecessors.shape != expected_shape:
            raise ValueError(
                "V2 rollout-mean predecessor states must align one-to-one with "
                "candidate action blocks."
            )
        step_tasks = task.unsqueeze(-2).expand(expected_shape)
        scores = self._goal_score(
            predecessors,
            action_candidates,
            step_tasks,
        )
        if scores.shape != (batch, samples, horizon):
            raise ValueError(
                "V2 rollout-mean G scores must have shape (batch, samples, horizon)."
            )
        return -scores.mean(dim=-1)


def make_actor_free_td_v2_policy(
    *,
    world_model: nn.Module,
    predictor: ActorFreeTDJEPAPredictorV1,
    planning: dict[str, Any],
    gamma: float,
    process: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
    score_mode: str | None = None,
    g_first_weight: float | None = None,
    wrapper_class: type[ActorFreeTDLeWMV1] = ActorFreeTDLeWMV2,
):
    """Build the public Stable World Model CEM policy around V2 online modules."""

    resolved_mode = score_mode or wrapper_class.default_score_mode
    if resolved_mode not in V2_SCORE_MODES:
        raise ValueError(f"Unsupported V2 score mode {resolved_mode!r}.")
    if int(planning["action_block"]) != ACTION_BLOCK_STEPS:
        raise ValueError("V2 requires planning.action_block=5.")
    if resolved_mode == "g_only" and int(planning["horizon"]) != 1:
        raise ValueError("V2 g_only requires planning.horizon=1.")
    if resolved_mode == "f_plus_g_first" and int(planning["horizon"]) != 5:
        raise ValueError("V2 f_plus_g_first requires planning.horizon=5.")
    if resolved_mode == "g_only_f_rollout_mean" and int(planning["horizon"]) != 5:
        raise ValueError("V2 g_only_f_rollout_mean requires planning.horizon=5.")
    resolved_g_first_weight = _resolve_g_first_weight(
        resolved_mode,
        g_first_weight,
    )

    import stable_worldmodel as swm

    wrapped = wrapper_class(
        world_model,
        predictor,
        gamma=gamma,
        score_mode=resolved_mode,
        g_first_weight=resolved_g_first_weight,
        lewm_history_size=LEWM_HISTORY_SIZE,
    ).to(device)
    wrapped.eval().requires_grad_(False)
    solver = swm.solver.CEMSolver(
        model=wrapped,
        batch_size=planning["solver_batch_size"],
        num_samples=planning["candidates"],
        var_scale=planning["initial_variance"],
        n_steps=planning["iterations"],
        topk=planning["elites"],
        device=device,
        seed=planning["planning_seed"],
    )
    plan_config = swm.PlanConfig(
        horizon=planning["horizon"],
        receding_horizon=planning["receding_horizon"],
        history_len=planning.get(
            "plan_config_history_len", planning.get("history_len", 1)
        ),
        action_block=planning["action_block"],
        warm_start=planning["warm_start"],
    )
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
    )


__all__ = [
    "ActorFreeTDLeWMV2",
    "ActorFreeTDV2MethodSpec",
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "IMPLEMENTATION_VERSION",
    "METHOD_FAMILY",
    "OBJECTIVE_VERSION",
    "SOURCE_ARTIFACTS",
    "SOURCE_V1_SHA256",
    "V2_SCORE_MODES",
    "load_actor_free_td_v2_checkpoint",
    "make_actor_free_td_v2_policy",
    "require_exact_values",
    "require_positive_float",
    "validate_actor_free_td_v2_payload",
    "validate_source_v1_v2",
]
