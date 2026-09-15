"""Independent variable-length trainer; the original fixed-H5 trainer is untouched."""
from collections import defaultdict

import numpy as np
import torch

from tdwm.methods.effplan import planner_loss, refine_state_path
from tdwm.methods.effplan_variable import generate_variable_state_path
from tdwm.methods.effplan_safety import PlannerSafetyRuntime, require_finite
from tdwm.training.eff_data import EffPlanReplayBatch
from tdwm.training.effplan_runtime import EffPlanTrainer, latent_planning_context
from tdwm.training.effplan_variable_data import VariablePlannerBatch


class VariableEffPlanTrainer(EffPlanTrainer):
    # Reuse checkpoint/optimizer/RNG handling and the unchanged P/G/V networks.
    # Only path assembly, bucket aggregation and the CEM horizon differ.
    def _loss(self, batch: EffPlanReplayBatch, *, validation: bool = False):
        import stable_worldmodel as swm
        from gymnasium.spaces import Box

        states = batch.real_states.to(self.device).detach()
        self.safety_runtime = (
            None
            if self.settings.safety is None
            else PlannerSafetyRuntime(self.settings.safety)
        )
        valid = batch.trajectory_valid.to(self.device)
        calibration = self.solver is not None and (
            validation or (self.global_step + 1) % self.settings.calibration_interval == 0
        )
        if calibration and self.settings.calibration_batch_size is not None:
            # The replay batch is IID; its first K paths are an unbiased subset.
            states = states[:self.settings.calibration_batch_size]
            valid = valid[:len(states)]
        self.last_calibration = calibration
        self.last_cem_paths = len(states) if calibration else 0
        if states.ndim != 3 or (states.shape[1] < 3 or states.shape[2] != 192) or batch.state_stride != 5:
            raise ValueError(
                "Variable P needs >=3 states at primitive stride five."
            )
        if self.settings.phase == "generation" and not bool(valid.all()):
            raise ValueError(
                "Generation phase requires real connected midpoint labels."
            )
        horizon = states.shape[1] - 1
        nodes = generate_variable_state_path(
            self.planner,
            states[:, 0],
            states[:, -1],
            self.value,
            horizon=horizon,
            epsilon=self.settings.epsilon,
            safety=self.safety_runtime,
        )
        loss_args = dict(
            real_states=states,
            value=self.value,
            epsilon=self.settings.epsilon,
            trajectory_coefficient=self.settings.trajectory_coefficient,
            efficiency_coefficient=self.settings.efficiency_coefficient,
            dynamics_coefficient=self.settings.dynamics_coefficient,
            trajectory_valid=valid,
            safety=self.safety_runtime,
        )
        losses = []
        if self.solver is None:
            losses.append(planner_loss(nodes, predicted_future=None, **loss_args))
        elif not calibration:
            loss_args["dynamics_coefficient"] = 0.0
            for _ in self.settings.search_iterations:
                nodes = refine_state_path(
                    self.planner, nodes, self.value, predicted_future=None,
                    epsilon=self.settings.epsilon, dynamics_coefficient=0.0,
                    safety=self.safety_runtime,
                )
                losses.append(planner_loss(nodes, predicted_future=None, **loss_args))
        else:
            self.solver.configure(
                action_space=Box(-1.0, 1.0, shape=(len(states), 5), dtype=np.float32),
                n_envs=len(states),
                config=swm.PlanConfig(
                    horizon=horizon,
                    receding_horizon=horizon,
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
                    safety=self.safety_runtime,
                )
                losses.append(planner_loss(nodes, predicted_future=future, **loss_args))
        selected = losses if self.settings.supervision == "mean_rounds" else losses[-1:]
        loss = torch.stack([x.total for x in selected]).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite planner loss; optimizer not updated.")
        return loss, selected, valid


    def _aggregate(self, batch: VariablePlannerBatch, *, validation: bool):
        if not batch.samples:
            raise ValueError("Empty variable planner batch.")
        calibration = self.solver is not None and (
            validation or (self.global_step + 1) % self.settings.calibration_interval == 0
        )
        limit = self.settings.calibration_batch_size if calibration else None
        chosen = batch.samples[:limit]
        count = len(chosen)
        metrics = defaultdict(float)
        active = 0
        for group in batch.groups(limit):
            size, length = group.real_states.shape[:2]
            if length == 2:
                # There is no P output for adjacent endpoints. No fictitious
                # midpoint, NaN empty mean, gradient, CEM search, or weight decay.
                metrics["no_interior_paths"] += size
                continue
            with torch.set_grad_enabled(not validation):
                loss, losses, valid = self._loss(group, validation=validation)
                if not validation:
                    (loss * (size / count)).backward()
            active += size
            for key, field in (
                ("total_loss", "total"), ("trajectory_loss", "trajectory"),
                ("efficiency_loss", "efficiency"), ("dynamics_loss", "dynamics"),
            ):
                metrics[key] += torch.stack([getattr(v, field).detach() for v in losses]).mean().item() * size / count
            if self.safety_runtime is not None:
                for key, value in self.safety_runtime.metrics().items():
                    metrics[key] += value * size / count
            metrics["trajectory_labelled_paths"] += int(valid.sum())
            metrics["cem_candidate_rollouts"] += self.last_cem_paths * self.settings.cem_candidates * sum(self.settings.search_iterations)
            metrics["returned_action_rerolls"] += self.last_cem_paths * len(self.settings.search_iterations)
        lengths = [s.real_states.shape[1] for s in batch.samples]
        for key in ("total_loss", "trajectory_loss", "efficiency_loss", "dynamics_loss",
                    "no_interior_paths", "trajectory_labelled_paths",
                    "cem_candidate_rollouts", "returned_action_rerolls"):
            metrics.setdefault(key, 0.0)
        metrics.update(
            sampled_paths=len(lengths), sampled_states_min=min(lengths),
            sampled_states_max=max(lengths), sampled_states_mean=sum(lengths) / len(lengths),
            optimized_paths=active, selected_paths=count, calibration_update=calibration,
        )
        return dict(metrics), active

    def step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        metrics, active = self._aggregate(batch, validation=False)
        if active:
            norm = torch.nn.utils.clip_grad_norm_(
                self.planner.parameters(), self.settings.gradient_clip, error_if_nonfinite=True,
            )
            self.optimizer.step()
        else:
            norm = torch.tensor(0.0)
        self.global_step += 1
        return dict(metrics, global_step=self.global_step, phase=self.settings.phase,
                    gradient_norm=norm.item())

    def validate(self, batch):
        mode = self.planner.training
        self.planner.eval()
        cem_rng = None if self.solver is None else self.solver.torch_gen.get_state()
        try:
            return self._aggregate(batch, validation=True)[0]
        finally:
            self.planner.train(mode)
            if cem_rng is not None:
                self.solver.torch_gen.set_state(cem_rng)
