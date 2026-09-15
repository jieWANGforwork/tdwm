"""Exact-objective fast path for both P stages; CEM calibration is unchanged."""
import torch

from tdwm.methods.effplan_packed import PackedPaths, packed_generate, packed_refine
from tdwm.methods.effplan_safety import PlannerSafetyRuntime, require_finite
from tdwm.training.effplan_variable_runtime import VariableEffPlanTrainer


class PackedEffPlanTrainer(VariableEffPlanTrainer):
    def _aggregate(self, batch, *, validation):
        calibration = self.solver is not None and (
            validation or (self.global_step+1) % self.settings.calibration_interval == 0
        )
        # Preserve exact solver draw/order, frequency, path cap and CEM budget.
        if calibration:
            return super()._aggregate(batch, validation=validation)
        paths = PackedPaths.from_samples(batch.samples, self.device)
        if paths is None:
            return super()._aggregate(batch, validation=validation)
        safety = None if self.settings.safety is None else PlannerSafetyRuntime(self.settings.safety)
        with torch.set_grad_enabled(not validation):
            nodes = packed_generate(self.planner, paths, self.value,
                                    epsilon=self.settings.epsilon, safety=safety)
            rounds = len(self.settings.search_iterations) if self.solver is not None else 1
            losses, trajectories, efficiencies = [], [], []
            for index in range(rounds):
                if self.solver is not None:
                    nodes = packed_refine(self.planner, nodes, paths, self.value,
                                          epsilon=self.settings.epsilon, safety=safety)
                if self.settings.supervision == "final" and index < rounds-1:
                    continue
                trajectory = paths.trajectory(nodes)
                # A generation-only logging term needs no backward graph.
                with torch.set_grad_enabled(not validation and self.settings.efficiency_coefficient != 0):
                    efficiency = -paths.efficiency(nodes, self.value, self.settings.epsilon, safety).sum()/paths.batch_count
                loss = self.settings.trajectory_coefficient * trajectory + self.settings.efficiency_coefficient * efficiency
                losses.append(loss)
                trajectories.append(trajectory)
                efficiencies.append(efficiency)
            loss = torch.stack(losses).mean()
            require_finite(loss, "packed P loss")
            if not validation:
                loss.backward()
        lengths = [s.real_states.shape[1] for s in batch.samples]
        active = len(paths.starts)
        metrics = dict(
            total_loss=loss.detach().item(), trajectory_loss=torch.stack(trajectories).detach().mean().item(),
            efficiency_loss=torch.stack(efficiencies).detach().mean().item(), dynamics_loss=0.0,
            no_interior_paths=len(lengths)-active, trajectory_labelled_paths=active,
            cem_candidate_rollouts=0, returned_action_rerolls=0,
            sampled_paths=len(lengths), sampled_states_min=min(lengths), sampled_states_max=max(lengths),
            sampled_states_mean=sum(lengths)/len(lengths), optimized_paths=active,
            selected_paths=len(lengths), calibration_update=False,
        )
        if safety is not None:
            # Diagnostics are over all actual nodes/paths, not means of buckets.
            metrics.update(safety.metrics())
        return metrics, active
