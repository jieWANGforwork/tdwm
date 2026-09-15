"""Predeclared tau=0.8 study: six paired evaluations, no training or threshold search."""

import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path

from tdwm.training.eff_run import run_directory_lock, write_json_atomic

THRESHOLD = 0.8
VARIANTS = ('old_v', 'extra_work_v')
OFFSETS = (25, 50, 100)
SCORE = 'adaptive_local_efficiency_rolling_v1'
DISTANCE_SCORE = 'adaptive_local_distance_efficiency_rolling_v2'
DISTANCE_ONLY_SCORE = 'adaptive_local_distance_only_rolling_v1'


def build_jobs(*, repo, runs_root, output_root, dataset, lewm_checkpoint, python, devices,
               local_distance_limit=None, distance_only=False):
    """Build exactly six jobs; paths/devices are explicit, never discovered by mutation."""
    repo, runs_root, output_root = (Path(p).expanduser().resolve() for p in (repo, runs_root, output_root))
    if not devices or any(not str(d).isdigit() for d in devices):
        raise ValueError('Provide explicit numeric GPU indices.')
    if distance_only and local_distance_limit is None:
        raise ValueError('Distance-only study requires an explicit distance limit.')
    if local_distance_limit is not None:
        from tdwm.methods.effplan_efficiency import validate_local_distance_limit
        validate_local_distance_limit(local_distance_limit)
    jobs = []
    for variant in VARIANTS:
        old = variant == 'old_v'
        config = repo/'configs/experiment'/(
            'effplan_cube_stable_p_sparse_v1.yaml' if old else 'effplan_cube_extra_work_sparse_v1.yaml')
        eff = runs_root/('formal/train' if old else 'corrected_v_compare_20260914/extra_work_v/train')
        planner = runs_root/'sparse_v_compare_20260914'/variant/'planner_refinement'
        for offset in OFFSETS:
            output = output_root/variant/f'O{offset}'
            inputs = dict(config=config, dataset=Path(dataset).expanduser().resolve(),
                          lewm_checkpoint=Path(lewm_checkpoint).expanduser().resolve(),
                          selection=runs_root/'formal/selections'/f'o{offset}_selection.json',
                          eff_checkpoint=eff/'last.pt', eff_manifest=eff/'training_manifest.json',
                          planner_checkpoint=planner/'last.pt', planner_manifest=planner/'planner_manifest.json')
            command = [str(python), '-u', str(repo/'scripts/evaluate_effplan.py'), 'evaluate']
            for key, value in inputs.items():
                command.extend(['--'+key.replace('_', '-'), str(value)])
            command.extend(['--output-dir', str(output), '--device', 'cuda', '--method', 'EffPlan',
                            '--adaptive-rolling'])
            command.extend(['--adaptive-distance-only'] if distance_only else
                           ['--adaptive-efficiency-threshold', str(THRESHOLD)])
            if local_distance_limit is not None:
                command.extend(['--adaptive-local-distance-limit', str(local_distance_limit)])
            jobs.append(dict(variant=variant, offset=offset, output=str(output),
                             log=str(output.with_suffix('.log')), command=command,
                             gpu=str(devices[len(jobs) % len(devices)]),
                             inputs={key: str(value) for key, value in inputs.items()}))
    return jobs


def run_jobs(jobs, *, repo, output_root):
    """All six start concurrently. Existing runs are never overwritten or relaunched."""
    output = Path(output_root)
    for job in jobs:
        for path in job['inputs'].values():
            if not Path(path).exists():
                raise FileNotFoundError(path)
        if Path(job['output']).exists() or Path(job['log']).exists():
            raise FileExistsError('Use a new study output; refusing to overwrite '+job['output'])
    with run_directory_lock(output):
        manifest_path = output/'study_manifest.json'
        if manifest_path.exists():
            raise FileExistsError('Study already launched; inspect recorded PIDs, do not duplicate.')
        distance_modes = {'--adaptive-distance-only' in j['command'] for j in jobs}
        if len(distance_modes) != 1:
            raise ValueError('Do not mix gate types in one study.')
        manifest = dict(status='launching', threshold=None if True in distance_modes else THRESHOLD, jobs=jobs,
                        formal_episodes=300, training=False)
        write_json_atomic(manifest_path, manifest)
        children = []
        try:
            for job in jobs:
                Path(job['log']).parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=job['gpu'], MUJOCO_GL='osmesa',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           PYTHONPATH=str(Path(repo)/'src'))
                with Path(job['log']).open('x') as log:
                    child = subprocess.Popen(job['command'], cwd=repo, env=env,
                                             stdout=log, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, start_new_session=True)
                job['pid'] = child.pid
                children.append((job, child))
                write_json_atomic(manifest_path, manifest)
            manifest['status'] = 'running'
            write_json_atomic(manifest_path, manifest)
            for job, child in children:
                job['exit_code'] = child.wait()
                write_json_atomic(manifest_path, manifest)
            manifest['status'] = 'evaluated' if all(j['exit_code'] == 0 for j in jobs) else 'failed'
        except BaseException:
            manifest['status'] = 'interrupted_or_launch_failed; inspect recorded PIDs before recovery'
            write_json_atomic(manifest_path, manifest)
            raise
        write_json_atomic(manifest_path, manifest)
        if manifest['status'] != 'evaluated':
            raise RuntimeError('Some evaluations failed; inspect per-job logs, no result substituted.')
    return manifest


def paired_comparison(current, reference):
    if current['selection_sha256'] != reference['selection_sha256']:
        raise ValueError('Selection differs; not a paired comparison.')
    a, b = current['episode_results'], reference['episode_results']
    if len(a) != 50 or len(b) != 50:
        raise ValueError('Expected 50 complete episodes.')
    for x, y in zip(a, b):
        if any(x[k] != y[k] for k in ('index', 'episode', 'start', 'goal')):
            raise ValueError('Episode identity/order differs.')
    new = sum(x['success'] and not y['success'] for x, y in zip(a, b))
    lost = sum(y['success'] and not x['success'] for x, y in zip(a, b))
    return dict(new=new, lost=lost, delta_pp=2*(new-lost))


def audit_records(result, rounds, *, offset, local_distance_limit=None, distance_only=False):
    if distance_only and local_distance_limit is None:
        raise ValueError('Distance-only audit needs its calibrated distance limit.')
    episodes = result['episode_results']
    if len(episodes) != 50 or len(rounds) != 50 or not result['formal']:
        raise ValueError('Formal result incomplete.')
    success = sum(e['success'] for e in episodes)
    if success != result['successes'] or not math.isclose(result['success_rate'], success*2):
        raise ValueError('Aggregate/episode success mismatch.')
    stops, blocks, gate_checks = Counter(), [], []
    for i, (episode, decisions) in enumerate(zip(episodes, rounds)):
        position = 0
        if not decisions or len(decisions) != episode['planning_calls']:
            raise ValueError('Missing decision records.')
        for j, r in enumerate(decisions):
            n = r['action_blocks']
            if not (r['index'] == i and r['decision_index'] == j
                    and r['start_primitive_step'] == position
                    and r['remaining_budget_before'] == 2*offset-position
                    and r['max_action_blocks'] == (2*offset-position)//5
                    and r['planned_primitive_steps'] == 5*n == 5*(r['intermediate_nodes']+1)
                    and 0 < r['executed_primitive_steps'] <= 5*n <= 2*offset-position):
                raise ValueError('Decision budget or node/action count mismatch.')
            if j+1 < len(decisions) and r['executed_primitive_steps'] != 5*n:
                raise ValueError('Replanned before consuming full decision.')
            criterion = ('local_distance' if distance_only else
                         'local_efficiency' if local_distance_limit is None else 'local_distance_efficiency')
            threshold = None if distance_only else THRESHOLD
            if r['criterion'] != criterion or r['efficiency_threshold'] != threshold:
                raise ValueError('Unexpected gate/threshold.')
            if local_distance_limit is not None and r.get('local_distance_limit') != local_distance_limit:
                raise ValueError('Distance limit changed between decisions.')
            position += r['executed_primitive_steps']
            blocks.append(n)
            for gate in r['split_attempts']:
                if gate['threshold'] != threshold:
                    raise ValueError('Gate threshold changed.')
                if local_distance_limit is not None and gate.get('local_distance_limit') != local_distance_limit:
                    raise ValueError('Distance limit changed between segments.')
                eta = gate['efficiency']
                if distance_only:
                    d = gate['distance']
                    if not math.isfinite(d) or d < 0 or eta is not None or gate['predicted_work'] is not None:
                        raise ValueError('Invalid distance-only record.')
                    if d <= 1e-6:
                        valid = not gate['accepted'] and gate['stop_reason'] == 'degenerate_segment'
                    elif d <= local_distance_limit:
                        valid = not gate['accepted'] and gate['stop_reason'] == 'distance_sufficient'
                    else:
                        valid = (gate['accepted'] and gate['stop_reason'] is None) or (
                            not gate['accepted'] and gate['stop_reason'] in ('budget_cap', 'duplicate_midpoint'))
                    if not valid:
                        raise ValueError('Distance-only stop/split contradicts distance boundary.')
                elif eta is None:
                    if (gate['distance'] > 1e-6 or gate['stop_reason'] != 'degenerate_segment'
                            or gate['accepted']):
                        raise ValueError('Invalid degenerate endpoint guard.')
                if eta is not None:
                    expected = gate['distance']/(gate['predicted_work']+1e-6)
                    if not math.isfinite(eta) or not math.isclose(eta, expected, rel_tol=1e-5, abs_tol=1e-7):
                        raise ValueError('Efficiency calculation mismatch.')
                    gate_checks.append(gate)
                    short = local_distance_limit is None or gate['distance'] <= local_distance_limit
                    if gate['accepted'] and eta >= THRESHOLD and short:
                        raise ValueError('Split a leaf satisfying both stopping conditions.')
                    sufficient = ('efficiency_sufficient' if local_distance_limit is None
                                  else 'distance_and_efficiency_sufficient')
                    if gate['stop_reason'] == sufficient and not (eta >= THRESHOLD and short):
                        raise ValueError('Stopped leaf without satisfying the distance/efficiency gate.')
                    if local_distance_limit is not None and gate['stop_reason'] == 'efficiency_sufficient':
                        raise ValueError('Used the legacy efficiency-only stop in a distance-gated run.')
                if not gate['accepted']:
                    stops[gate['stop_reason']] += 1
        if position != episode['executed_primitive_steps'] or (not episode['success'] and position != 2*offset):
            raise ValueError('Failure ended before budget or execution count differs.')
    floor_stops = sum(g['raw_work'] < g['distance'] and g['stop_reason'] in (
        'efficiency_sufficient', 'distance_and_efficiency_sufficient') for g in gate_checks)
    return dict(successes=success, success_rate=2*success, decisions=len(blocks),
                mean_blocks=sum(blocks)/len(blocks), max_blocks=max(blocks),
                single_block_decisions=sum(n == 1 for n in blocks), stop_reasons=dict(stops),
                geometric_floor_stop_count=floor_stops, evaluated_segments=len(gate_checks),
                mean_executed_steps=sum(e['executed_primitive_steps'] for e in episodes)/50)


def analyze_study(*, runs_root, output_root):
    """Audit against fixed-5 and first adaptive rolling; leave raw artifacts intact."""
    load = lambda p: json.loads(Path(p).read_text())
    root, output = Path(runs_root), Path(output_root)
    summary = {}
    table = {'fixed_5_blocks': [], 'adaptive_work_gain': [], 'adaptive_efficiency_080': []}
    study_gate = None
    for variant in VARIANTS:
        for offset in OFFSETS:
            p = output/variant/f'O{offset}'
            r, m = load(p/'result.json'), load(p/'protocol_manifest.json')
            if m['status'] != 'complete' or r['score_mode'] not in (SCORE, DISTANCE_SCORE, DISTANCE_ONLY_SCORE):
                raise ValueError('Wrong mode or incomplete evaluation.')
            local_limit = m['protocol_overrides']['adaptive_rolling'].get('local_distance_limit')
            distance_only = r['score_mode'] == DISTANCE_ONLY_SCORE
            if (r['score_mode'] in (DISTANCE_SCORE, DISTANCE_ONLY_SCORE)) != (local_limit is not None):
                raise ValueError('Distance score/manifest mismatch.')
            if local_limit is not None:
                from tdwm.methods.effplan_efficiency import validate_local_distance_limit
                validate_local_distance_limit(local_limit)
            current_gate = (r['score_mode'], local_limit)
            if study_gate is not None and current_gate != study_gate:
                raise ValueError('Do not mix distance scales or gate versions in one study.')
            study_gate = current_gate
            if m['protocol_overrides']['adaptive_rolling']['efficiency_threshold'] != (None if distance_only else THRESHOLD):
                raise ValueError('Threshold differs from predeclared study.')
            if r['episode_results'] != load(p/'episode_results.json')['episodes']:
                raise ValueError('Episode sidecar differs.')
            rounds = load(p/'adaptive_planning.json')['episodes']
            audit = audit_records(r, rounds, offset=offset, local_distance_limit=local_limit,
                                  distance_only=distance_only)
            audit.update(score_mode=r['score_mode'], local_distance_limit=local_limit)
            refs = {'fixed_5_blocks': root/'sparse_v_compare_20260914'/variant/'eval/EffPlan'/f'O{offset}',
                    'adaptive_work_gain': root/'adaptive_rolling_20260915'/variant/f'O{offset}'}
            for key, rp in refs.items():
                rr, rm = load(rp/'result.json'), load(rp/'protocol_manifest.json')
                if any(m['checkpoints'][name]['sha256'] != rm['checkpoints'][name]['sha256']
                       for name in ('LeWM', 'Eff', 'EffPlan')):
                    raise ValueError('Checkpoint mismatch; not an inference-only comparison.')
                audit[key] = paired_comparison(r, rr)
                table[key].append(rr['success_rate'])
            import torch
            plans = torch.load(p/'adaptive_plans.pt', map_location='cpu', weights_only=True)
            if len(plans) != 50:
                raise ValueError('Missing episode plans.')
            for decisions, records in zip(plans, rounds):
                if len(decisions) != len(records):
                    raise ValueError('Missing decision artifacts.')
                for a, record in zip(decisions, records):
                    n = record['action_blocks']
                    if (a['actions'].shape != (1,n,25) or a['initial_nodes'].shape != (1,n+1,192)
                            or a['final_nodes'].shape != (1,n+1,192)
                            or not all(torch.isfinite(t).all() for t in a.values())):
                        raise ValueError('Invalid action/state artifact.')
                    if not torch.equal(a['initial_nodes'][:, [0,-1]], a['final_nodes'][:, [0,-1]]):
                        raise ValueError('P changed a fixed endpoint.')
                    if not torch.equal(a['initial_nodes'][:, -1], decisions[0]['initial_nodes'][:, -1]):
                        raise ValueError('Goal changed between decisions.')
            table['adaptive_efficiency_080'].append(r['success_rate'])
            summary[f'{variant}/O{offset}'] = audit
    if study_gate[0] == DISTANCE_SCORE:
        table['adaptive_distance_efficiency_080'] = table.pop('adaptive_efficiency_080')
    if study_gate[0] == DISTANCE_ONLY_SCORE:
        table['adaptive_distance_only'] = table.pop('adaptive_efficiency_080')
    lines = ['# Adaptive EffPlan: ' + study_gate[0],
             f'Local distance limit: {study_gate[1]} (None = historical efficiency-only rule).', '',
             '| Method | Original V O25 | Original V O50 | Original V O100 | Extra-work V O25 | Extra-work V O50 | Extra-work V O100 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, rates in table.items():
        lines.append('| '+name+' | '+' | '.join(f'{x:g}%' for x in rates)+' |')
    lines.extend(['', '## Paired changes and mechanism diagnostics', ''])
    for key, d in summary.items():
        lines.append(f"- {key}: versus fixed {d['fixed_5_blocks']}; versus first adaptive {d['adaptive_work_gain']}; "
                     f"single-block decisions {d['single_block_decisions']}/{d['decisions']}; "
                     f"mean/max blocks {d['mean_blocks']:.2f}/{d['max_blocks']}; "
                     f"stop reasons {d['stop_reasons']}; floor-driven stops {d['geometric_floor_stop_count']}.")
    lines.extend(['', '## Interpretation limits', '',
                  '- Positive paired deltas show gains on this draw only; this is one training seed and 50 pairs per protocol.',
                  '- Many single-block decisions suggest short-horizon planning persists; many budget-cap stops suggest the gate often fails to stop naturally.',
                  '- Floor-driven stops mean V underestimated geometric distance and the protected efficiency became high. They are not evidence of genuine high efficiency.',
                  '- Node counts change action-search dimension, feedback frequency and compute. Do not claim equal-compute superiority.',
                  '- P refinement occurs after the generation-time gate. Its later state changes can alter efficiencies; the saved gate is not a final reachability certificate.',
                  '- Interpret associations as diagnostic hypotheses, not established causal mechanisms. Do not tune tau on these formal outcomes.'])
    write_json_atomic(output/'study_analysis.json', summary)
    (output/'study_analysis.md').write_text('\n'.join(lines)+'\n')
    return summary
