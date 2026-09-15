import copy
from pathlib import Path

import pytest

from tdwm.evaluation import effplan_efficiency_study as study


def jobs(tmp_path):
    return study.build_jobs(repo=tmp_path/'repo', runs_root=tmp_path/'historical',
                            output_root=tmp_path/'new', dataset=tmp_path/'dataset.lance',
                            lewm_checkpoint=tmp_path/'f.pt', python='python', devices=['0', '1', '2'])


def test_predeclared_six_jobs_no_training_no_f_only_no_threshold_search(tmp_path):
    matrix = jobs(tmp_path)
    assert len(matrix) == 6
    assert [j['gpu'] for j in matrix] == ['0', '1', '2', '0', '1', '2']
    assert [(j['variant'], j['offset']) for j in matrix] == [
        (v,o) for v in study.VARIANTS for o in study.OFFSETS]
    assert len({j['output'] for j in matrix}) == 6
    for j in matrix:
        cmd = j['command']
        assert cmd[cmd.index('--adaptive-efficiency-threshold')+1] == '0.8'
        assert '--adaptive-rolling' in cmd and '--adaptive-one-shot' not in cmd
        assert '--offset-window' not in cmd
        assert cmd[cmd.index('--method')+1] == 'EffPlan'
    assert not (tmp_path/'new').exists(), 'preview must not create an output'


def test_launcher_starts_every_job_before_waiting_and_refuses_duplicate(tmp_path, monkeypatch):
    matrix = jobs(tmp_path)
    for job in matrix:
        for name in job['inputs'].values():
            p = Path(name); p.parent.mkdir(parents=True, exist_ok=True); p.touch()
    started = []
    class Child:
        def __init__(self, command, **kwargs):
            started.append(command)
            self.pid = len(started)
            assert kwargs['env']['MUJOCO_GL'] == 'osmesa'
        def wait(self):
            assert len(started) == 6, 'jobs were serialized'
            return 0
    monkeypatch.setattr(study.subprocess, 'Popen', Child)
    result = study.run_jobs(matrix, repo=tmp_path/'repo', output_root=tmp_path/'new')
    assert result['status'] == 'evaluated'
    assert all('pid' in j and j['exit_code'] == 0 for j in result['jobs'])
    with pytest.raises(FileExistsError):
        study.run_jobs(matrix, repo=tmp_path/'repo', output_root=tmp_path/'new')
    assert len(started) == 6


def fixture_result(offset=25):
    episodes, records = [], []
    for i in range(50):
        episodes.append(dict(index=i, episode=8000+i, start=0, goal=offset, success=False,
                             planning_calls=2*offset//5, executed_primitive_steps=2*offset))
        rr = []
        for j in range(2*offset//5):
            rr.append(dict(index=i, decision_index=j, action_blocks=1, intermediate_nodes=0,
                           start_primitive_step=j*5, remaining_budget_before=2*offset-j*5,
                           max_action_blocks=2*offset//5-j, planned_primitive_steps=5, executed_primitive_steps=5,
                           criterion='local_efficiency', efficiency_threshold=0.8,
                           split_attempts=[dict(threshold=0.8, efficiency=10/(10+1e-6), distance=10,
                                                predicted_work=10, raw_work=2, accepted=False,
                                                stop_reason='efficiency_sufficient')]))
        records.append(rr)
    return dict(formal=True, successes=0, success_rate=0, selection_sha256='same',
                episode_results=episodes), records


def test_analysis_counts_single_block_floor_stops_and_full_budget():
    result, records = fixture_result()
    report = study.audit_records(result, records, offset=25)
    assert report['decisions'] == report['single_block_decisions'] == 500
    assert report['geometric_floor_stop_count'] == 500
    assert report['mean_executed_steps'] == 50


@pytest.mark.parametrize('fault', ['early', 'remaining', 'threshold', 'eta', 'false_stop'])
def test_analysis_rejects_protocol_or_gate_errors(fault):
    result, records = fixture_result()
    if fault == 'early':
        result['episode_results'][0]['executed_primitive_steps'] = 5
    elif fault == 'remaining':
        records[0][1]['remaining_budget_before'] = 50
    elif fault == 'threshold':
        records[0][0]['efficiency_threshold'] = 0.9
    elif fault == 'eta':
        records[0][0]['split_attempts'][0]['efficiency'] = 0.5
    else:
        g = records[0][0]['split_attempts'][0]
        g.update(predicted_work=20, raw_work=20, efficiency=10/(20+1e-6))
    with pytest.raises(ValueError):
        study.audit_records(result, records, offset=25)


def test_paired_new_lost_not_just_aggregate():
    current, _ = fixture_result()
    reference = copy.deepcopy(current)
    current['episode_results'][0]['success'] = True
    reference['episode_results'][1]['success'] = True
    assert study.paired_comparison(current, reference) == dict(new=1, lost=1, delta_pp=0)
    reference['episode_results'][1]['start'] = 1
    with pytest.raises(ValueError, match='identity'):
        study.paired_comparison(current, reference)


@pytest.mark.parametrize('local_limit,distance_only', [(None,False), (10.,False), (10.,True)])
def test_complete_analysis_writes_six_paired_reports_and_checks_saved_plans(tmp_path, local_limit, distance_only):
    import json
    import torch
    root, output = tmp_path/'history', tmp_path/'new'
    for v in study.VARIANTS:
        for o in study.OFFSETS:
            r, records = fixture_result(o)
            r['score_mode'] = study.SCORE
            m = dict(status='complete', checkpoints={k: {'sha256': k} for k in ('LeWM', 'Eff', 'EffPlan')},
                     protocol_overrides={'adaptive_rolling': {'efficiency_threshold': 0.8}})
            if local_limit is not None:
                r['score_mode'] = study.DISTANCE_SCORE
                m['protocol_overrides']['adaptive_rolling']['local_distance_limit'] = local_limit
                for rr in records:
                    for record in rr:
                        record.update(criterion='local_distance_efficiency',
                                      local_distance_limit=local_limit)
                        for gate in record['split_attempts']:
                            gate.update(local_distance_limit=local_limit,
                                        stop_reason='distance_and_efficiency_sufficient')
            if distance_only:
                r['score_mode'] = study.DISTANCE_ONLY_SCORE
                m['protocol_overrides']['adaptive_rolling']['efficiency_threshold'] = None
                for rr in records:
                    for record in rr:
                        record.update(criterion='local_distance', efficiency_threshold=None)
                        for gate in record['split_attempts']:
                            gate.update(threshold=None, efficiency=None, predicted_work=None,
                                        stop_reason='distance_sufficient')
            paths = [output/v/f'O{o}', root/'sparse_v_compare_20260914'/v/'eval/EffPlan'/f'O{o}',
                     root/'adaptive_rolling_20260915'/v/f'O{o}']
            for p in paths:
                p.mkdir(parents=True)
                (p/'result.json').write_text(json.dumps(r))
                (p/'protocol_manifest.json').write_text(json.dumps(m))
            p = paths[0]
            (p/'adaptive_planning.json').write_text(json.dumps({'episodes': records}))
            (p/'episode_results.json').write_text(json.dumps({'episodes': r['episode_results']}))
            plan = dict(initial_nodes=torch.zeros(1,2,192), final_nodes=torch.zeros(1,2,192),
                        actions=torch.zeros(1,1,25))
            torch.save([[plan for _ in rr] for rr in records], p/'adaptive_plans.pt')
    report = study.analyze_study(runs_root=root, output_root=output)
    assert len(report) == 6
    assert (output/'study_analysis.json').exists()
    text = (output/'study_analysis.md').read_text()
    assert 'Original V O100 | Extra-work V O25' in text
    name = 'adaptive_efficiency_080' if local_limit is None else 'adaptive_distance_efficiency_080'
    if distance_only:
        name = 'adaptive_distance_only'
    assert 'floor-driven stops' in text and name in text


def test_distance_study_passes_explicit_scale_to_all_jobs(tmp_path):
    matrix = study.build_jobs(repo=tmp_path/'repo', runs_root=tmp_path/'historical',
                              output_root=tmp_path/'new', dataset=tmp_path/'data',
                              lewm_checkpoint=tmp_path/'f', python='python',
                              devices=['0'], local_distance_limit=2.5)
    for job in matrix:
        command = job['command']
        assert command[command.index('--adaptive-local-distance-limit')+1] == '2.5'


def test_distance_audit_rejects_high_efficiency_but_long_stop():
    result, records = fixture_result()
    for rr in records:
        for r in rr:
            r.update(criterion='local_distance_efficiency', local_distance_limit=10.)
            for g in r['split_attempts']:
                g.update(local_distance_limit=10., stop_reason='distance_and_efficiency_sufficient')
    study.audit_records(result, records, offset=25, local_distance_limit=10.)
    bad = copy.deepcopy(records)
    bad[0][0]['split_attempts'][0].update(distance=11., predicted_work=11.,
                                         efficiency=11/(11+1e-6))
    with pytest.raises(ValueError, match='distance/efficiency'):
        study.audit_records(result, bad, offset=25, local_distance_limit=10.)


def test_distance_only_launcher_has_six_isolated_jobs_without_efficiency_flag(tmp_path):
    matrix = study.build_jobs(repo=tmp_path/'repo', runs_root=tmp_path/'history',
                              output_root=tmp_path/'new', dataset=tmp_path/'data',
                              lewm_checkpoint=tmp_path/'f', python='python',
                              devices=['0','1'], local_distance_limit=10., distance_only=True)
    assert len(matrix) == 6 and len({j['output'] for j in matrix}) == 6
    for j in matrix:
        assert '--adaptive-distance-only' in j['command']
        assert '--adaptive-efficiency-threshold' not in j['command']
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('fault', ['none', 'long_stop', 'short_split', 'eta', 'degenerate'])
def test_distance_only_auditor_checks_boundaries(fault):
    result, records = fixture_result()
    for rr in records:
        for r in rr:
            r.update(criterion='local_distance', efficiency_threshold=None, local_distance_limit=10.)
            for g in r['split_attempts']:
                g.update(threshold=None, efficiency=None, predicted_work=None,
                         local_distance_limit=10., stop_reason='distance_sufficient')
    g=records[0][0]['split_attempts'][0]
    if fault == 'long_stop': g['distance']=11.
    if fault == 'short_split': g.update(accepted=True,stop_reason=None)
    if fault == 'eta': g['efficiency']=.9
    if fault == 'degenerate': g['distance']=0.
    if fault == 'none':
        assert study.audit_records(result,records,offset=25,local_distance_limit=10.,
                                   distance_only=True)['geometric_floor_stop_count'] == 0
    else:
        with pytest.raises(ValueError):
            study.audit_records(result,records,offset=25,local_distance_limit=10.,distance_only=True)


def test_run_refuses_cpu_only_without_creating_outputs(tmp_path, monkeypatch):
    import runpy
    import sys
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(sys, 'argv', ['run_effplan_efficiency_study.py', 'run',
        '--runs-root', str(tmp_path/'history'), '--output-root', str(tmp_path/'new'),
        '--dataset', 'unused', '--lewm-checkpoint', 'unused', '--devices', '0',
        '--local-distance-limit', '10', '--distance-only'])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/run_effplan_efficiency_study.py'),
                       run_name='__main__')
    assert e.value.code == 2
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('fail', [False, True])
def test_explicit_cpu_runs_serially_and_stops_on_failure(tmp_path, monkeypatch, fail):
    matrix = study.build_jobs(repo=tmp_path/'repo', runs_root=tmp_path/'history',
        output_root=tmp_path/'cpu', dataset=tmp_path/'data', lewm_checkpoint=tmp_path/'f',
        python='python', devices=None, local_distance_limit=10., distance_only=True,
        execution_device='cpu')
    for j in matrix:
        assert j['command'][j['command'].index('--device')+1] == 'cpu'
        assert j['gpu'] == ''
        for name in j['inputs'].values():
            p = Path(name); p.parent.mkdir(parents=True,exist_ok=True); p.touch()
    started, waited = [], []
    class Child:
        def __init__(self, command, **kwargs):
            assert len(started) == len(waited), 'CPU jobs unexpectedly overlap'
            started.append(command)
            self.pid = len(started)
        def wait(self):
            waited.append(self.pid)
            return 1 if fail else 0
    monkeypatch.setattr(study.subprocess,'Popen',Child)
    if fail:
        with pytest.raises(RuntimeError,match='CPU evaluation failed'):
            study.run_jobs(matrix,repo=tmp_path/'repo',output_root=tmp_path/'cpu')
        assert len(started) == 1
    else:
        m=study.run_jobs(matrix,repo=tmp_path/'repo',output_root=tmp_path/'cpu')
        assert m['status']=='evaluated' and m['execution_device']=='cpu'
        assert m['max_parallel_jobs']==1 and len(started)==6 and len(waited)==6
