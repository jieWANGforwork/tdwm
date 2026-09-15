"""The explicit efficiency option must not silently modify another score mode."""

import runpy
import sys
from pathlib import Path

import pytest

from tdwm.evaluation import effplan as module


def arguments():
    args = ['evaluate_effplan.py', 'evaluate', '--method', 'EffPlan']
    for name in ['config', 'dataset', 'lewm-checkpoint', 'selection', 'output-dir', 'device']:
        args.extend(['--'+name, 'unused'])
    return args


def test_cli_forwards_explicit_threshold_without_selecting_one(monkeypatch):
    seen = {}
    monkeypatch.setattr(module, 'evaluate_effplan', lambda **kw: seen.update(kw) or {})
    monkeypatch.setattr(sys, 'argv', arguments()+['--adaptive-rolling', '--adaptive-efficiency-threshold', '0.8'])
    script = Path(__file__).resolve().parents[2]/'scripts/evaluate_effplan.py'
    runpy.run_path(str(script), run_name='__main__')
    assert seen['adaptive_rolling'] and seen['adaptive_efficiency_threshold'] == 0.8
    monkeypatch.setattr(sys, 'argv', arguments()+['--adaptive-rolling'])
    runpy.run_path(str(script), run_name='__main__')
    assert seen['adaptive_efficiency_threshold'] is None


@pytest.mark.parametrize('flags', [[], ['--adaptive-one-shot'], ['--offset-window'],
                                  ['--adaptive-rolling', '--adaptive-one-shot']])
def test_cli_rejects_incompatible_protocol_before_any_artifact_load(monkeypatch, flags):
    monkeypatch.setattr(sys, 'argv', arguments()+flags+['--adaptive-efficiency-threshold', '0.8'])
    script = Path(__file__).resolve().parents[2]/'scripts/evaluate_effplan.py'
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name='__main__')
    assert exc.value.code == 2


@pytest.mark.parametrize('kwargs', [
    {}, {'method': 'Eff', 'adaptive_rolling': True},
    {'adaptive_rolling': True, 'offset_window': True},
    {'adaptive_one_shot': True},
    {'adaptive_rolling': True, 'adaptive_efficiency_threshold': float('nan')},
    {'adaptive_rolling': True, 'adaptive_efficiency_threshold': 1},
])
def test_api_rejects_invalid_efficiency_mode_before_data_load(monkeypatch, kwargs):
    monkeypatch.setattr(module, 'load_eff_protocol', lambda *a, **kw: {})
    args = dict(config_path='unused', dataset_path='unused', lewm_checkpoint='unused',
                selection_path='unused', output_dir='unused', device='cpu',
                method='EffPlan', adaptive_efficiency_threshold=0.8)
    args.update(kwargs)
    with pytest.raises(ValueError):
        module.evaluate_effplan(**args)
