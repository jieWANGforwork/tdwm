"""Preview or run the six predeclared tau=0.8 jobs, or audit their completed results."""

import argparse
import json
import sys
from pathlib import Path

from tdwm.evaluation.effplan_efficiency_study import build_jobs, run_jobs, analyze_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preview', 'run', 'analyze'])
    parser.add_argument('--runs-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--lewm-checkpoint')
    parser.add_argument('--devices', nargs='+')
    parser.add_argument('--execution-device', choices=['cuda', 'cpu'], default='cuda',
                        help="Explicit CPU fallback runs one full evaluation at a time; CUDA stays default.")
    parser.add_argument('--distance-only', action='store_true',
                        help="Separate six-job distance-only study; no efficiency threshold.")
    parser.add_argument('--local-distance-limit', type=float,
                        help="Required for new preview/run: training-calibrated local latent distance.")
    args = parser.parse_args()
    if args.mode == 'analyze':
        result = analyze_study(runs_root=args.runs_root, output_root=args.output_root)
    else:
        if not args.dataset or not args.lewm_checkpoint or (args.execution_device == 'cuda' and not args.devices):
            parser.error('preview/run require --dataset, --lewm-checkpoint and --devices')
        if args.local_distance_limit is None:
            parser.error('preview/run require an explicit --local-distance-limit; no scale is guessed')
        repo = Path(__file__).resolve().parents[1]
        jobs = build_jobs(repo=repo, runs_root=args.runs_root, output_root=args.output_root,
                          dataset=args.dataset, lewm_checkpoint=args.lewm_checkpoint,
                          python=sys.executable, devices=args.devices,
                          local_distance_limit=args.local_distance_limit,
                          distance_only=args.distance_only, execution_device=args.execution_device)
        if args.mode == 'run':
            import torch
            if args.execution_device == 'cuda' and (
                not torch.cuda.is_available() or any(int(d) >= torch.cuda.device_count() for d in args.devices)
            ):
                parser.error('Requested CUDA GPUs are unavailable; no formal jobs launched. Use preview in CPU-only mode.')
            run_jobs(jobs, repo=repo, output_root=args.output_root)
            result = analyze_study(runs_root=args.runs_root, output_root=args.output_root)
        else:
            result = jobs
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
