"""Read the selected evaluation frames over SSH; preserve dataset JPEG bytes.

Uses an existing authenticated SSH control socket. No credentials are stored.
Generated assets are limited to the 150 selected reference clips; no whole dataset
or RGB cache is copied. Remote access is read-only; archives are streamed, not kept.
"""

import argparse
import json
import os
import shlex
import subprocess
import tarfile
from pathlib import Path

from .data import historical_trials

REMOTE = r"""
import io,json,sys,tarfile
from collections import defaultdict
import lance,numpy as np
config=json.load(sys.stdin)
ds=lance.dataset(config['dataset'])
by_episode=defaultdict(list)
for trial in config['trials']:
    by_episode[trial['episode']].append(trial)
out=tarfile.open(fileobj=sys.stdout.buffer,mode='w|')
total_bytes=0
def add(name,body):
    global total_bytes
    item=tarfile.TarInfo(name);item.size=len(body)
    out.addfile(item,io.BytesIO(body));total_bytes+=len(body)
def npy(values):
    buffer=io.BytesIO();np.save(buffer,np.asarray(values,dtype=np.float32),allow_pickle=False)
    return buffer.getvalue()
for i,(episode,trials) in enumerate(sorted(by_episode.items())):
    steps=sorted({s for t in trials for s in range(t['start'],t['goal']+1)})
    rows=ds.take([episode*201+s for s in steps],columns=['episode_idx','step_idx','pixels','observation','action']).to_pylist()
    indexed={}
    for step,row in zip(steps,rows):
        assert row['episode_idx']==episode and row['step_idx']==step, 'Dataset row identity mismatch'
        assert row['pixels'][:2]==b'\xff\xd8', 'Expected JPEG pixels'
        add(f'frames/ep{episode:06d}/step{step:03d}.jpg',row.pop('pixels'))
        indexed[step]=row
    for trial in trials:
        selected=[indexed[s] for s in range(trial['start'],trial['goal']+1)]
        prefix=trial['asset_prefix']
        add(prefix+'/states.npy',npy([r['observation'] for r in selected]))
        add(prefix+'/actions.npy',npy([r['action'] for r in selected[:-1]]))
    add(f'complete/ep{episode:06d}.json',json.dumps({'episode':episode,'unique_frames':len(steps)}).encode())
    print(f'Reference images: {i+1}/{len(by_episode)} episodes, {total_bytes/1e6:.1f} MB',file=sys.stderr,flush=True)
out.close()
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--history-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["RESULT_STUDIO_HISTORY_CONFIG"] = args.history_config
    root = Path(args.output).expanduser().resolve()
    if root.exists():
        raise ValueError(
            "Output already exists; choose a new directory to avoid overwriting data"
        )
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for offset in (25, 50, 100):
        trials, errors = historical_trials(offset)
        if errors or len(trials) != 50:
            raise ValueError(f"Invalid source selection O{offset}: {errors}")
        for trial in trials:
            records.append(
                {
                    "episode": trial.episode,
                    "offset": offset,
                    "start": trial.start,
                    "goal": trial.goal,
                    "asset_prefix": f"o{offset}/ep{trial.episode:06d}_s{trial.start:03d}_g{trial.goal:03d}",
                    "reference": {},
                    "methods": {
                        name: {
                            "executed": {
                                "success": tracks["executed"].success,
                                "attributes": tracks["executed"].attributes,
                            }
                        }
                        for name, tracks in trial.methods.items()
                    },
                }
            )
    dataset = args.dataset
    manifest = {
        "schema_version": 1,
        "provenance": {
            "dataset": dataset,
            "image_format": "original JPEG100 payloads from the server dataset; not re-encoded",
            "selection": "v1-c f_only / f_plus_g O25/O50/O100, all 50 trials per offset",
        },
        "trials": records,
    }

    def save_manifest():
        temp = root / "manifest.pending.json"
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        temp.replace(root / "manifest.json")

    save_manifest()
    command = [
        "ssh",
        "-S",
        args.socket,
        "-p",
        args.port,
        args.host,
        args.remote_python,
        "-c",
        shlex.quote(REMOTE),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        process.stdin.write(
            json.dumps({"dataset": dataset, "trials": records}).encode()
        )
        process.stdin.close()
        count = 0
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                destination = (root / member.name).resolve()
                if not member.isfile() or not destination.is_relative_to(
                    root.resolve()
                ):
                    raise ValueError("Unexpected archive entry")
                body = archive.extractfile(member).read()
                if member.name.startswith("complete/"):
                    episode = json.loads(body)["episode"]
                    for trial in records:
                        if trial["episode"] == episode:
                            prefix = trial["asset_prefix"]
                            trial["reference"] = {
                                "frames": [
                                    f"frames/ep{episode:06d}/step{s:03d}.jpg"
                                    for s in range(trial["start"], trial["goal"] + 1)
                                ],
                                "states": prefix + "/states.npy",
                                "actions": prefix + "/actions.npy",
                            }
                    save_manifest()
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(body)
                    count += 1
        if process.wait() != 0:
            raise RuntimeError(
                "Remote extraction failed; completed clips remain available"
            )
    except BaseException:
        process.terminate()
        process.wait()
        raise
    print(f"Imported {count} files. Manifest: {root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
