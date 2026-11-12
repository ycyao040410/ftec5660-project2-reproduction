"""Shared setup for the published benchmark and the paired ablation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

DATASET = "real_research7b"
UPSTREAM_COMMIT = "00c0673adc54f74be84babbcb63f67e460f54e41"
ROOT = Path(__file__).resolve().parents[1]


def arguments(description, default_output):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--upstream", type=Path, default=ROOT / "upstream")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / default_output)
    args = parser.parse_args()
    upstream, output = args.upstream.resolve(), args.output.resolve()
    if not (upstream / "BASELINE_MANIFEST.json").is_file():
        parser.error(f"Missing the upstream checkout at {upstream}. See README.md.")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        parser.error(f"Expected upstream commit {UPSTREAM_COMMIT}, found {commit}.")
    code_state = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "derail", "requirements-core.lock.txt",
         "BASELINE_MANIFEST.json", f"traces/{DATASET}/manifest.json"],
        cwd=upstream,
    )
    if code_state.returncode:
        parser.error("The upstream code, dependency lock or manifest differs from HEAD.")

    dataset_dir = upstream / "traces" / DATASET
    traces = json.loads((dataset_dir / "manifest.json").read_text())
    hashes = json.loads((upstream / "BASELINE_MANIFEST.json").read_text())
    reference_hashes = hashes["sections"]["traces"]
    if len(traces) != 291:
        parser.error(f"Expected 291 trace entries, found {len(traces)}.")
    for trace in traces:
        path = dataset_dir / trace["file"]
        expected = reference_hashes.get(path.relative_to(upstream).as_posix())
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            parser.error(f"Trace checksum failed for {path.name}.")

    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(upstream))
    output.mkdir(parents=True, exist_ok=True)
    return upstream, output


def load_split():
    from derail.common import Standardizer
    from derail.experiments.run_hybrid_study import REAL_DATASETS, load_real

    splits, channels = load_real(REAL_DATASETS[DATASET])
    train, validation, test = (splits[name] for name in ("train", "val", "test"))
    counts = [(sum(ep.is_healthy for ep in part), sum(not ep.is_healthy for ep in part))
              for part in (train, validation, test)]
    ids = {ep.episode_id for part in (train, validation, test) for ep in part}
    if counts != [(72, 0), (24, 0), (24, 171)] or len(ids) != 291:
        raise ValueError(f"Unexpected split counts or duplicate episode IDs: {counts}")
    if tuple(channels) != ("e", "u", "m", "x"):
        raise ValueError(f"Unexpected channel configuration: {channels}")
    return train, validation, test, channels, Standardizer().fit(train)


def fit_and_score(model, train, validation, test):
    from derail.evaluation.metrics import evaluate_alarms, pick_threshold
    from derail.experiments.run_hybrid_study import FA_BUDGET

    model.fit(train)
    validation_scores = [model.score_episode(episode) for episode in validation]
    threshold = float(pick_threshold(validation_scores, fa_budget=FA_BUDGET))
    scores = {episode.episode_id: model.score_episode(episode) for episode in test}
    alarms = evaluate_alarms(test, scores, threshold).set_index("episode_id")
    validation_alarms = sum(float(sequence.max()) > threshold for sequence in validation_scores)
    return threshold, scores, alarms, validation_alarms


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
