# FTEC5660 Project 2 — Failure Detection on Short LLM Agent Trajectories

This project reproduces the Section 5 comparison of ESN-CUSUM and Δ-Mahalanobis on
`real_research7b` from Sunny Dubey's [Real-Time Detection and Repair of LLM Agent Failures](https://arxiv.org/abs/2608.02464v1).
The baseline runs the original study twice at experiment seed 0 and checks the
results against the reference files. The ablation compares CUSUM and EWMA across
five paired seeds.

The [upstream implementation and recorded traces](https://github.com/sunnydubey1111/agent-trajectory-sentinel)
are used at commit `00c0673adc54f74be84babbcb63f67e460f54e41` under its
upstream [MIT license](https://github.com/sunnydubey1111/agent-trajectory-sentinel/blob/00c0673adc54f74be84babbcb63f67e460f54e41/LICENSE).
The scripts in this repository orchestrate the experiment, check the benchmark
results, and export the ablation evidence.

## Setup

From this repository's root, use Git and CPython 3.14.5. The reference runs used
Debian 13 (x86-64), an AMD EPYC CPU, and the upstream core dependency lock.
The recorded trajectories can be evaluated on CPU after setup. Evaluation
requires no API key, model download, or new agent calls. Git and pip need
network access during setup.

```bash
python3.14 --version

git init upstream
git -C upstream remote add origin https://github.com/sunnydubey1111/agent-trajectory-sentinel.git
git -C upstream fetch --depth 1 origin 00c0673adc54f74be84babbcb63f67e460f54e41
git -C upstream checkout --detach FETCH_HEAD
git -C upstream rev-parse HEAD

python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r upstream/requirements-core.lock.txt
```

The setup commands create a fresh `upstream/` directory. For later runs, activate
the existing environment with `source .venv/bin/activate`. If the initial fetch
is interrupted after adding the remote, resume at the fetch command.

The runners verify the commit and the SHA-256 hashes of the 291 traces listed
in `traces/real_research7b/manifest.json`. The dataset contains 120 healthy
episodes and 171 fault-injected episodes. The original split uses 72 healthy
training episodes, 24 healthy validation episodes, and 195 test episodes
(24 healthy and 171 fault-injected). The split seed is fixed at 0.
Numerical library thread counts are set to one before importing NumPy.

## Run the baseline

```bash
python scripts/run_baseline.py
```

The runner calls `derail.experiments.run_hybrid_study` twice using
`--datasets real_research7b --seed 0` and the prefixes `ftec_recheck_1` and
`ftec_recheck_2`. It creates temporary files for these runs, so existing
results cannot be mistaken for fresh output. It compares the two detectors'
non-timing benchmark values, 16 rows by detector and fault class, 171
episode-level valid-detection flag pairs, and the length diagnostic with the
upstream reference tables. It also recomputes AUROC from positive–negative
score pairs and independently checks the first crossing and alarm outcome.
The selected output and verification status are written to `runs/baseline/`.
Fit time and per-step latency are excluded from the comparisons.

| Detector | Paper AUROC | Reproduced AUROC | Valid detections | Healthy test false alarms |
|---|---:|---:|---:|---:|
| ESN-CUSUM | 0.777 | 0.7766 | 64/171 | 2/24 |
| Δ-Mahalanobis | 0.848 | 0.8484 | 71/171 | 1/24 |

## Run the paired ablation

```bash
python scripts/run_ablation.py
```

The intervention changes the `cusum` setting of the upstream
`ChannelMaxESNMonitor` from `True` to `False`. This selects EWMA scoring
instead of CUSUM. Both variants share the split, fitted standardizer, four
channels, eight ESNs per channel, channel-max aggregation, and paired
initialization. The script checks the equality of the fitted reservoir,
input and readout weights and residual scales. Study seeds 0–4 correspond to
monitor seeds 1300–1304. Each variant calibrates its own threshold on the
same 24 healthy validation episodes with the same 5% target. Output is written
to `runs/ablation/`.

AUROC uses the maximum anomaly score of each episode. An alarm occurs at the
first step with a score strictly above the threshold. A fault-injected
episode counts as a valid detection if the first alarm occurs at or after
fault onset. For mean post-alarm steps, episodes without a valid detection
contribute zero. Variances use the five seed-level values with denominator
`n - 1`. Rate variances use the 0–1 scale.

| Metric | CUSUM mean | EWMA mean | Change |
|---|---:|---:|---:|
| AUROC | 0.7717 | 0.7849 | +0.0132 |
| Valid detection rate | 36.14% | 36.84% | +0.70 percentage points |
| Healthy test false-alarm rate | 6.67% | 6.67% | 0.00 percentage points |
| Mean post-alarm steps | 0.2199 | 0.2374 | +0.0175 steps |

There are 309 CUSUM and 315 EWMA valid detections across the 855 seed–episode
evaluations. Among the 308 pairs detected by both, EWMA alarms earlier in 15
and at the same step in 293.

## Files and report tables

`evidence/` contains the checked outputs from a Linux run. To generate new
results, run the two commands above and read the corresponding files in
`runs/`. The optional `--upstream` and `--output` arguments allow explicit
input and output paths.

| Report item | Evidence |
|---|---|
| Table 1 and data integrity | Pinned upstream manifest, split checks in `scripts/common.py`, and `baseline/verification.json` |
| Table 2 and baseline repeatability | `baseline/ftec_recheck_{1,2}_benchmark.csv` and `baseline/verification.json` |
| Tables 3–4, per-fault and horizon analysis | `baseline/ftec_recheck_1_per_class.csv`, `baseline/ftec_recheck_1_diagnosis.csv`, and `baseline/episode_alarm_audit.csv` |
| Table 5, length diagnostic | `baseline/ftec_recheck_1_length_confound.csv` |
| Tables 6–7, five-seed comparison | `ablation/seed_metrics.csv`, `ablation/metric_summary.csv`, and `ablation/paired_outcomes.json` |
| Table 8, timeout example | `ablation/timeout_example.csv` with the seed-2 thresholds in `ablation/seed_metrics.csv` |
| Per-episode ablation outcomes | `ablation/episode_alarms.csv` |

The experiment evaluates alarms on recorded trajectories. Testing repair
would require running the agent with a repair policy and observing its
subsequent actions.
