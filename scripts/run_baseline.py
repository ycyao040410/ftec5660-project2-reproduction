"""Run the author's baseline twice and verify the selected results."""

import subprocess
import sys
from tempfile import TemporaryDirectory

from common import DATASET, UPSTREAM_COMMIT, arguments, fit_and_score, load_split, write_json

PREFIXES = ("ftec_recheck_1", "ftec_recheck_2")
MONITORS = ("esn_cusum_max", "delta_mahalanobis")
TABLES = {
    "benchmark": (
        ["monitor"],
        ["monitor", "auroc", "auprc", "detection_rate", "healthy_fa_rate",
         "early_alarm_rate", "median_delay", "mean_lead_all", "footprint_mb"],
        2,
    ),
    "per_class": (
        ["monitor", "failure_class"],
        ["monitor", "failure_class", "detection_rate", "median_delay", "mean_lead_all"],
        16,
    ),
    "diagnosis": (
        ["episode_id"],
        ["episode_id", "failure_class", "T", "tau", "horizon", "det_esn", "det_maha"],
        171,
    ),
    "length_confound": (
        ["monitor"],
        ["monitor", "healthy_len_score_spearman", "raw_auroc", "length_matched_auroc",
         "n_healthy", "n_injected", "n_matched"],
        2,
    ),
}


def selected_table(path, suffix):
    import pandas as pd

    keys, columns, expected_rows = TABLES[suffix]
    rows = pd.read_csv(path)
    rows = rows[rows["dataset"].eq(DATASET)]
    if "monitor" in rows:
        rows = rows[rows["monitor"].isin(MONITORS)]
    result = rows[columns].sort_values(keys).reset_index(drop=True)
    if len(result) != expected_rows or result.duplicated(keys).any():
        raise ValueError(f"{path.name}: unexpected or duplicate rows.")
    return result


def verify_scores(benchmark, diagnosis):
    import numpy as np
    import pandas as pd
    from derail.monitor.hybrid import make_hybrids

    train, validation, test, channels, standardizer = load_split()
    reference = benchmark.set_index("monitor")
    flags = diagnosis.set_index("episode_id")
    audit_rows = []
    for model, column in zip(make_hybrids(standardizer, channels=channels, seed=1300)[:2],
                             ("det_esn", "det_maha"), strict=True):
        threshold, scores, alarms, val_count = fit_and_score(model, train, validation, test)
        positives, negatives = [], []
        for episode in test:
            sequence = scores[episode.episode_id]
            crossings = np.flatnonzero(sequence > threshold)
            alarm = int(crossings[0]) if crossings.size else None
            if episode.is_healthy:
                outcome = "false_alarm" if alarm is not None else "correct_silence"
                negatives.append(float(sequence.max()))
            else:
                if alarm is None:
                    outcome = "miss"
                else:
                    outcome = "early_alarm" if alarm < episode.tau else "true_alarm"
                if (outcome == "true_alarm") != bool(flags.at[episode.episode_id, column]):
                    raise ValueError(f"Detection flag mismatch for {episode.episode_id}.")
                positives.append(float(sequence.max()))
            observed_step = alarms.at[episode.episode_id, "alarm_step"]
            if pd.isna(observed_step):
                observed_step = None
            if (outcome, alarm) != (alarms.at[episode.episode_id, "outcome"], observed_step):
                raise ValueError(f"Independent first-alarm check failed for {episode.episode_id}.")
            audit_rows.append({
                "monitor": model.name, "episode_id": episode.episode_id,
                "outcome": outcome, "first_alarm_step": alarm, "threshold": threshold,
            })
        pairwise = np.asarray(positives)[:, None] - np.asarray(negatives)[None, :]
        auc = float(((pairwise > 0).sum() + 0.5 * (pairwise == 0).sum()) / pairwise.size)
        counts = alarms["outcome"].value_counts()
        metrics = {"auroc": auc,
                   "detection_rate": counts.get("true_alarm", 0) / 171,
                   "healthy_fa_rate": counts.get("false_alarm", 0) / 24,
                   "early_alarm_rate": counts.get("early_alarm", 0) / 171}
        for metric, value in metrics.items():
            np.testing.assert_allclose(value, reference.at[model.name, metric],
                                       rtol=1e-10, atol=1e-12)
        print(f"{model.name}: AUROC={auc:.4f}, validation alarms={val_count}/24")
    return pd.DataFrame(audit_rows)


def main():
    upstream, output = arguments(__doc__, "baseline")
    from pandas.testing import assert_frame_equal

    references = {suffix: selected_table(upstream / "results" / "tables" /
                                         f"hybrid_{suffix}.csv", suffix)
                  for suffix in TABLES}
    with TemporaryDirectory(prefix="ftec5660-baseline-") as temporary:
        from pathlib import Path

        stage = Path(temporary)
        verified = {}
        for prefix in PREFIXES:
            subprocess.run(
                [sys.executable, "-m", "derail.experiments.run_hybrid_study",
                 "--datasets", DATASET, "--seed", "0", "--out-prefix", str(stage / prefix)],
                cwd=upstream, check=True,
            )
            for suffix, expected in references.items():
                filename = f"{prefix}_{suffix}.csv"
                actual = selected_table(stage / filename, suffix)
                assert_frame_equal(actual, expected, check_dtype=False, check_exact=False,
                                   rtol=1e-10, atol=1e-12, obj=filename)
                verified[filename] = actual
        audit = verify_scores(verified[f"{PREFIXES[0]}_benchmark.csv"],
                              verified[f"{PREFIXES[0]}_diagnosis.csv"])
        for filename, rows in verified.items():
            rows.to_csv(output / filename, index=False)
        audit.to_csv(output / "episode_alarm_audit.csv", index=False)
    write_json(output / "verification.json", {
        "status": "passed",
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset": DATASET,
        "trace_checksums_verified": 291,
        "run_prefixes": list(PREFIXES),
        "rows_per_run": {key: value[2] for key, value in TABLES.items()},
        "independently_checked_alarm_rows": len(audit),
    })
    print(f"Verified both baseline runs. Output: {output}")


if __name__ == "__main__":
    main()
