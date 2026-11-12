"""Compare CUSUM and EWMA on the same held-out trajectories across five seeds."""

from collections import Counter
from statistics import mean, variance

from common import DATASET, UPSTREAM_COMMIT, arguments, fit_and_score, load_split, write_json

METRICS = ("auroc", "detection_rate", "healthy_fa_rate", "early_alarm_rate",
           "median_delay", "mean_lead_all")
EXAMPLE_ID = "real-timeout-007"


def assert_same_fit(left, right):
    import numpy as np

    for original, modified in zip(left.subs, right.subs, strict=True):
        for attribute in ("_W", "_Win", "_Wout", "_sigma_err"):
            np.testing.assert_array_equal(
                getattr(original, attribute), getattr(modified, attribute),
                err_msg=f"The paired fit differs in {attribute}.",
            )


def main():
    _, output = arguments(__doc__, "ablation")
    import pandas as pd
    from derail.evaluation.metrics import episode_auc, summarize
    from derail.monitor.esn import ChannelMaxESNMonitor

    train, validation, test, channels, standardizer = load_split()
    episode_index = {ep.episode_id: ep for ep in test}
    seed_rows, pairs, alarms_by_variant, example_rows = [], [], [], []
    for seed in range(5):
        monitor_seed = 1300 + seed
        scores_by_variant, alarms, thresholds, metrics = {}, {}, {}, {}
        models = {
            "CUSUM": ChannelMaxESNMonitor(
                standardizer, channels=channels, K=8, seed=monitor_seed, cusum=True),
            "EWMA": ChannelMaxESNMonitor(
                standardizer, channels=channels, K=8, seed=monitor_seed, cusum=False),
        }
        for variant, model in models.items():
            threshold, scores, episode_alarms, val_count = fit_and_score(
                model, train, validation, test)
            thresholds[variant] = threshold
            scores_by_variant[variant] = scores
            alarms[variant] = episode_alarms
            metric_values = summarize(episode_alarms)
            metric_values["auroc"] = episode_auc(test, scores)
            metrics[variant] = {
                "variant": variant, "study_seed": seed, "monitor_seed": monitor_seed,
                "threshold": threshold, "validation_false_alarms": val_count,
                "valid_detections": int(episode_alarms["outcome"].eq("true_alarm").sum()),
                "healthy_false_alarms": int(episode_alarms["outcome"].eq("false_alarm").sum()),
                **{key: metric_values[key] for key in METRICS},
            }
            seed_rows.append(metrics[variant])
            alarms_by_variant.append(episode_alarms.reset_index().assign(
                variant=variant, study_seed=seed, monitor_seed=monitor_seed, threshold=threshold))

        assert_same_fit(models["CUSUM"], models["EWMA"])
        cusum, ewma = (alarms[name] for name in ("CUSUM", "EWMA"))
        cusum_valid = set(cusum.index[cusum["outcome"].eq("true_alarm")])
        ewma_valid = set(ewma.index[ewma["outcome"].eq("true_alarm")])
        gains = sorted(ewma_valid - cusum_valid)
        losses = sorted(cusum_valid - ewma_valid)
        timing = Counter()
        for episode_id in cusum_valid & ewma_valid:
            shift = ewma.at[episode_id, "alarm_step"] - cusum.at[episode_id, "alarm_step"]
            timing["earlier" if shift < 0 else "later" if shift > 0 else "same"] += 1
        pairs.append({
            "study_seed": seed,
            "gains": [{"episode_id": eid, "fault_class": episode_index[eid].failure_class}
                      for eid in gains],
            "losses": [{"episode_id": eid, "fault_class": episode_index[eid].failure_class}
                       for eid in losses],
            "ewma_earlier": timing["earlier"],
            "same_step": timing["same"],
            "ewma_later": timing["later"],
            "same_healthy_false_alarm_ids": set(cusum.index[cusum["outcome"].eq("false_alarm")])
            == set(ewma.index[ewma["outcome"].eq("false_alarm")]),
        })
        if seed == 2:
            episode = episode_index[EXAMPLE_ID]
            for step in range(episode.T):
                example_rows.append({
                    "study_seed": seed, "episode_id": EXAMPLE_ID, "step": step,
                    "fault_onset": step == episode.tau,
                    "cusum_ratio": float(scores_by_variant["CUSUM"][EXAMPLE_ID][step]
                                         / thresholds["CUSUM"]),
                    "ewma_ratio": float(scores_by_variant["EWMA"][EXAMPLE_ID][step]
                                        / thresholds["EWMA"]),
                    "cusum_first_alarm": bool(step == cusum.at[EXAMPLE_ID, "alarm_step"]),
                    "ewma_first_alarm": bool(step == ewma.at[EXAMPLE_ID, "alarm_step"]),
                })
        print(f"Seed {seed}: AUROC {metrics['CUSUM']['auroc']:.4f} -> "
              f"{metrics['EWMA']['auroc']:.4f}, gains/losses {len(gains)}/{len(losses)}")

    summary = []
    for metric in METRICS:
        original, modified = ([row[metric] for row in seed_rows if row["variant"] == name]
                              for name in ("CUSUM", "EWMA"))
        changes = [right - left for left, right in zip(original, modified, strict=True)]
        summary.append({
            "metric": metric, "cusum_mean": mean(original), "cusum_sample_variance": variance(original),
            "ewma_mean": mean(modified), "ewma_sample_variance": variance(modified),
            "mean_change": mean(changes),
        })

    pd.DataFrame(seed_rows).to_csv(output / "seed_metrics.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "metric_summary.csv", index=False)
    pd.concat(alarms_by_variant, ignore_index=True).to_csv(output / "episode_alarms.csv", index=False)
    pd.DataFrame(example_rows).to_csv(output / "timeout_example.csv", index=False)
    write_json(output / "paired_outcomes.json", {
        "dataset": DATASET, "upstream_commit": UPSTREAM_COMMIT, "pairs": pairs,
    })
    print(f"Finished five paired runs. Output: {output}")


if __name__ == "__main__":
    main()
