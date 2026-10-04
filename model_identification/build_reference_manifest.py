#!/usr/bin/env python3
"""Build the exact per-seed/fold targets used by journal_results.ipynb.

This is a provenance helper.  Normal checkpoint identification uses the
checked-in ``paper_result_targets.json`` and does not need the archived CSV
tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path


CCC_GROUPS = (
    "Mean Activation CCC_ind",
    "Activation CCC",
    "Mean Valence CCC_ind",
    "Valence CCC",
)
CONSENSUS_CCC_GROUPS = ("Activation CCC", "Valence CCC")
PROBABILITY_GROUPS = (
    "Argmax Activation UAR",
    "Argmax Valence UAR",
    "Total Variation Distance",
    "Jensen-Shannon Divergence",
)

BASE_TARGETS = {
    "baseline": {"Test full-dataset": ("II", "VI")},
    "baseline_aggregate": {"Test full-dataset": ("II",)},
    "individual_annotator": {
        "Test full-dataset": ("II", "III", "IV", "VI"),
        "Test 100plus-annotations": ("II", "III", "IV"),
        "Test 1000plus-annotations": ("II", "III", "IV"),
    },
    "individual_annotator_plus": {
        "Test full-dataset": ("II", "VI"),
        "Test 100plus-annotations": ("II",),
        "Test 1000plus-annotations": ("II",),
    },
    "individual_annotator_var": {
        "Test full-dataset": ("II", "VI"),
        "Test 100plus-annotations": ("II",),
        "Test 1000plus-annotations": ("II",),
    },
    "individual_annotator_plus_var": {
        "Test full-dataset": ("II", "VI"),
        "Test 100plus-annotations": ("II",),
        "Test 1000plus-annotations": ("II",),
    },
    "individual_annotator_100plus": {
        "Test full-dataset similarity": ("III",),
        "Test 100plus-annotations": ("III",),
        "Test 1000plus-annotations": ("III",),
    },
    "individual_annotator_plus_100plus": {
        "Test full-dataset similarity": ("III",),
        "Test 100plus-annotations": ("III",),
        "Test 1000plus-annotations": ("III",),
    },
    "individual_annotator_var_100plus": {
        "Test full-dataset similarity": ("III",),
        "Test 100plus-annotations": ("III",),
        "Test 1000plus-annotations": ("III",),
    },
    "individual_annotator_plus_var_100plus": {
        "Test full-dataset similarity": ("III",),
        "Test 100plus-annotations": ("III",),
        "Test 1000plus-annotations": ("III",),
    },
    "individual_annotator_1000plus": {
        "Test full-dataset similarity": ("IV",),
        "Test 100plus-annotations similarity": ("IV",),
        "Test 1000plus-annotations": ("IV",),
    },
    "individual_annotator_plus_1000plus": {
        "Test full-dataset similarity": ("IV",),
        "Test 100plus-annotations similarity": ("IV",),
        "Test 1000plus-annotations": ("IV",),
    },
    "individual_annotator_var_1000plus": {
        "Test full-dataset similarity": ("IV",),
        "Test 100plus-annotations similarity": ("IV",),
        "Test 1000plus-annotations": ("IV",),
    },
    "individual_annotator_plus_var_1000plus": {
        "Test full-dataset similarity": ("IV",),
        "Test 100plus-annotations similarity": ("IV",),
        "Test 1000plus-annotations": ("IV",),
    },
    "baseline_podcast_1.8": {"Test full-dataset": ("V",)},
    "baseline_aggregate_podcast_1.8": {"Test full-dataset": ("V",)},
    "individual_annotator_podcast_1.8": {"Test full-dataset": ("V",)},
    "individual_annotator_plus_podcast_1.8": {"Test full-dataset": ("V",)},
    "individual_annotator_var_podcast_1.8": {"Test full-dataset": ("V",)},
    "individual_annotator_plus_var_podcast_1.8": {
        "Test full-dataset": ("V",)
    },
}

RUN_FILE_RE = re.compile(r"model_prob\d+_seed_(\d+)_fold_(\d+)\.csv$")


def newest_run(model_dir: Path) -> Path:
    runs = [path for path in model_dir.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No result runs in {model_dir}")

    def timestamp(path: Path) -> datetime:
        value = "_".join(path.name.split("_")[-4:])
        return datetime.strptime(value, "%d_%m_%Y_%H:%M:%S")

    return max(runs, key=timestamp)


def read_metrics(path: Path, log_type: str, groups: tuple[str, ...]) -> dict[str, float]:
    found: dict[str, list[float]] = {group: [] for group in groups}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            group = row["group"]
            if row["log_type"] == log_type and group in found:
                found[group].append(float(row["scalar_value"]))

    duplicated = {group: values for group, values in found.items() if len(values) != 1}
    if duplicated:
        counts = {group: len(values) for group, values in duplicated.items()}
        raise ValueError(f"Expected one value per metric in {path} ({log_type}): {counts}")
    return {group: values[0] for group, values in found.items()}


def target_entry(
    *,
    experiment: str,
    dataset: str,
    seed: int,
    fold: int,
    log_type: str,
    tables: tuple[str, ...],
    metrics: dict[str, float],
    source_csv: Path,
    source_root: Path,
) -> dict:
    return {
        "target_id": f"{experiment}|seed={seed}|fold={fold}|{log_type}",
        "experiment": experiment,
        "dataset": dataset,
        "seed": seed,
        "fold": fold,
        "log_type": log_type,
        "tables": list(tables),
        "metrics": metrics,
        "source_csv": str(source_csv.relative_to(source_root)),
    }


def base_targets(results_root: Path) -> list[dict]:
    targets = []
    base_root = results_root / "journal_experiments"
    probability_models = {
        "baseline",
        "individual_annotator",
        "individual_annotator_plus",
        "individual_annotator_var",
        "individual_annotator_plus_var",
    }

    for experiment, log_types in BASE_TARGETS.items():
        run_dir = newest_run(base_root / experiment)
        csv_paths = sorted(run_dir.glob("*.csv"))
        if len(csv_paths) != 5:
            raise ValueError(f"Expected five seeds in {run_dir}; found {len(csv_paths)}")
        for path in csv_paths:
            match = RUN_FILE_RE.match(path.name)
            if not match:
                raise ValueError(f"Unexpected result filename: {path}")
            seed, fold = map(int, match.groups())
            for log_type, tables in log_types.items():
                groups = CCC_GROUPS
                if "podcast_1.8" in experiment:
                    groups = CONSENSUS_CCC_GROUPS
                if experiment in probability_models and log_type == "Test full-dataset":
                    groups = groups + PROBABILITY_GROUPS
                targets.append(
                    target_entry(
                        experiment=experiment,
                        dataset="podcast-1.8" if "podcast_1.8" in experiment else "podcast-1.11",
                        seed=seed,
                        fold=fold,
                        log_type=log_type,
                        tables=tables,
                        metrics=read_metrics(path, log_type, groups),
                        source_csv=path,
                        source_root=results_root,
                    )
                )
    return targets


def fine_tune_spec(dataset: str) -> dict[str, dict[str, tuple[str, ...]]]:
    spec: dict[str, dict[str, tuple[str, ...]]] = {
        f"baseline_aggregate_ft_on_{dataset}": {
            "Test zero-shot": ("VII", "VIII"),
            "Test": ("VIII",),
        }
    }
    for base in (
        "individual_annotator",
        "individual_annotator_100plus",
        "individual_annotator_1000plus",
    ):
        for separate in (False, True):
            suffix = "_separate_act_val_mapping" if separate else ""
            experiment = f"{base}_ft_on_{dataset}_with_1_per_new_annotator{suffix}"
            spec[experiment] = {"Test zero-shot": ("VII",)}

    # Table VIII additionally compares random mapping for the all-annotator IA
    # model, and its fine-tuned rows use the one-to-two checkpoints.
    prefix = f"individual_annotator_ft_on_{dataset}_with_1_per_new_annotator"
    spec[prefix]["Test zero-shot"] = ("VII", "VIII")
    spec[f"{prefix}_random_map"] = {"Test zero-shot": ("VIII",)}
    spec[f"{prefix}_separate_act_val_mapping"]["Test"] = ("VIII",)
    spec[f"{prefix}_separate_act_val_mapping_random_map"] = {
        "Test": ("VIII",)
    }
    return spec


def fine_tune_targets(results_root: Path) -> list[dict]:
    targets = []
    for dataset in ("muse", "iemocap", "improv"):
        dataset_root = results_root / f"journal_experiments_ft_on_{dataset}"
        for experiment, log_types in fine_tune_spec(dataset).items():
            run_dir = newest_run(dataset_root / experiment)
            csv_paths = sorted(run_dir.glob("*.csv"))
            if len(csv_paths) != 25:
                raise ValueError(
                    f"Expected five seeds x five folds in {run_dir}; found {len(csv_paths)}"
                )
            for path in csv_paths:
                match = RUN_FILE_RE.match(path.name)
                if not match:
                    raise ValueError(f"Unexpected result filename: {path}")
                seed, fold = map(int, match.groups())
                for log_type, tables in log_types.items():
                    targets.append(
                        target_entry(
                            experiment=experiment,
                            dataset=dataset,
                            seed=seed,
                            fold=fold,
                            log_type=log_type,
                            tables=tables,
                            metrics=read_metrics(path, log_type, CCC_GROUPS),
                            source_csv=path,
                            source_root=results_root,
                        )
                    )
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_root",
        type=Path,
        help="Directory containing journal_experiments* result directories",
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    root = args.results_root.resolve()
    targets = base_targets(root) + fine_tune_targets(root)
    targets.sort(
        key=lambda item: (
            item["experiment"],
            item["seed"],
            item["fold"],
            item["log_type"],
        )
    )
    manifest = {
        "schema_version": 1,
        "description": (
            "Exact per-seed/fold scalar values read by journal_results.ipynb "
            "for paper Tables II--VIII."
        ),
        "comparison": {"default_atol": 1e-6, "rtol": 0.0},
        "targets": targets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(targets)} targets to {args.output}")


if __name__ == "__main__":
    main()
