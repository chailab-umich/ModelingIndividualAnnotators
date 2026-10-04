#!/usr/bin/env python3
"""Identify recovered checkpoints by reproducing the paper's exact results.

The parent process inventories checkpoints and schedules one long-lived worker
per GPU.  Workers evaluate candidates sequentially on their assigned GPU using
the same DatasetManager, ExperimentRunner, dataloaders, and metric functions as
the journal experiments.  Per-job JSON files make an interrupted run resumable.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "run_experiments" / "configs" / "journal_experiments.yaml"
DEFAULT_MANIFEST = Path(__file__).with_name("paper_result_targets.json")
SEED_FOLD_RE = re.compile(r"model_prob\d+_seed_(\d+)_fold_(\d+)")
RECOVERY_TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}_\d+$")
PUBLICATION_GROUPS = {
    "Mean Activation CCC_ind",
    "Activation CCC",
    "Mean Valence CCC_ind",
    "Valence CCC",
    "Argmax Activation UAR",
    "Argmax Valence UAR",
    "Total Variation Distance",
    "Jensen-Shannon Divergence",
}
PROBABILITY_TABLE_EXPERIMENTS = {
    "baseline",
    "individual_annotator",
    "individual_annotator_plus",
    "individual_annotator_var",
    "individual_annotator_plus_var",
}
EVALUATION_SCHEMA_VERSION = 3
CACHE_FEATURE_KEYS = (
    "audio_feature_type",
    "audio_feature_layer",
    "text_feature_type",
)
CACHE_REQUIRED_COLUMNS = {
    "FileName",
    "Audio",
    "Text",
    "AudioFeatures",
    "TextFeatures",
    "act",
    "val",
    "soft_act_labels",
    "soft_val_labels",
    "annotators",
    "Dataset",
    *(f"kde_2d_probability_generation_{index}" for index in range(5)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=PROJECT_ROOT / "potential_model_weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "model_identification" / "results",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        help=(
            "Publication-era dataset cache. If omitted, compatible caches in "
            "the configured path and sibling working copies are discovered."
        ),
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2",
        help="Comma-separated physical GPU IDs (default: 0,1,2)",
    )
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Optional publication experiment names to evaluate",
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N planned jobs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Evaluate byte-identical files separately instead of once with aliases",
    )
    parser.add_argument(
        "--skip-data-preflight",
        action="store_true",
        help="Do not warm/check required dataset caches before starting GPU workers",
    )
    return parser.parse_args()


def add_project_import_paths() -> None:
    for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "run_experiments"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    add_project_import_paths()
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    args = state.get("model_args")
    if not isinstance(args, dict):
        raise ValueError("checkpoint has no model_args dictionary")
    model_type = args.get("model_type")
    model_type_name = getattr(model_type, "name", str(model_type))
    annotators = args.get("annotators") or []
    metadata = {
        "model_type": model_type_name,
        "determinism": bool(args.get("determinism", False)),
        "determinism_type": str(args.get("determinism_type", "default")),
        "disable_kde": bool(args.get("disable_kde", False)),
        "annotator_count": len(annotators),
        "real_annotator_count": len(annotators) - int("aggregate-annotator" in annotators),
        "has_aggregate_annotator": "aggregate-annotator" in annotators,
        "prob_grid_size": int(args.get("prob_grid_size", 0)),
    }
    del state
    return metadata


def path_information(path: Path) -> dict[str, Any]:
    match = SEED_FOLD_RE.search(path.as_posix())
    if not match:
        raise ValueError("path does not contain model_prob*_seed_*_fold_*")
    seed, fold = map(int, match.groups())
    parts = path.parts
    group_index = next(
        (
            index
            for index, part in enumerate(parts)
            if part == "journal_experiments" or part.startswith("journal_experiments_ft_on_")
        ),
        None,
    )
    if group_index is None or group_index + 1 >= len(parts):
        group = None
        hint = path.parent.parent.name
    else:
        group = parts[group_index]
        hint = parts[group_index + 1]
    hint = RECOVERY_TIMESTAMP_RE.sub("", hint)
    if hint.startswith("benchmarking_"):
        hint = hint.removeprefix("benchmarking_")
    hint = re.sub(r"_IAES(?:_.*)?$", "", hint)
    dataset = None
    if group and group.startswith("journal_experiments_ft_on_"):
        dataset = group.removeprefix("journal_experiments_ft_on_")
    return {
        "seed": seed,
        "fold": fold,
        "source_group": group,
        "experiment_hint": hint,
        "finetune_dataset": dataset,
    }


def route_base_experiment(metadata: dict[str, Any], path_info: dict[str, Any]) -> str | None:
    model_type = metadata["model_type"]
    is_v18 = "podcast_1.8" in path_info["experiment_hint"]
    podcast_suffix = "_podcast_1.8" if is_v18 else ""
    if model_type == "KDE_2D":
        return f"baseline{podcast_suffix}"
    if model_type == "BASELINE_AGGREGATE":
        return f"baseline_aggregate{podcast_suffix}"
    if model_type not in {"INDIVIDUAL_ANNOTATOR", "INDIVIDUAL_ANNOTATOR_PLUS"}:
        return None

    count = metadata["real_annotator_count"]
    if count <= 0:
        return None
    if count < 100:
        threshold_suffix = "_1000plus"
    elif count < 1000:
        threshold_suffix = "_100plus"
    else:
        threshold_suffix = ""

    base = (
        "individual_annotator_plus"
        if model_type == "INDIVIDUAL_ANNOTATOR_PLUS"
        else "individual_annotator"
    )
    if metadata["determinism"]:
        base += "_var"
    if is_v18:
        if threshold_suffix:
            return None
        return f"{base}_podcast_1.8"
    return f"{base}{threshold_suffix}"


def route_experiment(metadata: dict[str, Any], path_info: dict[str, Any]) -> str | None:
    dataset = path_info["finetune_dataset"]
    if dataset:
        hint = path_info["experiment_hint"]
        return hint if f"_ft_on_{dataset}" in hint else None
    return route_base_experiment(metadata, path_info)


def load_manifest(path: Path) -> tuple[dict[str, Any], dict[tuple[str, int, int], list[dict]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported target manifest schema: {manifest.get('schema_version')}")
    index: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for target in manifest["targets"]:
        index[(target["experiment"], target["seed"], target["fold"])].append(target)
    return manifest, index


def _required_cache_datasets(
    inventory: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    evaluable = [item for item in inventory if item["status"] == "evaluable"]
    podcast_versions = {
        "1.8" if "podcast_1.8" in item["routed_experiment"] else "1.11"
        for item in evaluable
        if not item["finetune_dataset"]
    }
    finetune_datasets = {
        item["finetune_dataset"]
        for item in evaluable
        if item["finetune_dataset"]
    }
    return podcast_versions, finetune_datasets


def _split_paths(cache_dir: Path, stem: str) -> list[Path]:
    return [cache_dir / f"{stem}_{split}.parquet" for split in ("train", "dev", "test")]


def validate_dataset_cache(
    cache_dir: Path,
    inventory: list[dict[str, Any]],
    data_config: dict[str, Any],
) -> tuple[list[str], list[Path]]:
    """Return incompatibilities and the parquet files used by this run."""
    import yaml

    problems: list[str] = []
    required_files: list[Path] = []
    if not cache_dir.is_dir():
        return ["directory does not exist"], []

    metadata_path = cache_dir / "config_used_for_cache.yaml"
    if not metadata_path.is_file():
        return ["missing config_used_for_cache.yaml"], []
    try:
        cache_config = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    except Exception as error:
        return [f"cannot read cache metadata: {type(error).__name__}: {error}"], []

    for key in CACHE_FEATURE_KEYS:
        expected = data_config.get(key)
        cached = cache_config.get(key)
        if cached != expected:
            problems.append(f"{key} is {cached!r}, expected {expected!r}")

    podcast_versions, finetune_datasets = _required_cache_datasets(inventory)
    if "1.11" in podcast_versions:
        explicit = _split_paths(cache_dir, "podcastv1.11")
        historical = _split_paths(cache_dir, "podcast")
        if all(path.is_file() for path in explicit):
            required_files.extend(explicit)
        elif all(path.is_file() for path in historical):
            cached_source = str(cache_config.get("podcast_directory", ""))
            if "1.11" not in cached_source:
                problems.append(
                    "unsuffixed podcast cache is not identified as MSP-Podcast 1.11"
                )
            required_files.extend(historical)
        else:
            problems.append(
                "missing complete MSP-Podcast 1.11 train/dev/test parquet set"
            )

    if "1.8" in podcast_versions:
        version_files = _split_paths(cache_dir, "podcastv1.8")
        if all(path.is_file() for path in version_files):
            required_files.extend(version_files)
        else:
            problems.append(
                "missing complete MSP-Podcast 1.8 train/dev/test parquet set"
            )

    for dataset in sorted(finetune_datasets):
        dataset_files = _split_paths(cache_dir, dataset)
        if all(path.is_file() for path in dataset_files):
            required_files.extend(dataset_files)
        else:
            problems.append(f"missing complete {dataset} train/dev/test parquet set")

    required_files = sorted(set(required_files))
    if required_files:
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            problems.append(f"cannot validate parquet schemas: {error}")
        else:
            for path in required_files:
                try:
                    columns = set(parquet.read_schema(path).names)
                except Exception as error:
                    problems.append(
                        f"cannot read {path.name}: {type(error).__name__}: {error}"
                    )
                    continue
                missing = sorted(CACHE_REQUIRED_COLUMNS - columns)
                if missing:
                    problems.append(
                        f"{path.name} is missing required columns: {', '.join(missing)}"
                    )
    return problems, required_files


def configure_dataset_cache(
    explicit_cache: Path | None,
    inventory: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Select a feature-compatible historical cache and mount it read-only."""
    import yaml

    data_config_path = PROJECT_ROOT / "data_config.yaml"
    data_config = yaml.safe_load(data_config_path.read_text(encoding="utf-8")) or {}
    candidates: list[Path] = []
    if explicit_cache is not None:
        candidates.append(explicit_cache.expanduser().resolve())
    else:
        configured = data_config.get("cache_dataset_path")
        if configured:
            candidates.append(Path(configured).expanduser().resolve())
        candidates.append(PROJECT_ROOT / "run_ICASSP26_experiments" / "dataset_caches")
        candidates.extend(
            sorted(PROJECT_ROOT.parent.glob("*/run_ICASSP26_experiments/dataset_caches"))
        )

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)

    checked: list[tuple[Path, list[str]]] = []
    selected: Path | None = None
    required_files: list[Path] = []
    for candidate in unique_candidates:
        problems, files = validate_dataset_cache(candidate, inventory, data_config)
        checked.append((candidate, problems))
        if not problems:
            selected = candidate
            required_files = files
            break
    if selected is None:
        details = "\n".join(
            f"  - {path}: {'; '.join(problems) or 'not selected'}"
            for path, problems in checked
        )
        raise RuntimeError(
            "No compatible publication dataset cache was found. "
            "Pass --dataset-cache with a cache containing the required BERT/WavLM "
            f"features and five KDE generations. Checked:\n{details}"
        )

    metadata_path = selected / "config_used_for_cache.yaml"
    identity = {
        "path": str(selected),
        "metadata_sha256": file_digest(metadata_path),
        "parquets": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in required_files
        ],
    }
    identity["signature"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (output_dir / "dataset_cache.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )
    os.environ["MIA_DATA_CONFIG_PATH"] = str(data_config_path)
    os.environ["MIA_DATASET_CACHE_PATH"] = str(selected)
    os.environ["MIA_DATASET_CACHE_READ_ONLY"] = "1"
    # The validated publication cache is the source of truth for both the
    # historical MSP-Podcast releases and the fine-tuning annotations.  This
    # avoids reconstructing labels from later dataset releases (and avoids a
    # dependency on private annotator metadata such as muse_annotators.csv).
    _, finetune_datasets = _required_cache_datasets(inventory)
    trusted_datasets = {"podcast", *finetune_datasets}
    os.environ["MIA_DATASET_CACHE_TRUSTED_DATASETS"] = ",".join(
        sorted(trusted_datasets)
    )
    print(f"Using validated read-only dataset cache: {selected}", flush=True)
    return identity


def inventory_candidates(
    weights_dir: Path,
    target_index: dict[tuple[str, int, int], list[dict]],
    experiments: set[str] | None,
) -> list[dict[str, Any]]:
    candidates = []
    paths = sorted(path for path in weights_dir.rglob("*.ckpt") if path.is_file())
    for number, path in enumerate(paths, start=1):
        relative = str(path.relative_to(weights_dir))
        record: dict[str, Any] = {
            "candidate": str(path.resolve()),
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "status": "planning",
        }
        try:
            info = path_information(path)
            metadata = checkpoint_metadata(path)
            record.update(info)
            record.update(metadata)
            record["sha256"] = file_digest(path)
            experiment = route_experiment(metadata, info)
            record["routed_experiment"] = experiment
            if experiment is None:
                record["status"] = "unsupported_checkpoint_type"
            elif experiments is not None and experiment not in experiments:
                record["status"] = "excluded_by_experiment_filter"
            elif (experiment, info["seed"], info["fold"]) not in target_index:
                record["status"] = "no_paper_target"
            else:
                record["status"] = "evaluable"
        except Exception as error:
            record["status"] = "metadata_error"
            record["error"] = f"{type(error).__name__}: {error}"
        candidates.append(record)
        if number % 25 == 0 or number == len(paths):
            print(f"Inventoried {number}/{len(paths)} checkpoints", flush=True)
    return candidates


def build_jobs(
    candidates: list[dict[str, Any]],
    no_deduplicate: bool,
    evaluation_signature: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate["status"] != "evaluable":
            continue
        dedupe = candidate["relative_path"] if no_deduplicate else candidate["sha256"]
        key = (
            dedupe,
            candidate["routed_experiment"],
            candidate["seed"],
            candidate["fold"],
        )
        groups[key].append(candidate)

    jobs = []
    for key, aliases in sorted(groups.items(), key=lambda item: item[1][0]["relative_path"]):
        representative = aliases[0]
        identity = "|".join((*map(str, key), evaluation_signature))
        job_id = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        jobs.append(
            {
                "job_id": job_id,
                "candidate": representative["candidate"],
                "relative_path": representative["relative_path"],
                "aliases": [item["candidate"] for item in aliases],
                "alias_relative_paths": [item["relative_path"] for item in aliases],
                "sha256": representative["sha256"],
                "experiment": representative["routed_experiment"],
                "seed": representative["seed"],
                "fold": representative["fold"],
                "finetune_dataset": representative["finetune_dataset"],
                "metadata": {
                    key: representative[key]
                    for key in (
                        "model_type",
                        "determinism",
                        "determinism_type",
                        "disable_kde",
                        "annotator_count",
                        "real_annotator_count",
                        "has_aggregate_annotator",
                        "prob_grid_size",
                    )
                },
                "evaluation_signature": evaluation_signature,
            }
        )
    return jobs


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fine_tune_config_for(experiment: str):
    from MIA.config import FineTuningConfig

    is_baseline = experiment.startswith("baseline_aggregate_")
    config = FineTuningConfig.create(
        None if is_baseline else 1,
        map_separately="_separate_act_val_mapping" in experiment,
        random="_random_map" in experiment,
        samples="all",
        epochs=None,
    )
    config.map_settings = None
    return config


def create_job_config(job: dict[str, Any], config_path: Path, work_dir: Path):
    add_project_import_paths()
    import yaml
    from MIA.config import ExperimentType
    from experiment_runner import create_experiment_config

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["output_path"] = str(work_dir)
    config["training"] = copy.deepcopy(config["training"])
    config["training"]["test_only"] = True
    experiment = job["experiment"]
    finetune_config = None
    base_experiment = experiment
    if job["finetune_dataset"]:
        base_experiment = experiment.split("_ft_on_", 1)[0]
        config["training"]["finetune_on"] = job["finetune_dataset"]
        finetune_config = fine_tune_config_for(experiment)
    else:
        config["training"]["finetune_on"] = None

    if base_experiment not in config["experiments"]:
        raise KeyError(f"No YAML configuration for {base_experiment}")
    experiment_definition = copy.deepcopy(config["experiments"][base_experiment])
    # The current config builder defaults disable_kde to True, while the
    # publication runs behind Table VI evaluated the probability outputs.  The
    # flag is evaluation-only (load_best_model explicitly preserves it), so
    # restore the publication setting for exactly those table rows.
    if experiment in PROBABILITY_TABLE_EXPERIMENTS:
        experiment_definition["disable_kde"] = False
    return create_experiment_config(
        experiment,
        experiment_definition,
        config,
        ExperimentType.JOURNAL,
        finetune_config,
        use_ia_early_stopping=False,
    )


def observed_metrics(frame) -> dict[tuple[str, str], float]:
    observed = {}
    for row in frame.to_dict("records"):
        group = row["group"]
        if group not in PUBLICATION_GROUPS:
            continue
        try:
            value = float(row["scalar_value"])
        except (TypeError, ValueError):
            continue
        observed[(row["log_type"], group)] = value
    return observed


def compare_targets(
    job: dict[str, Any],
    observed: dict[tuple[str, str], float],
    target_index: dict[tuple[str, int, int], list[dict]],
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    comparisons = []
    targets = target_index[(job["experiment"], job["seed"], job["fold"])]
    for target in targets:
        # Fine-tuned checkpoints are evaluated once under "Test".  The same
        # observed values are compared independently with the archived
        # zero-shot and final-model targets to identify the checkpoint role.
        observed_log_type = "Test" if job["finetune_dataset"] else target["log_type"]
        errors = {}
        missing = []
        for group, expected in target["metrics"].items():
            key = (observed_log_type, group)
            if key not in observed:
                missing.append(group)
                continue
            errors[group] = abs(observed[key] - expected)
        complete = not missing and len(errors) == len(target["metrics"])
        exact = complete and all(
            math.isclose(
                observed[(observed_log_type, group)],
                expected,
                rel_tol=rtol,
                abs_tol=atol,
            )
            for group, expected in target["metrics"].items()
        )
        rounded_3dp = complete and all(
            round(observed[(observed_log_type, group)], 3) == round(expected, 3)
            for group, expected in target["metrics"].items()
        )
        comparisons.append(
            {
                "target_id": target["target_id"],
                "target_log_type": target["log_type"],
                "tables": target["tables"],
                "exact_match": exact,
                "rounded_3dp_match": rounded_3dp,
                "max_abs_error": max(errors.values()) if complete else None,
                "missing_metrics": missing,
                "metric_errors": errors,
            }
        )
    return comparisons


def evaluate_job(
    job: dict[str, Any],
    gpu: int,
    config_path: Path,
    output_dir: Path,
    target_index: dict[tuple[str, int, int], list[dict]],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    import torch
    from MIA.experiment import ExperimentRunner

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available in worker assigned to physical GPU {gpu}")
    torch.cuda.set_device(0)
    work_dir = output_dir / "work" / job["job_id"]
    work_dir.mkdir(parents=True, exist_ok=True)
    config = create_job_config(job, config_path, work_dir)
    frame = ExperimentRunner(config).evaluate_checkpoint(
        job["candidate"], job["seed"], job["fold"]
    )
    observed = observed_metrics(frame)
    return {
        **job,
        "status": "completed",
        "gpu": gpu,
        "atol": atol,
        "rtol": rtol,
        "comparisons": compare_targets(job, observed, target_index, atol, rtol),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def worker_main(
    gpu: int,
    job_queue,
    result_queue,
    config_path: str,
    output_dir: str,
    manifest_path: str,
    atol: float,
    rtol: float,
) -> None:
    import gc

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    add_project_import_paths()
    _, target_index = load_manifest(Path(manifest_path))
    output = Path(output_dir)
    for job in iter(job_queue.get, None):
        result_path = output / "jobs" / f"{job['job_id']}.json"
        log_path = output / "logs" / f"{job['job_id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    print(f"Physical GPU: {gpu}", flush=True)
                    print(f"Checkpoint: {job['candidate']}", flush=True)
                    result = evaluate_job(
                        job,
                        gpu,
                        Path(config_path),
                        output,
                        target_index,
                        atol,
                        rtol,
                    )
        except Exception as error:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            result = {
                **job,
                "status": "failed",
                "gpu": gpu,
                "error": f"{type(error).__name__}: {error}",
                "comparisons": [],
            }
        finally:
            # Long-lived workers evaluate many large models.  Release any
            # cyclic references and cached blocks before taking the next job.
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        atomic_json(result_path, result)
        result_queue.put((job["job_id"], result["status"]))


def data_preflight(jobs: list[dict[str, Any]]) -> None:
    add_project_import_paths()
    from MIA.data import make_audio_datasets

    base_versions = {
        "1.8" if "podcast_1.8" in job["experiment"] else "1.11"
        for job in jobs
        if not job["finetune_dataset"]
    }
    for version in sorted(base_versions):
        print(f"Checking MSP-Podcast {version} dataset cache", flush=True)
        make_audio_datasets(
            datasets_to_load=["podcast"],
            kde_size=4,
            podcast_version=version,
            prune_annotators=True,
        )
    for dataset in sorted({job["finetune_dataset"] for job in jobs if job["finetune_dataset"]}):
        print(f"Checking {dataset} five-fold dataset cache", flush=True)
        make_audio_datasets(
            datasets_to_load=[dataset],
            kde_size=4,
            podcast_version="1.11",
            cross_validation_folds=5,
            prune_annotators=True,
            legacy_journal_splits=True,
        )


def flatten_results(
    output_dir: Path,
    inventory: list[dict[str, Any]],
    manifest: dict[str, Any],
    evaluation_signature: str,
) -> dict[str, Any]:
    completed = []
    for path in sorted((output_dir / "jobs").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("evaluation_signature") == evaluation_signature:
            completed.append(result)

    candidate_rows = []
    comparison_rows = []
    completed_aliases = {}
    for result in completed:
        for absolute, relative in zip(result["aliases"], result["alias_relative_paths"]):
            exact_count = sum(item["exact_match"] for item in result.get("comparisons", []))
            candidate_rows.append(
                {
                    "candidate": absolute,
                    "relative_path": relative,
                    "experiment": result["experiment"],
                    "seed": result["seed"],
                    "fold": result["fold"],
                    "sha256": result["sha256"],
                    "status": result["status"],
                    "gpu": result.get("gpu"),
                    "exact_target_count": exact_count,
                    "error": result.get("error", ""),
                }
            )
            completed_aliases[absolute] = result["status"]
            for comparison in result.get("comparisons", []):
                comparison_rows.append(
                    {
                        "candidate": absolute,
                        "relative_path": relative,
                        "experiment": result["experiment"],
                        "seed": result["seed"],
                        "fold": result["fold"],
                        "target_id": comparison["target_id"],
                        "target_log_type": comparison["target_log_type"],
                        "tables": ",".join(comparison["tables"]),
                        "exact_match": comparison["exact_match"],
                        "rounded_3dp_match": comparison["rounded_3dp_match"],
                        "max_abs_error": comparison["max_abs_error"],
                        "missing_metrics": ",".join(comparison["missing_metrics"]),
                    }
                )

    for item in inventory:
        if item["candidate"] in completed_aliases:
            continue
        candidate_rows.append(
            {
                "candidate": item["candidate"],
                "relative_path": item["relative_path"],
                "experiment": item.get("routed_experiment"),
                "seed": item.get("seed"),
                "fold": item.get("fold"),
                "sha256": item.get("sha256"),
                "status": item["status"],
                "gpu": None,
                "exact_target_count": 0,
                "error": item.get("error", ""),
            }
        )

    candidate_rows.sort(key=lambda row: row["relative_path"])
    comparison_rows.sort(key=lambda row: (row["target_id"], row["relative_path"]))
    exact_rows = [row for row in comparison_rows if row["exact_match"]]
    best_rows = []
    for target_id, rows in _group_rows(comparison_rows, "target_id").items():
        numeric = [row for row in rows if row["max_abs_error"] is not None]
        if not numeric:
            continue
        best_error = min(row["max_abs_error"] for row in numeric)
        best_rows.extend(row for row in numeric if row["max_abs_error"] == best_error)

    write_csv(output_dir / "all_candidates.csv", candidate_rows)
    write_csv(output_dir / "target_comparisons.csv", comparison_rows)
    write_csv(output_dir / "exact_matches.csv", exact_rows)
    write_csv(output_dir / "best_candidates.csv", best_rows)

    model_map = {
        target["target_id"]: {
            "experiment": target["experiment"],
            "dataset": target["dataset"],
            "seed": target["seed"],
            "fold": target["fold"],
            "log_type": target["log_type"],
            "tables": target["tables"],
            "exact_matches": [],
            "best_candidates": [],
            "best_max_abs_error": None,
            "status": "no_candidate_evaluated",
        }
        for target in manifest["targets"]
    }
    comparison_by_target = _group_rows(comparison_rows, "target_id")
    for target_id, rows in comparison_by_target.items():
        exact = [row["candidate"] for row in rows if row["exact_match"]]
        numeric = [row for row in rows if row["max_abs_error"] is not None]
        best_error = min((row["max_abs_error"] for row in numeric), default=None)
        model_map[target_id].update(
            {
                "exact_matches": exact,
                "best_candidates": [
                    row["candidate"]
                    for row in numeric
                    if row["max_abs_error"] == best_error
                ],
                "best_max_abs_error": best_error,
                "status": "exact_match" if exact else "compared_no_exact_match",
            }
        )
    (output_dir / "experiment_model_map.json").write_text(
        json.dumps(model_map, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "inventoried_checkpoint_files": len(inventory),
        "completed_or_failed_jobs": len(completed),
        "evaluation_signature": evaluation_signature,
        "candidate_rows": len(candidate_rows),
        "paper_targets": len(manifest["targets"]),
        "paper_targets_compared": len(comparison_by_target),
        "target_comparisons": len(comparison_rows),
        "exact_matches": len(exact_rows),
        "targets_with_an_exact_match": sum(
            bool(value["exact_matches"]) for value in model_map.values()
        ),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


def main() -> None:
    args = parse_args()
    weights_dir = args.weights_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest.resolve()
    config_path = args.config.resolve()
    data_config_path = PROJECT_ROOT / "data_config.yaml"
    os.environ["MIA_DATA_CONFIG_PATH"] = str(data_config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, target_index = load_manifest(manifest_path)
    atol = args.atol if args.atol is not None else manifest["comparison"]["default_atol"]
    rtol = args.rtol if args.rtol is not None else manifest["comparison"]["rtol"]
    experiments = set(args.experiments) if args.experiments else None

    print(f"Scanning {weights_dir}", flush=True)
    inventory = inventory_candidates(weights_dir, target_index, experiments)
    write_csv(output_dir / "checkpoint_inventory.csv", inventory)
    cache_identity = None
    if not args.dry_run:
        cache_identity = configure_dataset_cache(
            args.dataset_cache,
            inventory,
            output_dir,
        )
    evaluation_signature = hashlib.sha256(
        json.dumps(
            {
                "schema": EVALUATION_SCHEMA_VERSION,
                "manifest_sha256": file_digest(manifest_path),
                "config_sha256": file_digest(config_path),
                "data_config_sha256": file_digest(data_config_path),
                "dataset_cache_signature": (
                    cache_identity["signature"] if cache_identity else "dry-run-not-checked"
                ),
                "atol": atol,
                "rtol": rtol,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    jobs = build_jobs(inventory, args.no_deduplicate, evaluation_signature)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    print(
        f"Planned {len(jobs)} GPU evaluations from {len(inventory)} checkpoint files",
        flush=True,
    )
    if args.dry_run:
        plan = {
            "checkpoint_files": len(inventory),
            "planned_gpu_jobs": len(jobs),
            "status_counts": {
                status: sum(item["status"] == status for item in inventory)
                for status in sorted({item["status"] for item in inventory})
            },
            "jobs": jobs,
        }
        (output_dir / "dry_run_plan.json").write_text(
            json.dumps(plan, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({key: value for key, value in plan.items() if key != "jobs"}, indent=2))
        return

    if not args.skip_data_preflight:
        data_preflight(jobs)

    pending = []
    for job in jobs:
        result_path = output_dir / "jobs" / f"{job['job_id']}.json"
        if result_path.exists() and not args.rerun:
            try:
                previous = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = None
            if (
                previous is not None
                and previous.get("status") == "completed"
                and previous.get("evaluation_signature") == evaluation_signature
            ):
                continue
        pending.append(job)
    print(f"Pending evaluations: {len(pending)} (resuming {len(jobs) - len(pending)})")

    gpu_ids = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must name at least one GPU")
    if pending:
        context = mp.get_context("spawn")
        job_queue = context.Queue()
        result_queue = context.Queue()
        workers = [
            context.Process(
                target=worker_main,
                args=(
                    gpu,
                    job_queue,
                    result_queue,
                    str(config_path),
                    str(output_dir),
                    str(manifest_path),
                    atol,
                    rtol,
                ),
                name=f"checkpoint-gpu-{gpu}",
            )
            for gpu in gpu_ids
        ]
        for worker in workers:
            worker.start()
        for job in pending:
            job_queue.put(job)
        for _ in workers:
            job_queue.put(None)
        try:
            completed = 0
            while completed < len(pending):
                try:
                    job_id, status = result_queue.get(timeout=30)
                except queue.Empty:
                    crashed = [
                        worker
                        for worker in workers
                        if worker.exitcode is not None and worker.exitcode != 0
                    ]
                    if crashed:
                        details = ", ".join(
                            f"{worker.name} exit={worker.exitcode}" for worker in crashed
                        )
                        raise RuntimeError(f"GPU worker terminated unexpectedly: {details}")
                    if not any(worker.is_alive() for worker in workers):
                        raise RuntimeError(
                            "All GPU workers exited before every queued job returned a result"
                        )
                    continue
                completed += 1
                print(f"[{completed}/{len(pending)}] {job_id}: {status}", flush=True)
        except KeyboardInterrupt:
            print("Interrupted; completed job JSON files are safe and will be resumed.")
            for worker in workers:
                worker.terminate()
            raise
        finally:
            for worker in workers:
                worker.join()

    summary = flatten_results(output_dir, inventory, manifest, evaluation_signature)
    print(json.dumps(summary, indent=2))
    print(f"Exact model map: {output_dir / 'experiment_model_map.json'}")


if __name__ == "__main__":
    main()
