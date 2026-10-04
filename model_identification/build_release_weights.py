#!/usr/bin/env python3
"""Build, repair, and verify the minimal checkpoint set for the paper.

The workflow first stages every checkpoint identified in the recovery pool,
then retrains any missing *base* experiment seed, and finally reloads every
staged checkpoint and compares it with its archived paper targets. One
persistent worker is assigned to each requested GPU and all GPU work is
resumable.

Mapping-derived zero-shot models are intentionally not saved. Missing final
fine-tuned Table VIII models are recreated by ``verify_paper_results.py``;
they are not base release checkpoints.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "run_experiments" / "configs" / "journal_experiments.yaml"
WORKFLOW_SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=PROJECT_ROOT / "potential_model_weights",
        help="Unsorted recovery pool",
    )
    parser.add_argument(
        "--identification-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=SCRIPT_DIR / "paper_result_targets.json",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "paper_model_weights",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=SCRIPT_DIR / "release_build",
        help="Resumable logs, training outputs, and verification results",
    )
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink", "move"),
        default="copy",
        help="How recovered files are staged (default: copy)",
    )
    parser.add_argument(
        "--no-retrain-missing",
        action="store_true",
        help="Report missing base checkpoints without recreating them",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Do not reload and verify every staged checkpoint",
    )
    parser.add_argument("--skip-data-preflight", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def add_import_paths() -> None:
    for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "run_experiments", SCRIPT_DIR):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def choose_base_checkpoints(
    rows: list[dict[str, str]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if "_ft_on_" in row["experiment"]:
            continue
        grouped[
            (row["experiment"], int(row["seed"]), int(row["fold"]), row["candidate"])
        ].append(row)

    by_run: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for (experiment, seed, fold, candidate), comparisons in grouped.items():
        errors = [
            float(row["max_abs_error"])
            for row in comparisons
            if row["max_abs_error"]
        ]
        by_run[(experiment, seed, fold)].append(
            {
                "candidate": candidate,
                "relative_path": comparisons[0]["relative_path"],
                "exact_target_count": sum(truth(row["exact_match"]) for row in comparisons),
                "rounded_target_count": sum(
                    truth(row["rounded_3dp_match"]) for row in comparisons
                ),
                "best_max_abs_error": min(errors, default=float("inf")),
                "worst_max_abs_error": max(errors, default=float("inf")),
            }
        )

    selected: dict[tuple[str, int, int], dict[str, Any]] = {}
    for key, candidates in by_run.items():
        candidates.sort(
            key=lambda item: (
                -item["exact_target_count"],
                -item["rounded_target_count"],
                item["best_max_abs_error"],
                item["worst_max_abs_error"],
                item["relative_path"],
            )
        )
        best = candidates[0]
        if best["exact_target_count"]:
            best["selection"] = "exact"
            selected[key] = best
        elif key[0] == "baseline_aggregate" and best["best_max_abs_error"] <= 2e-6:
            # The seed-1 consensus model is 1.37e-6 from the archived result,
            # while its four sibling seeds match. Preserve it, but label the
            # numerical drift rather than silently widening the tolerance.
            best["selection"] = "near_exact_numeric_drift"
            selected[key] = best
    return selected


def choose_direct_finetuned_checkpoints(
    rows: list[dict[str, str]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    candidates: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if (
            "_ft_on_" in row["experiment"]
            and row["target_log_type"] == "Test"
            and truth(row["exact_match"])
        ):
            candidates[(row["experiment"], int(row["seed"]), int(row["fold"]))].append(row)
    selected = {}
    for key, matches in candidates.items():
        matches.sort(key=lambda row: (float(row["max_abs_error"]), row["relative_path"]))
        row = matches[0]
        selected[key] = {
            "candidate": row["candidate"],
            "relative_path": row["relative_path"],
            "selection": "exact_final",
            "best_max_abs_error": float(row["max_abs_error"]),
        }
    return selected


def destination_for(kind: str, experiment: str, seed: int, fold: int) -> Path:
    return (
        Path("checkpoints")
        / kind
        / experiment
        / f"model_prob4_seed_{seed}_fold_{fold}"
        / "paper.ckpt"
    )


def materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise FileExistsError(f"Different checkpoint already exists: {destination}")
        if mode == "move" and source.resolve() != destination.resolve():
            source.unlink()
        return
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.move(source, destination)


def recovered_records(
    weights_dir: Path,
    output_dir: Path,
    selections_by_kind: tuple[
        tuple[str, dict[tuple[str, int, int], dict[str, Any]]], ...
    ],
    mode: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    records = []
    for kind, selections in selections_by_kind:
        for (experiment, seed, fold), selection in sorted(selections.items()):
            source = Path(selection["candidate"]).resolve()
            try:
                source_relative = source.relative_to(weights_dir).as_posix()
            except ValueError as error:
                raise ValueError(f"Selected checkpoint is outside {weights_dir}: {source}") from error
            destination_relative = destination_for(kind, experiment, seed, fold)
            destination = output_dir / destination_relative
            if not source.is_file() and not destination.is_file():
                # A previous successful move remains valid because its
                # destination exists; otherwise this is now a missing model.
                continue
            if not dry_run and source.is_file():
                materialize(source, destination, mode)
            checkpoint = destination if destination.is_file() else source
            records.append(
                {
                    "kind": kind,
                    "experiment": experiment,
                    "seed": seed,
                    "fold": fold,
                    "path": destination_relative.as_posix(),
                    "source_relative_path": source_relative,
                    "sha256": sha256(checkpoint),
                    "size_bytes": checkpoint.stat().st_size,
                    "selection": selection["selection"],
                    "identification_max_abs_error": selection["best_max_abs_error"],
                }
            )
    return records


def reusable_generated_records(output_dir: Path) -> list[dict[str, Any]]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reusable = []
    for record in manifest.get("checkpoints", []):
        if not str(record.get("selection", "")).startswith("retrained"):
            continue
        path = output_dir / record["path"]
        if path.is_file() and sha256(path) == record.get("sha256"):
            reusable.append(record)
    return reusable


def target_index(targets_document: dict[str, Any]):
    by_id = {}
    by_run: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for target in targets_document["targets"]:
        by_id[target["target_id"]] = target
        by_run[(target["experiment"], target["seed"], target["fold"])].append(target)
    return by_id, by_run


def missing_base_jobs(
    lookup: dict[tuple[str, int, int, str], dict[str, Any]],
    targets_by_run: dict[tuple[str, int, int], list[dict[str, Any]]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    jobs = []
    for (experiment, seed, fold), targets in sorted(targets_by_run.items()):
        if "_ft_on_" in experiment or (experiment, seed, fold, "base") in lookup:
            continue
        destination_relative = destination_for("base", experiment, seed, fold)
        jobs.append(
            {
                "kind": "train_base",
                "experiment": experiment,
                "seed": seed,
                "fold": fold,
                "finetune_dataset": None,
                "target_ids": [target["target_id"] for target in targets],
                "destination": str((output_dir / destination_relative).resolve()),
                "destination_relative": destination_relative.as_posix(),
            }
        )
    return jobs


def create_training_config(job: dict[str, Any], config_path: Path, work_dir: Path):
    add_import_paths()
    from identify_paper_models import create_job_config

    config = create_job_config(
        {"experiment": job["experiment"], "finetune_dataset": None},
        config_path,
        work_dir,
    )
    config.training_config.test_only = False
    return config


def training_worker_main(
    gpu: int,
    job_queue,
    result_queue,
    config_path: str,
    state_dir: str,
    targets_path: str,
    atol: float,
    rtol: float,
) -> None:
    import gc

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    add_import_paths()
    from MIA.experiment import ExperimentRunner
    from verify_paper_results import compare_target, observed_metrics

    targets_document = json.loads(Path(targets_path).read_text(encoding="utf-8"))
    targets_by_id = {
        target["target_id"]: target for target in targets_document["targets"]
    }
    state = Path(state_dir)
    for job in iter(job_queue.get, None):
        result_path = state / "training" / "jobs" / f"{job['job_id']}.json"
        log_path = state / "training" / "logs" / f"{job['job_id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    print(f"Physical GPU: {gpu}", flush=True)
                    print(
                        f"Retraining {job['experiment']} seed={job['seed']} "
                        f"fold={job['fold']}",
                        flush=True,
                    )
                    work_dir = state / "training" / "work" / job["job_id"]
                    work_dir.mkdir(parents=True, exist_ok=True)
                    config = create_training_config(job, Path(config_path), work_dir)
                    frame, trained_path = ExperimentRunner(
                        config
                    ).train_and_evaluate_checkpoint(job["seed"], job["fold"])
                    destination = Path(job["destination"])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(
                        destination.suffix + f".{os.getpid()}.tmp"
                    )
                    shutil.copy2(trained_path, temporary)
                    temporary.replace(destination)
                    frame_path = state / "training" / "frames" / f"{job['job_id']}.csv"
                    frame_path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_csv(frame_path, index=False)
                    observed = observed_metrics(frame)
                    comparisons = [
                        compare_target(targets_by_id[target_id], observed, atol, rtol)
                        for target_id in job["target_ids"]
                    ]
                    digest = sha256(destination)
                    max_errors = [
                        comparison["max_abs_error"]
                        for comparison in comparisons
                        if comparison["max_abs_error"] is not None
                    ]
                    all_exact = all(
                        comparison["exact_match"] for comparison in comparisons
                    )
                    all_rounded = all(
                        comparison["rounded_3dp_match"] for comparison in comparisons
                    )
                    result = {
                        **job,
                        "status": "completed",
                        "gpu": gpu,
                        "frame_path": str(frame_path.resolve()),
                        "comparisons": comparisons,
                        "checkpoint_record": {
                            "kind": "base",
                            "experiment": job["experiment"],
                            "seed": job["seed"],
                            "fold": job["fold"],
                            "path": job["destination_relative"],
                            "source_relative_path": None,
                            "sha256": digest,
                            "size_bytes": destination.stat().st_size,
                            "selection": (
                                "retrained_exact"
                                if all_exact
                                else "retrained_rounded_3dp"
                                if all_rounded
                                else "retrained_mismatch"
                            ),
                            "identification_max_abs_error": max(max_errors, default=None),
                            "training_job_id": job["job_id"],
                        },
                    }
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
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        atomic_json(result_path, result)
        result_queue.put((job["job_id"], result["status"]))


def parse_gpus(value: str) -> list[int]:
    gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU ID is required")
    return gpus


def completed_results(directory: Path, signature: str) -> list[dict[str, Any]]:
    results = []
    for path in sorted(directory.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("evaluation_signature") == signature:
            results.append(result)
    return results


def reusable_training_result(result: dict[str, Any], rerun: bool) -> bool:
    if rerun or result.get("status") != "completed":
        return False
    destination = Path(result.get("destination", ""))
    record = result.get("checkpoint_record") or {}
    return (
        destination.is_file()
        and bool(record.get("sha256"))
        and sha256(destination) == record["sha256"]
    )


def wait_for_workers(workers, result_queue, expected: int, action: str) -> None:
    finished = 0
    while finished < expected:
        try:
            job_id, status = result_queue.get(timeout=30)
        except queue.Empty:
            dead = [worker for worker in workers if not worker.is_alive() and worker.exitcode]
            if dead:
                raise RuntimeError(
                    f"GPU worker exited unexpectedly: {[worker.exitcode for worker in dead]}"
                )
            continue
        finished += 1
        print(f"{action.capitalize()} {finished}/{expected}: {job_id} ({status})", flush=True)
    for worker in workers:
        worker.join()


def run_training_jobs(
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    signature: str,
) -> list[dict[str, Any]]:
    jobs_dir = args.state_dir / "training" / "jobs"
    for job in jobs:
        job["evaluation_signature"] = signature
        identity = {
            key: job[key]
            for key in ("experiment", "seed", "fold", "destination_relative")
        }
        identity["signature"] = signature
        job["job_id"] = hashlib.sha1(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
    existing = {
        result["job_id"]: result
        for result in completed_results(jobs_dir, signature)
        if reusable_training_result(result, args.rerun)
    }
    pending = [job for job in jobs if job["job_id"] not in existing]
    print(
        f"Missing-base training: {len(jobs)} planned, "
        f"{len(existing)} reusable, {len(pending)} pending",
        flush=True,
    )
    if pending:
        context = mp.get_context("spawn")
        job_queue = context.Queue()
        result_queue = context.Queue()
        workers = [
            context.Process(
                target=training_worker_main,
                args=(
                    gpu,
                    job_queue,
                    result_queue,
                    str(args.config.resolve()),
                    str(args.state_dir.resolve()),
                    str(args.targets.resolve()),
                    args.atol,
                    args.rtol,
                ),
            )
            for gpu in parse_gpus(args.gpus)
        ]
        for worker in workers:
            worker.start()
        for job in pending:
            job_queue.put(job)
        for _ in workers:
            job_queue.put(None)
        wait_for_workers(workers, result_queue, len(pending), "trained")
    job_ids = {job["job_id"] for job in jobs}
    return [
        result
        for result in completed_results(jobs_dir, signature)
        if result["job_id"] in job_ids
    ]


def verification_jobs(
    records: list[dict[str, Any]],
    output_dir: Path,
    targets_by_run: dict[tuple[str, int, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    jobs = []
    for record in records:
        targets = targets_by_run.get(
            (record["experiment"], record["seed"], record["fold"]), []
        )
        if record["kind"] == "finetuned":
            targets = [target for target in targets if target["log_type"] == "Test"]
        if not targets:
            raise ValueError(f"No paper targets for release checkpoint: {record}")
        dataset = None
        if "_ft_on_" in record["experiment"]:
            dataset = record["experiment"].split("_ft_on_", 1)[1].split("_", 1)[0]
        jobs.append(
            {
                "kind": "direct" if dataset else "base",
                "experiment": record["experiment"],
                "seed": record["seed"],
                "fold": record["fold"],
                "finetune_dataset": dataset,
                "checkpoint": str((output_dir / record["path"]).resolve()),
                "checkpoint_sha256": record["sha256"],
                "target_ids": [target["target_id"] for target in targets],
            }
        )
    return jobs


def validate_staged_records(records: list[dict[str, Any]], output_dir: Path) -> None:
    """Fail before GPU verification if a manifest path is absent or corrupted."""
    for record in records:
        path = output_dir / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Staged release checkpoint is missing: {path}")
        actual = sha256(path)
        if actual != record["sha256"]:
            raise ValueError(
                f"Staged release checkpoint SHA-256 mismatch: {path} "
                f"({actual} != {record['sha256']})"
            )


def run_verification_jobs(
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    signature: str,
) -> list[dict[str, Any]]:
    from verify_paper_results import worker_main

    verification_dir = args.state_dir / "verification"
    for job in jobs:
        identity = {
            key: job[key]
            for key in (
                "kind", "experiment", "seed", "fold", "checkpoint_sha256", "target_ids"
            )
        }
        identity["signature"] = signature
        job["evaluation_signature"] = signature
        job["job_id"] = hashlib.sha1(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
    existing = {
        result["job_id"]: result
        for result in completed_results(verification_dir / "jobs", signature)
        if result["status"] == "completed" and not args.rerun
    }
    pending = [job for job in jobs if job["job_id"] not in existing]
    print(
        f"Release verification: {len(jobs)} planned, "
        f"{len(existing)} reusable, {len(pending)} pending",
        flush=True,
    )
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
                    str(args.config.resolve()),
                    str(verification_dir.resolve()),
                    str(args.targets.resolve()),
                    args.atol,
                    args.rtol,
                ),
            )
            for gpu in parse_gpus(args.gpus)
        ]
        for worker in workers:
            worker.start()
        for job in pending:
            job_queue.put(job)
        for _ in workers:
            job_queue.put(None)
        wait_for_workers(workers, result_queue, len(pending), "verified")
    job_ids = {job["job_id"] for job in jobs}
    return [
        result
        for result in completed_results(verification_dir / "jobs", signature)
        if result["job_id"] in job_ids
    ]


def apply_verification(
    records: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    results: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    result_by_id = {result["job_id"]: result for result in results}
    job_by_checkpoint = {job["checkpoint"]: job for job in jobs}
    statuses = Counter()
    rows = []
    for record in records:
        checkpoint = str((output_dir / record["path"]).resolve())
        job = job_by_checkpoint[checkpoint]
        result = result_by_id.get(job["job_id"])
        if result is None:
            status = "not_run"
            comparisons = []
            error = ""
        elif result["status"] == "failed":
            status = "failed"
            comparisons = []
            error = result.get("error", "")
        else:
            comparisons = result.get("comparisons", [])
            if comparisons and all(item["exact_match"] for item in comparisons):
                status = "exact_match"
            elif comparisons and all(item.get("rounded_3dp_match") for item in comparisons):
                status = "rounded_3dp_match"
            else:
                status = "mismatch"
            error = ""
        max_errors = [
            item["max_abs_error"]
            for item in comparisons
            if item.get("max_abs_error") is not None
        ]
        record["verification"] = {
            "status": status,
            "job_id": job["job_id"],
            "max_abs_error": max(max_errors, default=None),
            "target_count": len(comparisons),
            "error": error,
        }
        statuses[status] += 1
        rows.append(
            {
                "experiment": record["experiment"],
                "seed": record["seed"],
                "fold": record["fold"],
                "kind": record["kind"],
                "path": record["path"],
                **record["verification"],
            }
        )
    return {
        "checkpoint_statuses": dict(sorted(statuses.items())),
        "all_checkpoints_exact": statuses.get("exact_match", 0) == len(records),
        "rows": rows,
    }


def build_recipes(
    targets_document: dict[str, Any],
    lookup: dict[tuple[str, int, int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    recipes = []
    for target in targets_document["targets"]:
        experiment = target["experiment"]
        seed = int(target["seed"])
        fold = int(target["fold"])
        if "_ft_on_" not in experiment:
            checkpoint = lookup.get((experiment, seed, fold, "base"))
            method = "evaluate_checkpoint" if checkpoint else "missing_base_checkpoint"
        else:
            base_experiment = experiment.split("_ft_on_", 1)[0]
            base_checkpoint = lookup.get((base_experiment, seed, 0, "base"))
            direct = lookup.get((experiment, seed, fold, "finetuned"))
            if target["log_type"] == "Test zero-shot":
                checkpoint = base_checkpoint
                method = "map_then_test" if checkpoint else "missing_base_checkpoint"
            elif direct:
                checkpoint = direct
                method = "evaluate_checkpoint"
            else:
                checkpoint = base_checkpoint
                method = "retrain_then_test" if checkpoint else "missing_base_checkpoint"
        recipes.append(
            {
                "target_id": target["target_id"],
                "experiment": experiment,
                "dataset": target["dataset"],
                "seed": seed,
                "fold": fold,
                "log_type": target["log_type"],
                "tables": target["tables"],
                "method": method,
                "checkpoint_path": checkpoint["path"] if checkpoint else None,
            }
        )
    return recipes


def write_verification_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    add_import_paths()
    from identify_paper_models import configure_dataset_cache, data_preflight

    args.weights_dir = args.weights_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.state_dir = args.state_dir.resolve()
    comparisons = read_csv(args.identification_dir / "target_comparisons.csv")
    targets_document = json.loads(args.targets.read_text(encoding="utf-8"))
    _, targets_by_run = target_index(targets_document)
    args.atol = (
        args.atol
        if args.atol is not None
        else targets_document["comparison"]["default_atol"]
    )
    args.rtol = (
        args.rtol
        if args.rtol is not None
        else targets_document["comparison"]["rtol"]
    )

    base = choose_base_checkpoints(comparisons)
    final = choose_direct_finetuned_checkpoints(comparisons)
    records = recovered_records(
        args.weights_dir,
        args.output_dir,
        (("base", base), ("finetuned", final)),
        args.mode,
        args.dry_run,
    )
    keys = {
        (record["experiment"], record["seed"], record["fold"], record["kind"])
        for record in records
    }
    for record in reusable_generated_records(args.output_dir):
        key = (record["experiment"], record["seed"], record["fold"], record["kind"])
        if key not in keys:
            records.append(record)
            keys.add(key)
    lookup = {
        (record["experiment"], record["seed"], record["fold"], record["kind"]): record
        for record in records
    }
    train_jobs = missing_base_jobs(lookup, targets_by_run, args.output_dir)
    verification_plan = verification_jobs(records, args.output_dir, targets_by_run)
    print(
        f"Staged/reusable checkpoints: {len(records)}; "
        f"missing base checkpoints: {len(train_jobs)}; "
        f"current verification jobs: {len(verification_plan)}",
        flush=True,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "checkpoint_files": len(records),
                    "missing_base_runs": [
                        {
                            key: job[key]
                            for key in ("experiment", "seed", "fold", "destination_relative")
                        }
                        for job in train_jobs
                    ],
                    "verification_jobs_after_successful_retraining": (
                        len(records) + len(train_jobs)
                    ),
                    "would_retrain": not args.no_retrain_missing,
                    "would_verify": not args.no_verify,
                },
                indent=2,
            )
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    inventory_jobs = verification_plan + train_jobs
    inventory = [
        {
            "status": "evaluable",
            "routed_experiment": job["experiment"],
            "finetune_dataset": job["finetune_dataset"],
        }
        for job in inventory_jobs
    ]
    cache_identity = configure_dataset_cache(
        args.dataset_cache, inventory, args.state_dir
    )
    signature_payload = {
        "schema": WORKFLOW_SCHEMA_VERSION,
        "targets": sha256(args.targets),
        "config": sha256(args.config),
        "cache": cache_identity["signature"],
        "atol": args.atol,
        "rtol": args.rtol,
    }
    workflow_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if not args.skip_data_preflight:
        data_preflight(inventory_jobs)

    training_results = []
    if train_jobs and not args.no_retrain_missing:
        training_results = run_training_jobs(train_jobs, args, workflow_signature)
        for result in training_results:
            if result["status"] != "completed":
                continue
            record = result["checkpoint_record"]
            key = (record["experiment"], record["seed"], record["fold"], record["kind"])
            lookup[key] = record
        records = sorted(
            lookup.values(),
            key=lambda item: (
                item["kind"], item["experiment"], item["seed"], item["fold"]
            ),
        )

    unresolved_jobs = missing_base_jobs(lookup, targets_by_run, args.output_dir)
    verification = {
        "checkpoint_statuses": {"not_run": len(records)},
        "all_checkpoints_exact": False,
        "rows": [],
    }
    if not args.no_verify:
        validate_staged_records(records, args.output_dir)
        verification_plan = verification_jobs(records, args.output_dir, targets_by_run)
        verification_signature = hashlib.sha256(
            json.dumps(
                {
                    **signature_payload,
                    "checkpoints": sorted(
                        (record["path"], record["sha256"]) for record in records
                    ),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        verification_results = run_verification_jobs(
            verification_plan, args, verification_signature
        )
        verification = apply_verification(
            records, verification_plan, verification_results, args.output_dir
        )
        write_verification_csv(
            args.state_dir / "checkpoint_verification.csv", verification["rows"]
        )
        atomic_json(
            args.state_dir / "verification_summary.json",
            {key: value for key, value in verification.items() if key != "rows"},
        )

    recipes = build_recipes(targets_document, lookup)
    method_counts = Counter(recipe["method"] for recipe in recipes)
    training_statuses = Counter(result["status"] for result in training_results)
    summary = {
        "checkpoint_files": len(records),
        "checkpoint_bytes": sum(item["size_bytes"] for item in records),
        "paper_targets": len(recipes),
        "methods": dict(sorted(method_counts.items())),
        "missing_base_runs": len(unresolved_jobs),
        "training_statuses": dict(sorted(training_statuses.items())),
        "checkpoint_verification": {
            key: value for key, value in verification.items() if key != "rows"
        },
    }
    manifest = {
        "schema_version": 1,
        "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
        "description": "Minimal checkpoint set and complete Tables II--VIII reproduction recipes.",
        "source_weights_directory": (
            args.weights_dir.relative_to(PROJECT_ROOT).as_posix()
            if args.weights_dir.is_relative_to(PROJECT_ROOT)
            else args.weights_dir.name
        ),
        "materialization_mode": args.mode,
        "checkpoints": records,
        "target_recipes": recipes,
        "summary": summary,
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)

    complete = (
        not unresolved_jobs
        and (args.no_verify or verification["all_checkpoints_exact"])
    )
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
