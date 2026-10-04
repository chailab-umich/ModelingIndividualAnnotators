#!/usr/bin/env python3
"""Reproduce every archived paper scalar and verify the notebook LaTeX.

One persistent worker is assigned to each requested GPU.  The run is
resumable, writes notebook-compatible CSV trees, compares all per-seed/fold
metrics at full precision, and executes ``journal_results.ipynb`` only after
all scalar targets match.
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
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RELEASE_DIR = PROJECT_ROOT / "paper_model_weights"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reproduction_results"
DEFAULT_TARGETS = SCRIPT_DIR / "paper_result_targets.json"
DEFAULT_CONFIG = PROJECT_ROOT / "run_experiments" / "configs" / "journal_experiments.yaml"
DEFAULT_LATEX = SCRIPT_DIR / "published_latex_tables.json"
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument(
        "--retrain-missing",
        action="store_true",
        help="Recreate missing Table VIII final checkpoints (potentially long)",
    )
    parser.add_argument("--skip-notebook", action="store_true")
    parser.add_argument("--notebook", type=Path, default=PROJECT_ROOT / "journal_results.ipynb")
    parser.add_argument("--latex-reference", type=Path, default=DEFAULT_LATEX)
    parser.add_argument("--notebook-timeout", type=int, default=1800)
    parser.add_argument("--experiments", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--skip-data-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def add_import_paths() -> None:
    for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "run_experiments", SCRIPT_DIR):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_documents(args: argparse.Namespace):
    release_path = args.release_manifest or args.weights_dir / "manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    if release.get("schema_version") != 1 or targets.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    target_by_id = {target["target_id"]: target for target in targets["targets"]}
    recipes = {recipe["target_id"]: recipe for recipe in release["target_recipes"]}
    if set(recipes) != set(target_by_id):
        raise ValueError("Release recipes and paper target IDs differ; rebuild release weights")
    return release_path, release, targets, target_by_id, recipes


def validate_checkpoints(weights_dir: Path, release: dict[str, Any]) -> None:
    for number, record in enumerate(release["checkpoints"], start=1):
        path = weights_dir / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Release checkpoint is missing: {path}")
        actual = sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual}")
        if number % 25 == 0 or number == len(release["checkpoints"]):
            print(f"Validated {number}/{len(release['checkpoints'])} release checkpoints", flush=True)


def fine_dataset(experiment: str) -> str | None:
    return experiment.split("_ft_on_", 1)[1].split("_", 1)[0] if "_ft_on_" in experiment else None


def build_jobs(
    weights_dir: Path,
    target_by_id: dict[str, dict[str, Any]],
    recipes: dict[str, dict[str, Any]],
    retrain_missing: bool,
    experiments: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    target_status: dict[str, str] = {}
    for target_id, target in target_by_id.items():
        if experiments is not None and target["experiment"] not in experiments:
            target_status[target_id] = "excluded_by_filter"
            continue
        grouped[(target["experiment"], target["seed"], target["fold"])].append(target)

    jobs: list[dict[str, Any]] = []
    for (experiment, seed, fold), targets in sorted(grouped.items()):
        target_for_log = {target["log_type"]: target for target in targets}
        dataset = fine_dataset(experiment)
        if dataset is None:
            recipe = recipes[targets[0]["target_id"]]
            if recipe["method"] == "missing_base_checkpoint":
                for target in targets:
                    target_status[target["target_id"]] = "missing_base_checkpoint"
                continue
            jobs.append(
                {
                    "kind": "base",
                    "experiment": experiment,
                    "seed": seed,
                    "fold": fold,
                    "finetune_dataset": None,
                    "checkpoint": str((weights_dir / recipe["checkpoint_path"]).resolve()),
                    "target_ids": [target["target_id"] for target in targets],
                }
            )
            continue

        zero_target = target_for_log.get("Test zero-shot")
        final_target = target_for_log.get("Test")
        final_recipe = recipes[final_target["target_id"]] if final_target else None
        if final_recipe and final_recipe["method"] == "retrain_then_test":
            if retrain_missing:
                target_ids = [final_target["target_id"]]
                if zero_target:
                    target_ids.append(zero_target["target_id"])
                jobs.append(
                    {
                        "kind": "retrain",
                        "experiment": experiment,
                        "seed": seed,
                        "fold": fold,
                        "finetune_dataset": dataset,
                        "checkpoint": str((weights_dir / final_recipe["checkpoint_path"]).resolve()),
                        "target_ids": target_ids,
                    }
                )
            else:
                target_status[final_target["target_id"]] = "retraining_not_requested"
                if zero_target:
                    zero_recipe = recipes[zero_target["target_id"]]
                    jobs.append(
                        {
                            "kind": "mapping",
                            "experiment": experiment,
                            "seed": seed,
                            "fold": fold,
                            "finetune_dataset": dataset,
                            "checkpoint": str((weights_dir / zero_recipe["checkpoint_path"]).resolve()),
                            "target_ids": [zero_target["target_id"]],
                        }
                    )
            continue

        if zero_target:
            zero_recipe = recipes[zero_target["target_id"]]
            if zero_recipe["method"] == "missing_base_checkpoint":
                target_status[zero_target["target_id"]] = "missing_base_checkpoint"
            else:
                jobs.append(
                    {
                        "kind": "mapping",
                        "experiment": experiment,
                        "seed": seed,
                        "fold": fold,
                        "finetune_dataset": dataset,
                        "checkpoint": str((weights_dir / zero_recipe["checkpoint_path"]).resolve()),
                        "target_ids": [zero_target["target_id"]],
                    }
                )
        if final_target:
            if final_recipe["method"] == "missing_base_checkpoint":
                target_status[final_target["target_id"]] = "missing_base_checkpoint"
            else:
                jobs.append(
                    {
                        "kind": "direct",
                        "experiment": experiment,
                        "seed": seed,
                        "fold": fold,
                        "finetune_dataset": dataset,
                        "checkpoint": str((weights_dir / final_recipe["checkpoint_path"]).resolve()),
                        "target_ids": [final_target["target_id"]],
                    }
                )
    return jobs, target_status


def create_config(job: dict[str, Any], config_path: Path, work_dir: Path):
    add_import_paths()
    from identify_paper_models import create_job_config

    config_job = {
        "experiment": job["experiment"],
        "finetune_dataset": job["finetune_dataset"],
    }
    config = create_job_config(config_job, config_path, work_dir)
    config.training_config.test_only = job["kind"] != "retrain"
    return config


def observed_metrics(frame) -> dict[tuple[str, str], float]:
    add_import_paths()
    from identify_paper_models import observed_metrics as identify_observed_metrics

    return identify_observed_metrics(frame)


def compare_target(
    target: dict[str, Any],
    observed: dict[tuple[str, str], float],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    values = {}
    errors = {}
    missing = []
    for group, expected in target["metrics"].items():
        key = (target["log_type"], group)
        if key not in observed:
            missing.append(group)
            continue
        value = observed[key]
        values[group] = value
        errors[group] = abs(value - expected)
    complete = not missing and len(values) == len(target["metrics"])
    exact = complete and all(
        math.isclose(values[group], expected, rel_tol=rtol, abs_tol=atol)
        for group, expected in target["metrics"].items()
    )
    rounded_3dp = complete and all(
        round(values[group], 3) == round(expected, 3)
        for group, expected in target["metrics"].items()
    )
    return {
        "target_id": target["target_id"],
        "experiment": target["experiment"],
        "dataset": target["dataset"],
        "seed": target["seed"],
        "fold": target["fold"],
        "log_type": target["log_type"],
        "tables": target["tables"],
        "status": "exact_match" if exact else "mismatch",
        "exact_match": exact,
        "rounded_3dp_match": rounded_3dp,
        "max_abs_error": max(errors.values()) if complete else None,
        "missing_metrics": missing,
        "observed": values,
        "metric_errors": errors,
    }


def evaluate_job(
    job: dict[str, Any],
    gpu: int,
    config_path: Path,
    output_dir: Path,
    target_by_id: dict[str, dict[str, Any]],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    import torch
    from MIA.experiment import ExperimentRunner

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable in worker for physical GPU {gpu}")
    torch.cuda.set_device(0)
    work_dir = output_dir / "work" / job["job_id"]
    work_dir.mkdir(parents=True, exist_ok=True)
    config = create_config(job, config_path, work_dir)
    runner = ExperimentRunner(config)
    if job["kind"] in {"base", "direct"}:
        frame = runner.evaluate_checkpoint(job["checkpoint"], job["seed"], job["fold"])
    elif job["kind"] == "mapping":
        frame = runner.evaluate_mapped_checkpoint(job["checkpoint"], job["seed"], job["fold"])
    elif job["kind"] == "retrain":
        frame = runner.reproduce_finetuning(job["checkpoint"], job["seed"], job["fold"])
    else:
        raise ValueError(f"Unknown job kind: {job['kind']}")

    frame_path = output_dir / "frames" / f"{job['job_id']}.csv"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(frame_path, index=False)
    observed = observed_metrics(frame)
    comparisons = [
        compare_target(target_by_id[target_id], observed, atol, rtol)
        for target_id in job["target_ids"]
    ]
    trained_checkpoint = None
    if job["kind"] == "retrain":
        model_dir = Path(config.model_save_path) / f"model_prob4_seed_{job['seed']}_fold_{job['fold']}"
        ordinary = sorted(
            path for path in model_dir.glob("*_cp.ckpt")
            if path.stem.removesuffix("_cp").isdigit()
        )
        if len(ordinary) == 1:
            trained_checkpoint = str(ordinary[0].resolve())
    return {
        **job,
        "status": "completed",
        "gpu": gpu,
        "frame_path": str(frame_path.resolve()),
        "trained_checkpoint": trained_checkpoint,
        "comparisons": comparisons,
    }


def worker_main(
    gpu: int,
    job_queue,
    result_queue,
    config_path: str,
    output_dir: str,
    targets_path: str,
    atol: float,
    rtol: float,
) -> None:
    import gc

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    add_import_paths()
    targets = json.loads(Path(targets_path).read_text(encoding="utf-8"))
    target_by_id = {target["target_id"]: target for target in targets["targets"]}
    output = Path(output_dir)
    for job in iter(job_queue.get, None):
        result_path = output / "jobs" / f"{job['job_id']}.json"
        log_path = output / "logs" / f"{job['job_id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    print(f"Physical GPU: {gpu}", flush=True)
                    print(f"Job: {job['kind']} {job['experiment']} seed={job['seed']} fold={job['fold']}", flush=True)
                    result = evaluate_job(
                        job, gpu, Path(config_path), output, target_by_id, atol, rtol
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
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        atomic_json(result_path, result)
        result_queue.put((job["job_id"], result["status"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def completed_results(output_dir: Path, signature: str) -> list[dict[str, Any]]:
    results = []
    for path in sorted((output_dir / "jobs").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("evaluation_signature") == signature:
            results.append(result)
    return results


def assemble_csv_tree(output_dir: Path, results: list[dict[str, Any]]) -> Path:
    import pandas as pd

    grouped: dict[tuple[str, str | None, int, int], list[Path]] = defaultdict(list)
    for result in results:
        if result["status"] != "completed" or not result.get("frame_path"):
            continue
        grouped[(
            result["experiment"],
            result["finetune_dataset"],
            result["seed"],
            result["fold"],
        )].append(Path(result["frame_path"]))

    root = output_dir / "csv_results"
    for (experiment, dataset, seed, fold), frame_paths in grouped.items():
        frames = [pd.read_csv(path) for path in sorted(frame_paths)]
        combined = pd.concat(frames, ignore_index=True).drop_duplicates()
        group = "journal_experiments" if dataset is None else f"journal_experiments_ft_on_{dataset}"
        destination = (
            root / group / experiment / "paper_reproduction"
            / f"model_prob4_seed_{seed}_fold_{fold}.csv"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(destination, index=False)
    return root


def extract_stdout(cell: dict[str, Any]) -> str:
    pieces = []
    for output in cell.get("outputs", []):
        if output.get("output_type") != "stream" or output.get("name") != "stdout":
            continue
        value = output.get("text", "")
        pieces.extend(value if isinstance(value, list) else [value])
    return "".join(pieces).replace("\r\n", "\n").strip()


def verify_notebook(
    notebook_path: Path,
    csv_root: Path,
    output_dir: Path,
    reference_path: Path,
    timeout: int,
) -> dict[str, Any]:
    import nbformat
    from nbclient import NotebookClient

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    notebook = nbformat.read(notebook_path, as_version=4)
    old_root = os.environ.get("MIA_PAPER_RESULTS_ROOT")
    os.environ["MIA_PAPER_RESULTS_ROOT"] = str(csv_root.resolve())
    try:
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(PROJECT_ROOT)}},
        )
        client.execute(cwd=str(PROJECT_ROOT))
    finally:
        if old_root is None:
            os.environ.pop("MIA_PAPER_RESULTS_ROOT", None)
        else:
            os.environ["MIA_PAPER_RESULTS_ROOT"] = old_root

    executed = output_dir / "executed_journal_results.ipynb"
    nbformat.write(notebook, executed)
    by_id = {cell.get("id"): cell for cell in notebook.cells}
    table_results = []
    latex_dir = output_dir / "latex_tables"
    latex_dir.mkdir(parents=True, exist_ok=True)
    for expected in reference["tables"]:
        text = extract_stdout(by_id[expected["cell_id"]])
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        (latex_dir / f"table_{expected['table']}.tex").write_text(text + "\n", encoding="utf-8")
        table_results.append(
            {
                **expected,
                "actual_sha256": digest,
                "exact_match": digest == expected["sha256"],
            }
        )
    result = {
        "status": "exact_match" if all(row["exact_match"] for row in table_results) else "mismatch",
        "executed_notebook": str(executed.resolve()),
        "tables": table_results,
    }
    atomic_json(output_dir / "notebook_verification.json", result)
    return result


def summarize(
    output_dir: Path,
    targets: list[dict[str, Any]],
    initial_status: dict[str, str],
    results: list[dict[str, Any]],
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    targets_by_id = {target["target_id"]: target for target in targets}
    target_rows: dict[str, dict[str, Any]] = {
        target["target_id"]: {
            "target_id": target["target_id"],
            "experiment": target["experiment"],
            "dataset": target["dataset"],
            "seed": target["seed"],
            "fold": target["fold"],
            "log_type": target["log_type"],
            "tables": ",".join(target["tables"]),
            "status": initial_status.get(target["target_id"], "not_run"),
            "max_abs_error": None,
            "error": "",
        }
        for target in targets
    }
    for result in results:
        if result["status"] == "failed":
            for target_id in result["target_ids"]:
                target_rows[target_id]["status"] = "job_failed"
                target_rows[target_id]["error"] = result.get("error", "")
            continue
        for comparison in result.get("comparisons", []):
            row = target_rows[comparison["target_id"]]
            row["status"] = comparison["status"]
            row["max_abs_error"] = comparison["max_abs_error"]
            row["error"] = ",".join(comparison["missing_metrics"])
    rows = sorted(target_rows.values(), key=lambda row: row["target_id"])
    counts = Counter(row["status"] for row in rows)
    metric_rows = []
    comparisons = {
        comparison["target_id"]: comparison
        for result in results
        for comparison in result.get("comparisons", [])
    }
    for target_id, target in targets_by_id.items():
        target_status_value = target_rows[target_id]["status"]
        comparison = comparisons.get(target_id)
        for group, expected in target["metrics"].items():
            observed = comparison.get("observed", {}).get(group) if comparison else None
            error = comparison.get("metric_errors", {}).get(group) if comparison else None
            exact = (
                observed is not None
                and math.isclose(observed, expected, rel_tol=rtol, abs_tol=atol)
            )
            metric_rows.append(
                {
                    "target_id": target_id,
                    "experiment": target["experiment"],
                    "dataset": target["dataset"],
                    "seed": target["seed"],
                    "fold": target["fold"],
                    "log_type": target["log_type"],
                    "group": group,
                    "expected": expected,
                    "observed": observed,
                    "abs_error": error,
                    "status": "exact_match" if exact else target_status_value,
                }
            )
    metric_counts = Counter(row["status"] for row in metric_rows)
    summary = {
        "paper_targets": len(rows),
        "target_statuses": dict(sorted(counts.items())),
        "all_targets_exact": counts.get("exact_match", 0) == len(rows),
        "paper_scalar_values": len(metric_rows),
        "scalar_statuses": dict(sorted(metric_counts.items())),
        "all_scalar_values_exact": metric_counts.get("exact_match", 0) == len(metric_rows),
        "completed_jobs": sum(result["status"] == "completed" for result in results),
        "failed_jobs": sum(result["status"] == "failed" for result in results),
    }
    write_csv(output_dir / "target_results.csv", rows)
    write_csv(output_dir / "metric_results.csv", metric_rows)
    return summary, rows


def main() -> None:
    args = parse_args()
    add_import_paths()
    from identify_paper_models import configure_dataset_cache, data_preflight

    args.weights_dir = args.weights_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    release_path, release, targets_document, target_by_id, recipes = load_documents(args)
    validate_checkpoints(args.weights_dir, release)
    atol = args.atol if args.atol is not None else targets_document["comparison"]["default_atol"]
    rtol = args.rtol if args.rtol is not None else targets_document["comparison"]["rtol"]
    experiments = set(args.experiments) if args.experiments else None
    jobs, target_status = build_jobs(
        args.weights_dir, target_by_id, recipes, args.retrain_missing, experiments
    )

    inventory = [
        {
            "status": "evaluable",
            "routed_experiment": job["experiment"],
            "finetune_dataset": job["finetune_dataset"],
        }
        for job in jobs
    ]
    cache_identity = configure_dataset_cache(args.dataset_cache, inventory, args.output_dir)
    signature_payload = {
        "schema": SCHEMA_VERSION,
        "release_manifest": sha256(release_path),
        "targets": sha256(args.targets),
        "config": sha256(args.config),
        "cache": cache_identity["signature"],
        "atol": atol,
        "rtol": rtol,
        "retrain_missing": args.retrain_missing,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    for job in jobs:
        identity = {
            key: job[key]
            for key in ("kind", "experiment", "seed", "fold", "checkpoint", "target_ids")
        }
        identity["evaluation_signature"] = signature
        job["evaluation_signature"] = signature
        job["job_id"] = hashlib.sha1(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    existing = {
        result["job_id"]: result
        for result in completed_results(args.output_dir, signature)
        if result["status"] == "completed" and not args.rerun
    }
    pending = [job for job in jobs if job["job_id"] not in existing]
    print(
        f"Planned {len(jobs)} jobs ({len(existing)} reusable, {len(pending)} pending); "
        f"{Counter(job['kind'] for job in jobs)}",
        flush=True,
    )
    print(f"Targets unavailable before execution: {Counter(target_status.values())}", flush=True)
    if args.dry_run:
        return
    if not args.skip_data_preflight:
        data_preflight(jobs)

    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required")
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
                    str(args.output_dir),
                    str(args.targets.resolve()),
                    atol,
                    rtol,
                ),
            )
            for gpu in gpu_ids
        ]
        for worker in workers:
            worker.start()
        for job in pending:
            job_queue.put(job)
        for _ in workers:
            job_queue.put(None)
        finished = 0
        while finished < len(pending):
            try:
                job_id, status = result_queue.get(timeout=30)
            except queue.Empty:
                dead = [worker for worker in workers if not worker.is_alive() and worker.exitcode]
                if dead:
                    raise RuntimeError(f"GPU worker exited unexpectedly: {[worker.exitcode for worker in dead]}")
                continue
            finished += 1
            print(f"Finished {finished}/{len(pending)}: {job_id} ({status})", flush=True)
        for worker in workers:
            worker.join()

    results = [
        result for result in completed_results(args.output_dir, signature)
        if result["job_id"] in {job["job_id"] for job in jobs}
    ]
    csv_root = assemble_csv_tree(args.output_dir, results)
    summary, _ = summarize(
        args.output_dir,
        targets_document["targets"],
        target_status,
        results,
        atol,
        rtol,
    )
    summary["evaluation_signature"] = signature
    summary["csv_results_root"] = str(csv_root.resolve())
    summary["notebook"] = {"status": "skipped"}
    if summary["all_targets_exact"] and not args.skip_notebook:
        summary["notebook"] = verify_notebook(
            args.notebook.resolve(),
            csv_root,
            args.output_dir,
            args.latex_reference.resolve(),
            args.notebook_timeout,
        )
    elif not args.skip_notebook:
        summary["notebook"] = {
            "status": "blocked_by_non_exact_csv_targets",
            "reason": "The notebook is executed only after every archived scalar matches.",
        }
    summary["complete"] = (
        summary["all_targets_exact"]
        and (args.skip_notebook or summary["notebook"]["status"] == "exact_match")
    )
    atomic_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
