# Recovered checkpoint identification

`identify_paper_models.py` evaluates recovered `.ckpt` files against the exact
per-seed/per-fold scalar values used to build Tables II--VIII in
`journal_results.ipynb`. It uses one persistent worker on each of GPUs 0, 1,
and 2, so three checkpoints are evaluated in parallel without placing several
models on the same GPU.

The checked-in `paper_result_targets.json` contains only aggregate result
metrics. It does not contain annotations or other source data.

## Run

Use the same Python environment and data configuration as the journal
experiments:

```bash
python model_identification/identify_paper_models.py \
  --weights-dir potential_model_weights \
  --gpus 0,1,2
```

Before starting the workers, the script selects a compatible publication-era
dataset cache. It checks the cached transformer identities, all required
train/dev/test files, and the feature and five KDE-generation columns. The
selected cache is used read-only, so its historical metadata and parquet files
cannot be changed by evaluation. The configured cache and sibling journal
working copies are searched automatically. To select one explicitly:

```bash
python model_identification/identify_paper_models.py \
  --dataset-cache /path/to/publication/dataset_caches \
  --gpus 0,1,2
```

The validated cache is also treated as the authoritative source for historical
labels and paths. This avoids reconstructing MSP-Podcast 1.8 from later raw
releases, preserves the publication-era fine-tuning annotations, and avoids a
dependency on private annotator metadata such as `muse_annotators.csv`.

Journal fine-tuning uses the original publication split procedure: GroupKFold
selects the held-out speakers and a seed-0 permutation selects one quarter of
the remaining speaker groups for validation. This differs from the later split
procedure used by non-journal experiments.

The selected cache identity is recorded in `results/dataset_cache.json` and in
the evaluation signature used for resumability.

First inspect checkpoint routing without using a GPU:

```bash
python model_identification/identify_paper_models.py --dry-run
```

The dry run still reads every checkpoint on the CPU to inspect `model_args` and
calculate a SHA-256 digest, but it does not require or validate a dataset cache.
This lets the full run evaluate byte-identical recovery copies once while
reporting the result for every pathname. Pass `--no-deduplicate` to execute
identical copies separately.

The default comparison is `abs(observed - archived) <= 1e-6` with zero relative
tolerance. `--atol` and `--rtol` override this. The report also records a
three-decimal per-value comparison, but only the full-precision comparison is
called an exact match.

The run is resumable. Each completed evaluation is atomically written under
`model_identification/results/jobs/`; rerunning the command skips those jobs.
Use `--rerun` to replace them. Per-checkpoint stdout/stderr is under `logs/`.

Useful narrower runs are:

```bash
python model_identification/identify_paper_models.py \
  --experiments individual_annotator individual_annotator_100plus

python model_identification/identify_paper_models.py --limit 3
```

`--limit 3` is intended as a GPU smoke test. Do not interpret its summary as a
complete paper-model map.

## Outputs

- `checkpoint_inventory.csv`: every `.ckpt`, its saved model signature, route,
  digest, and reason if it is not a paper candidate.
- `all_candidates.csv`: final status for every pathname, including aliases of
  byte-identical files.
- `target_comparisons.csv`: every candidate-to-paper-target comparison and its
  maximum absolute error.
- `exact_matches.csv`: full-precision matches only.
- `best_candidates.csv`: closest candidate(s) for every tested target, useful
  when environment/library drift prevents a strict match.
- `experiment_model_map.json`: exact and best checkpoint path(s) for each
  experiment, seed, fold, and result role (`Test`, `Test zero-shot`, or an
  MSP-Podcast annotation-threshold test).
- `run_summary.json`: compact counts for the run.

Fine-tuned checkpoints are evaluated once on their seed/fold test set, then
compared independently with the archived zero-shot and final-model values. This
is necessary because the recovered filename may no longer reliably describe
the checkpoint role. Base MSP-Podcast checkpoints are evaluated on all three
paper test populations, including the published unseen-annotator strategy.

Files from non-paper model types (for example the loop implementation used only
for benchmarking) remain in the inventory with an explicit status; they are
not compared with a paper row.

## Paper notation

The table labels map to checkpoint experiment names as follows:

- `Consensus` = `baseline_aggregate`
- `2D KDE` = `baseline`
- `IA` = `individual_annotator`
- `IA_mu` = `individual_annotator_plus`
- `IA_sigma^2` = `individual_annotator_var`
- `IA_mu_sigma^2` = `individual_annotator_plus_var`

Suffixes such as `_100plus`, `_1000plus`, and `_podcast_1.8` identify the
additional training-data restriction or dataset version used by that row.

## Target provenance

`build_reference_manifest.py` regenerates the compact target file from the
archived CSV result tree selected by `journal_results.ipynb` (latest timestamp
for each publication experiment):

```bash
python model_identification/build_reference_manifest.py \
  /path/to/.output_files/csv_results \
  model_identification/paper_result_targets.json
```

This provenance command is not needed for normal identification.

## Minimal release checkpoint set

After checkpoint identification, inspect the provenance-tracked release plan:

```bash
python model_identification/build_release_weights.py \
  --gpus 0,1,2 \
  --dry-run
```

Then build it:

```bash
python model_identification/build_release_weights.py --gpus 0,1,2
```

The default is a recoverable copy into `paper_model_weights/checkpoints/` and
an auditable `paper_model_weights/manifest.json`. The manifest records the
source path, SHA-256, identification error, and the evaluation recipe for all
1,045 per-seed/per-fold paper targets. Mapping-derived zero-shot models are not
stored because the verifier recreates them from their base checkpoint.

The builder checks every required base experiment/seed before starting GPU
work. If a base checkpoint is absent, it recreates that one seed with the
publication configuration and historical dataset cache. This currently plans
five MSP-Podcast 1.11 and five MSP-Podcast 1.8 2D-KDE training jobs. Retrained
weights are retained even when their results drift from the archived values;
their selection is labelled `retrained_exact`, `retrained_rounded_3dp`, or
`retrained_mismatch` rather than assuming determinism.

After repairing missing models, the builder reloads **every** staged base and
fine-tuned checkpoint and compares all result populations associated with that
checkpoint. Training and verification use one persistent worker on each of
GPUs 0, 1, and 2 and are resumable under `model_identification/release_build/`.
The per-checkpoint report is `checkpoint_verification.csv`; the same status and
maximum error are embedded in the release manifest. The command exits nonzero
if any checkpoint is missing, fails to load, or does not match all associated
targets at the configured full-precision tolerance. `--no-retrain-missing` and
`--no-verify` are available for diagnostic partial builds.

Only after a complete verification should the selected files be removed from
the recovery pool:

```bash
python model_identification/build_release_weights.py --mode move
```

This explicit move mode also removes a source file when an identical staged
copy already exists. It does not delete unselected recovery files.

## End-to-end paper verification

Inspect the complete three-GPU job plan without evaluating models:

```bash
python model_identification/verify_paper_results.py \
  --gpus 0,1,2 \
  --retrain-missing \
  --dry-run
```

Run every available result, including deterministic recreation of Table VIII
fine-tuned models whose final checkpoint was not recovered:

```bash
python model_identification/verify_paper_results.py \
  --gpus 0,1,2 \
  --retrain-missing
```

There is one persistent worker per GPU and the run is resumable. Outputs are
written under `model_identification/reproduction_results/`, including:

- `target_results.csv`: exact full-precision status for every archived
  experiment/seed/fold/result-population target;
- `metric_results.csv`: expected value, observed value, and absolute error for
  every individual paper metric;
- `csv_results/`: the directory structure consumed by `journal_results.ipynb`;
- `run_summary.json`: counts of exact, mismatched, missing, and failed targets;
- `executed_journal_results.ipynb` and `latex_tables/`: created only after all
  CSV targets match; and
- `notebook_verification.json`: exact SHA-256 comparisons against the retained
  published LaTeX for Tables II--VIII.

The notebook accepts `MIA_PAPER_RESULTS_ROOT` so the verifier can execute it on
the isolated result tree. It now loads only the nine cross-corpus experiment
families actually used by Tables VII and VIII.

The recovery pool does **not** contain the journal 2D-KDE checkpoints for
`baseline` (MSP-Podcast 1.11) or `baseline_podcast_1.8`. The only routed 1.11
KDE files are benchmarking checkpoints and differ substantially from the
paper values; there is no routed 1.8 KDE file. The release builder therefore
reconstructs those ten checkpoints before the end-to-end verifier runs.
