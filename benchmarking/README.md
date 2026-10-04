# Benchmarking

This directory contains the complete workflow for the runtime and peak-memory
figures in the journal paper.

## Workflow

1. `benchmark_collation_models.py` runs the two benchmark experiments used by
   the plotting notebook:
   - experiment 1 writes `experiment_1_natural_datasets.csv`;
   - experiment 2 writes `experiment_2_subsampled_annotators.csv`.
2. `compare_models.ipynb` reads those CSV files and writes:
   - `forward_time_ms_comparison.pdf`;
   - `forward_memory_peak_mb_comparison.pdf`.

The checked-in CSV files allow the plotting notebook to be rerun without the
original datasets or a GPU. The benchmark generator itself requires the
datasets configured for the main experiments and is intended to run on a CUDA
machine.

## Run

From the repository root, generate both CSV files with:

```bash
python benchmarking/benchmark_collation_models.py
```

To run only one experiment:

```bash
python benchmarking/benchmark_collation_models.py --experiment 1
python benchmarking/benchmark_collation_models.py --experiment 2
```

Then execute `benchmarking/compare_models.ipynb`. The notebook supports being
run either from the repository root or from the `benchmarking` directory.
