# Modeling Individual Annotators journal experiments

This repository contains the experiment and analysis code accompanying the
[journal paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13593162/).

## Setup

The experiments require Python, PyTorch, CUDA-capable hardware, and local copies
of MSP-Podcast, MSP-Improv, IEMOCAP, and MuSE. Install the Python environment and
the local `MIA` package from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e src
```

Edit `data_config.yaml` so the four `*_directory` entries and
`cache_dataset_path` point to your local datasets and cache. If running a script
from inside `run_experiments/`, update the copy of `data_config.yaml` in that
directory instead. MuSE experiments additionally require an authorized local
copy of the restricted annotator metadata; set `muse_annotator_info_path` to that
file. The metadata is intentionally excluded from this repository and must not
be committed.

Pretrained model weights are available from the
[model-weight archive](https://drive.google.com/drive/folders/1iq6bfHt9C6cZaJt29TCAO3thpCB7OKdO?usp=sharing).
Place downloaded experiment outputs under `.output_files/` when reproducing an
analysis from saved checkpoints or logs.

## Experiments

The retained entry points reproduce the journal experiment families:

```bash
python3 run_experiments/run_journal_experiments_new.py
python3 run_experiments/run_journal_experiments_new_finetune.py
```

Experiment definitions are in
`run_experiments/configs/journal_experiments.yaml`. Each runner contains an
`experiments_to_run` list that selects the desired definitions. Outputs are
written to `.output_files/journal_experiments/` by default.

## Analysis

- `journal_results.ipynb` produces the paper-format LaTeX for Tables II–VIII
  from saved experiment CSVs in `.output_files/csv_results/`.
- `journal_new_figure.ipynb` produces the individual-annotator analysis figure
  from the experiment CSV outputs and the cached MSP-Podcast parquet datasets.
- `benchmarking/compare_models.ipynb` produces the runtime and peak-memory
  figures from the checked-in benchmark CSVs. See `benchmarking/README.md` for
  the complete benchmark workflow.

Run the notebooks from the repository root so their relative `.output_files/`
paths resolve correctly.
