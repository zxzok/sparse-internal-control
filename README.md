# Sparse Internal Control of Language Models

[English](README.md) · [中文](README.zh-CN.md) · [Reproduction scope](REPRODUCIBILITY.md) · [Experiment guide](docs/experiments.md)

Code and measurement records for **Sparse Internal Control of Language Models** by **Xizhe Zhang** (2026).

The project studies sparse activation interventions in frozen language models. It includes Driver-OMP, Coverage-Driver and the comparison methods used in the paper, with task-specific evaluations of reachability, signed reversal, dose response, feedback controllability, off-target preservation and held-out reuse.

The main task instances are IOI and Sentiment on GPT-2 small, MMLU Humanities and STEM on Qwen3-4B, and Refusal on Qwen3-1.7B. Additional model comparisons are indexed in the [experiment guide](docs/experiments.md).

## Included material

| Material | Contents |
|---|---|
| Experimental code | Selection methods, activation adapters, task construction, model runners and metrics |
| Configurations | 21 experimental series, model profiles and available input/control identities |
| Formal reachability | 272 complete source/target/patched task-readout triples, 39 groups and 195 threshold classifications |
| Preservation | 3,072 prompt–seed KL measurements |
| Result calculations | 13 numerical tables, three definition/configuration tables, threshold sensitivity and supporting statistics |

The measurements underlying the paper's figures are indexed in [experiments/figure_records.json](experiments/figure_records.json). Plotting code, rendered figures, model weights and full-vocabulary logit arrays are not included.

## Quick start: recalculate tables and statistics

Use Python 3.12. These calculations run on CPU without loading a model or requiring a GPU. Keep environments and generated outputs outside the checkout.

```bash
git clone https://github.com/zxzok/sparse-internal-control.git
cd sparse-internal-control

python3.12 -m venv ../sic-replay-env
source ../sic-replay-env/bin/activate
python -m pip install -r requirements/replay.lock.txt
python -m pip install --no-deps -e .

python -m sparse_control check-release --manifest
python -m sparse_control reproduce --output ../paper-replay
python -m sparse_control verify --output ../paper-replay
python -m pytest -q
```

On Windows, activate the environment with `..\sic-replay-env\Scripts\Activate.ps1` in PowerShell.

Calculations use the Python standard library; the lock file installs test dependencies. Model-dependent tests skip when their dependencies or explicit model fixture are absent. GitHub Actions runs this CPU-only record-replay path.

Generated outputs mirror [results/](results/). The `verify` command checks all requested CSV, JSON and TeX files, compares 161 numerical rows and 15 supporting values with independently transcribed paper values, and checks formal residuals and threshold classifications with a separate arithmetic implementation. Its reported scope is `tabular_saved_records`.

For a subset, pass the same artifact IDs to both commands:

```bash
python -m sparse_control reproduce --artifacts formal_reachability_samepanel,formal_reachability_heldout --output ../reachability-replay
python -m sparse_control verify --artifacts formal_reachability_samepanel,formal_reachability_heldout --output ../reachability-replay
```

Use a new output directory for each run. Missing files, changed values and mismatched verification scopes cause failure.

## Running model experiments

Model execution uses a separate environment and an explicit [execution profile](experiments/execution_profiles.json). See [experiment recipes](docs/experiments.md) for registered stages and [compute resources](docs/compute_resources.md) for software and hardware details.

For example, a CUDA head-level run uses:

```bash
python3.12 -m venv ../sic-model-env
source ../sic-model-env/bin/activate
python -m pip install -r requirements/model-cuda_head_fp32.lock.txt
python -m pip install --no-deps -e .

python -m sparse_control run --experiment mcqa_humanities_head --stage nominal --profile cuda_head_fp32 --seeds 0 --plan
python -m sparse_control run --experiment mcqa_humanities_head --stage nominal --profile cuda_head_fp32 --seeds 0 --output ../humanities-fixed
```

The first run command prints the resolved plan without loading a model. The second evaluates saved supports and coefficients at nominal dose, checking stored inputs and available source/target baselines first. Model assets must be available locally or accessible through their listed Hugging Face repositories.

Fixed-intervention evaluation and full fitting are separate workflows. `--stage full` runs the registered fitting producer; `--stage fit` aliases that full sequence. Fresh outputs do not replace the supplied paper measurements.

## Interpreting the results

- **Normalization:** `Rel_tar` is normalized by source-to-target displacement; `R_clean` is normalized by clean target output. Table 2 reports same-panel `Rel_tar(1)` from `formal_reachability_samepanel`; the supplemental `nominal_reconstruction` table reports `R_clean`. The two metrics are not interchangeable.
- **Direction conditions:** evaluation-pair directions (P) use activation contrasts from evaluation pairs; training-mean directions (F) use the fitting split. Feedback can adapt dose even with fixed F directions.
- **Sentiment:** unpaired dose response and paired fixed-dose reuse are separate experiments; feedback is not separately evaluated.
- **Preservation:** the reported statistics are reproduced from supplied prompt-level KL measurements. Recomputing these measurements from model outputs requires running the preservation experiments.
- **Scope:** record replay checks calculations from supplied measurements; it does not establish full model replication. Missing original input or execution identities are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Repository guide

| Path | Purpose |
|---|---|
| [configs/](configs/) | Per-series settings and experiment parameters |
| [experiments/](experiments/) | Experiment, artifact, model and data registries |
| [src/sparse_control/](src/sparse_control/) | Selection, adapters, tasks, metrics, runners and reporting |
| [data/](data/) | Evaluation inputs, saved interventions and measurements |
| [results/](results/) | Supplied tables, statistics and comparison targets |
| [tests/](tests/) | Metric, input/profile, output coverage and corruption checks |
| [docs/](docs/) | Methods, data dictionary, recipes, resources and attribution |

## Citation

The author of the accompanying paper is Xizhe Zhang. To cite this software release:

```bibtex
@software{zhang2026sparseinternalcontrol,
  author  = {Zhang, Xizhe},
  title   = {Sparse Internal Control of Language Models},
  year    = {2026},
  version = {1.2.0},
  url     = {https://github.com/zxzok/sparse-internal-control}
}
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

## License and questions

The code is licensed under the [Apache License, Version 2.0](LICENSE); [NOTICE](NOTICE) records the copyright and relicensing history (versions up to 1.1.0 were MIT), and [PATENTS.md](PATENTS.md) states the copyright holder's patent commitments. Third-party datasets and models retain their respective licenses; see [attribution](docs/third_party_assets.md).

For reproducibility questions or bugs, [open an issue](https://github.com/zxzok/sparse-internal-control/issues) with the experiment ID, command, software profile and relevant error or numerical difference.
