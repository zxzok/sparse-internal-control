# Reproducibility

1. **Record replay** recalculates tables and statistics from distributed measurements, including negative outcomes. The verify command checks 272 complete source/target/patched vectors at dose one, with arithmetic implemented separately from the table generator.
2. **Fixed-intervention inference** loads pinned model assets under an explicit profile and evaluates saved supports and coefficients without selection or refitting.
3. **Full fitting** executes the registered producer, including intermediate budget/comparison calls required by its execution sequence. Only declared paper subsets are aggregated.

## Verification scope

`verify` reports `scope: tabular_saved_records`. It checks every requested CSV, JSON and TeX output, independently transcribed paper values, and formal-reachability arithmetic. A subset requires an explicit matching `--artifacts` argument.

Preservation verification aggregates 3,072 saved KL scalars; it does not recalculate logits-to-KL. Plotting code and rendered figures are absent. Saved-record verification does not establish full model replication. Full fitting campaigns have not been rerun as part of package preparation.

## Inputs and models

For five nominal-measurement series, 32 saved batches include ordered tokens, masks, readout IDs/positions and task-specific fields. Fixed evaluation reconstructs and compares every stored field. Other series report `reconstructed_without_saved_tokens`; a constructor is not a record of the original tokenization.

Available complete source/target vectors are checked before patches at absolute tolerance 1e-4 and relative tolerance zero. Current word-set token IDs are saved in run outputs; where original IDs were not stored separately, baseline agreement does not establish their original identity.

Both fixed and full producer loaders apply profile dtype, tokenizer sides and loader options. Metadata records actual parameter dtype/device, tokenizer sides, requested revision and loader-reported revision when exposed. Missing original runtime identities remain unknown.

Unpaired Sentiment curves retain measurements and a class-pool producer but have no bound saved support/strength tensors. Paired fixed-dose reuse is a separate series. Sentiment feedback is not evaluated.

## Output safety

Use an external virtual environment and new output directories. `--plan` loads no models. Do not replace paper results with smoke outputs. Model profiles pin named packages; they do not establish all original environments. See [compute resources](docs/compute_resources.md).

`release_files.txt` and `MANIFEST.sha256` cover the release tree. The external checksum covers the ZIP container. Build a separate archive with:

```bash
python -m sparse_control build-release --output ../sparse-internal-control.zip
```
