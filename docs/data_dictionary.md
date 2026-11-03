# Data dictionary

The directories separate inputs, interventions, measurements and generated summaries. JSONL preserves full floating-point precision. Every normalized measurement has a unique `record_id` and `experiment_id`; the original numerical and experimental fields are retained.

`data/inputs/<series>/manifest.json` lists the available files and identifies missing token-batch records. `batches.jsonl` retains the actual recorded tokens, masks and positions when available. Pair manifests preserve ordered identities. `preservation/panel.json` contains the shared 32 MMLU prompts, dataset row identities and content hashes. Task modules preserve paired templates, class pools and first-subtoken word sets. Reconstructing a batch from these assets is a new tokenization operation, not a replacement for an absent saved batch record.

`data/interventions/<series>/interventions.jsonl` records ordered candidates, coefficients, actual support size, seed, method, direction condition and nominal dose. A candidate label encodes basis, layer, position rule and optional head. Both coefficient conditions are included for module/IOI reuse. Distinct fits with otherwise similar labels are retained as distinct IDs. `direction_array_ref: null` denotes no saved activation-direction tensor; the task constructor and explicit model profile provide the reconstruction route.

`data/measurements/<series>/` includes complete vectors, dose points, terminal endpoints and trajectories, prompt KL, reuse summaries and generation tokens according to the series. Fields named `seed` are experimental replicate identifiers; prompts and multiple endpoints within one seed are not independent seed replicates. Paired Sentiment fixed-dose records each contain both P and F outcomes: 96 rows represent 192 conditions.

`formal_reachability/readouts.jsonl` contains all 272 source/target/forward vectors and their intervention IDs. The same task-specific records also occur in the corresponding experiment directory for direct model evaluation. Those copies are not additional replicates. Run metadata adjacent to these vectors applies only to the indicated nominal measurement files, not to other feedback, dose or reversal records in the directory.

Preservation records retain measured scalar KL values and any recorded array identifiers as provenance. Full-vocabulary arrays are not included. Replay verifies scalar summaries only.

The 16-token Refusal continuations retain token IDs, text and lexical readouts. They do not provide semantic refusal annotations or completed final answers. The unpaired Sentiment curves remain distinct from paired experiments and do not have bound fitted controls in this distribution.

`results/sentiment_displacement_sensitivity/` reports the Appendix E.3 check on the paired Sentiment evaluation vectors. It excludes pairs with exactly equal saved source and target readouts, retains the same interventions, and recomputes the complete-vector residual with stabilizer 1e-6. The 48 per-seed rows and six condition summaries report both residuals and pass counts at epsilon 0.4. Five of the sixteen source/target template pairs have identical strings; the seeded evaluation halves contain two or three such pairs. This analysis does not create new experimental trials.

Feedback fields `iter_n_evals` (IOI) and `n_evals` (MCQA/Refusal) count recorded readout evaluations, including the initial zero-dose and final endpoint readouts. A zero-dose call may return the cached baseline; these counts are distinct from controller updates and fresh model forward passes. Table G.8 includes all 24 IOI endpoints and the MCQA/Refusal summaries.
