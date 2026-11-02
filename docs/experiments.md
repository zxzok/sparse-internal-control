# Experiments

Every configuration lists its seeds, input and intervention paths, readout, ridge/normalization, available stages and explicit execution profiles. A plan can be inspected without loading a model: `python -m sparse_control run --experiment <id> --stage <listed-stage> --profile <compatible-profile> --plan`. Replace the angle-bracket arguments with registered values from the table/configuration.

| Experiment | Model and panel | Registered stages |
|---|---|---|
| `ioi_nominal` | GPT-2 small; 64 paired prompts; fitting/evaluation same panel | full, fit, nominal, reversal |
| `mcqa_humanities_nominal` | Qwen3-4B; 8 Humanities questions; same panel | full, fit, nominal |
| `mcqa_stem_nominal` | Qwen3-4B; 8 STEM questions; same panel | full, fit, nominal |
| `mcqa_humanities_head` | Qwen3-4B; 24 Humanities questions, fit/eval=12/12 | full, fit, nominal, dose, reversal, feedback |
| `mcqa_stem_head` | Qwen3-4B; 24 STEM questions, fit/eval=12/12 | full, fit, nominal, dose, reversal, feedback |
| `sentiment_paired` | GPT-2 small; 16 pairs, fit/eval=8/8 | full, fit, nominal, reversal |
| `refusal_head` | Qwen3-1.7B; 20 pairs, fit/eval=10/10 | full, fit, nominal, dose, reversal, feedback |
| `ioi_feedback` | GPT-2 small; 64 paired prompts; same panel | full, fit |
| `ioi_dose` | GPT-2 small; 64 paired prompts; same panel | full, fit |
| `sentiment_unpaired_dose` | GPT-2 small; 16 positive and 16 negative templates; 8 fit/8 eval per class | full, fit |
| `preservation` | task-specific models; fixed 32-prompt MMLU panel: 16 Humanities+16 STEM | preservation |
| `mcqa_humanities_reuse` | Qwen3-4B; 32 Humanities questions, fit/eval=16/16 | full, fit |
| `mcqa_stem_reuse` | Qwen3-4B; 32 STEM questions, fit/eval=16/16 | full, fit |
| `ioi_reuse` | GPT-2 small; split and exact IDs from recorded IOI reuse inputs | full, fit |
| `ioi_recurrence` | GPT-2 small; recorded IOI recurrence panel | full, fit |
| `ioi_backup` | GPT-2 small; separate 64-prompt panel | full, fit |
| `model_qwen3_8b` | Qwen3-8B; 4 Humanities questions; same panel | full, fit |
| `model_mistral7b` | Mistral-7B; 4 Humanities questions; same panel | full, fit |
| `refusal_generation` | Qwen3-1.7B; 10 harmful prompts/seed; 20 unique prompts reused | generation |
| `refusal_single_site` | Qwen3-1.7B; same Refusal control panels; independent preservation panel | dose, reversal, feedback, preservation |
| `formal_reachability` | task-specific models; the seven applicable nominal/held-out series | record replay |

`nominal`, `dose`, `reversal`, and head-level `feedback` stages read fixed saved interventions. Full IOI dose and IOI feedback use their own producer sequence because their saved trajectories do not include every fitted control tensor. `preservation` uses the shared 32-prompt panel; `--task` selects one task or omitting it selects all four. `generation` reads saved Refusal controls and performs exactly 16 single-prompt greedy steps with no EOS early stop. The L15.H0 single-site reference has its own fixed-control evaluation stages; it is not a standard residual-stream ActAdd implementation.

Full producers are invoked with the complete registered argument sequence, including any intermediate budget calls. Output contains their native measurement records and resolved configuration. Native producer outputs are not automatically substituted for the distributed interventions; any explicit --interventions file must follow the documented intervention schema and match the intended input panel. Fixed-intervention outputs additionally contain full source/target/patched readouts, input batches and reconstructed F direction arrays. The nominal `--smoke` path is intentionally restricted to one seed, two examples, eight candidates and budget one for adapter validation.

Model runs use new output directories. The delivered `results/` are calculated from the distributed saved records, never silently replaced by fresh model outputs. Compare matching experiment/seed/method/direction/panel identities before interpreting numerical differences.

Tabular mappings are in `experiments/artifacts.json`; measurement-only figure mappings are in `experiments/figure_records.json`. Table 2 remains R_clean; the formal target-residual tables are `formal_reachability_samepanel` and `formal_reachability_heldout`. Updating a display number does not change an artifact ID or metric definition.

Fixed input checks compare stored fields before controls, including order, tokens, masks, positions and answer IDs. Available baseline vectors use 1e-4 absolute tolerance. Series without saved token batches are marked as reconstructed. External `--interventions` is accepted only for fixed nominal/dose/reversal/feedback stages; unsupported combinations are rejected.
