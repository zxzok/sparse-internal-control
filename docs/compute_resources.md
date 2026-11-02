# Compute resources and execution profiles

Tabular replay uses Python 3.12 and the standard library on CPU. The replay lock file installs only the test dependencies. Model weights are required only for explicit model execution.

`experiments/execution_profiles.json` specifies loader, dtype, tokenizer sides and named software versions. Fixed evaluation and full producers apply these parameters at model creation and record observed dtype/device/sides. A current explicit profile does not establish the original runtime identity of every saved series.

## Reported experiment hardware

The hardware assignments below are those stated in Appendix H. Device count, peak RAM/VRAM and complete campaign wall time were not recorded per series; they are `not_recorded`, not estimates.

| Experiment / stage | Hardware / backend | Device count | Precision evidence | Panel / batching | Peak memory | Campaign time |
|---|---|---|---|---|---|---|
| Humanities and STEM module nominal | Apple M2 Ultra / MPS | not_recorded | HF producer uses bf16 on MPS | 8 questions, same panel | not_recorded | not_recorded |
| Qwen3-8B and Mistral-7B model comparison | Apple M2 Ultra / MPS | not_recorded | HF producer uses bf16 on MPS | 4 questions | not_recorded | not_recorded |
| IOI reversal | NVIDIA RTX A6000 / CUDA | not_recorded | TransformerLens fp32 profile; per-run dtype not_recorded | 64 pairs | not_recorded | not_recorded |
| Sentiment reversal | NVIDIA RTX A6000 / CUDA | not_recorded | TransformerLens fp32 profile; per-run dtype not_recorded | 8 evaluation pairs | not_recorded | not_recorded |
| MCQA and Sentiment preservation | NVIDIA RTX A6000 / CUDA | not_recorded | TransformerLens fp32 profile; per-run dtype not_recorded | 32 prompts, per-prompt measurements | not_recorded | not_recorded |
| STEM independent dose | NVIDIA RTX A6000 / CUDA | not_recorded | TransformerLens fp32 profile; per-run dtype not_recorded | 12 evaluation pairs, 10 doses | not_recorded | not_recorded |
| IOI nominal, dose, feedback, reuse, recurrence and backup | not_recorded by series | not_recorded | see configurations; original dtype not_recorded | series-specific; see experiment matrix | not_recorded | not_recorded |
| Humanities/STEM head fitting and feedback | not_recorded by series | not_recorded | see configurations; original dtype not_recorded | 24 questions, fit/eval 12/12; fitting microbatch is seed-specific | not_recorded | not_recorded |
| Paired Sentiment reuse and unpaired dose | not_recorded by series | not_recorded | see configurations; original dtype not_recorded | paired: 16 pairs; unpaired: 16 prompts/class | not_recorded | not_recorded |
| Refusal control, preservation, generation and single-site comparison | not_recorded by series | not_recorded | see configurations; original dtype not_recorded | control: 20 pairs, fit/eval 10/10; generation: unpadded singleton | not_recorded | not_recorded |
| Humanities/STEM module reuse | not_recorded by series | not_recorded | HF producer uses bf16 on accelerator | 32 questions, fit/eval 16/16 | not_recorded | not_recorded |

Producer settings in the precision column describe executable configuration where no actual dtype record exists. They must not be read as recovered hardware/dtype telemetry.

## Timed fixed-control forward measurements

These times come from the distributed `data/measurements/<series>/run_metadata.json` files. They cover only the specified nominal forward measurements, excluding the original fitting campaigns. No peak-memory measurement or precise device model is stored in these metadata.

| Series | Backend | Observed dtype | Seeds | Timed measurement rows | Elapsed seconds | Phase |
|---|---|---|---|---|---|---|
| IOI nominal | MPS | float32 | 0–9 | 20 | 10.697 | fixed nominal readouts |
| Humanities module nominal | MPS | bfloat16 | 0–2 | 15 | 68.139 | fixed nominal readouts |
| STEM module nominal | MPS | bfloat16 | 0–2 | 15 | 35.862 | fixed nominal readouts |
| Humanities head | CUDA | float32 | 0–7 | 48 | 344.695 | fixed nominal readouts |
| Refusal head | MPS | float32 | 0–7 | 48 | 23.879 | fixed nominal readouts |

The stored Humanities head measurement uses singleton activation caching. Other cache microbatch settings and full-producer arguments are in each configuration. Full input batch size and activation-cache microbatch size are different quantities.

## Software and execution

MPS profiles name PyTorch 2.11.0, TransformerLens 3.0.0, Transformers 5.6.2, NumPy 2.4.4 and datasets 4.8.4. CUDA profiles name PyTorch 2.10.0, TransformerLens 3.0.0, Transformers 5.5.3, NumPy 2.3.5 and datasets 4.8.4. Matching requirement files are in `requirements/`; these pins do not claim a complete transitive lock for every saved run.

MCQA head candidate libraries contain 1,152 candidates. Memory and time depend on sequence length, precision and microbatching. No unmeasured VRAM or runtime guarantee is supplied. Use an external model environment/cache and a new output directory. Cached execution can use `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; missing assets then stop the run.
