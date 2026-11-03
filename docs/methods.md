# Methods and metric contracts

## Executed selectors

The MCQA head-control and Refusal selectors fit raw activation contrasts by coefficient ridge, lambda 1, without an independent energy constraint or off-target penalty. Module MCQA nominal reconstruction uses lambda 1e-5; module held-out reuse divides the dictionary and displacement by sqrt(nrow), then uses lambda 1. IOI and Sentiment retain their recorded raw-dictionary settings.

Executed OMP ranks squared residual correlations divided by column squared norm and candidate cost. Its current-support stopping rule differs from the formal trial rollback. Executed Coverage uses absolute-target row weights, soft saturation, no signed split and `max(abs(q), 1e-6)` normalization. The formal reference algorithms are specified in Appendix C; this package retains the executed empirical kernels.

The nominal producer's execution-context kernels preserve the order of method calls required by the registered pipeline. The public paper comparisons are Driver-OMP, Coverage-Driver, Top-k, Random and Nonlinear greedy. Nonlinear greedy shortlists up to 32 candidates and selects by measured nonlinear reconstruction after ridge refitting; it is not a global bound.

`DriverResult.selected_indices` indexes the returned, possibly pruned `drivers` in the candidate library, in the same order as `strengths`. The module-reuse `build_local_tables` interface raises `NotImplementedError` for a supplied off-target batch; the registered experiments do not use that branch.

## Direction and information conditions

P means evaluation-pair activation contrasts; F means fitting-split mean contrasts. Direction mode is independent of coefficient fitting and dose updates. Feedback holds support, base coefficients and direction rule fixed, and adjusts dose using observed target error. Fully frozen open-loop reuse fixes dose as well. Sentiment feedback is not evaluated.

MCQA retains the original reference answer index after exchanging correct and seeded incorrect options. Module nominal results use four option logits; head control uses correct-minus-mean-wrong margin. IOI retains clean-member IO and subject token references on both pair members. Refusal forward direction is harmful-minus-harmless; reversal applies negative coefficients to the harmful target member.

## Metrics

- `rel_tar`: norm(target - patched) / (norm(target - source) + 1e-6), calculated on each complete readout vector before across-seed aggregation.
- `r_clean`: norm(target - patched) / max(norm(target), 1e-12); it has a different denominator and is not rel_tar.
- Reversal ratio: distance from reversed output to target divided by distance to source. Positive infinity and 0/0 are represented explicitly.
- Dose response: signed Spearman over the declared fixed grid. Ties use average ranks; a constant curve is undefined. Negative correlation is not counted as positive dose ordering.
- Feedback: terminal endpoint error, with initial hits and recorded readout-evaluation counts reported separately. Counts include the initial zero-dose readout and final endpoint readout; zero-dose calls may reuse a cached unpatched baseline, so these are not necessarily fresh model forward passes. MCQA/Refusal use fixed gains without clipping or gain halving. IOI retains its own controller.
- Preservation: full-vocabulary KL(unpatched || patched), in nats. P95 is nearest rank; at 256 observations it is the 244th sorted value. Seed-mean and all-prompt pass counts are separate. The paired-position IOI delta-KL and unpaired Sentiment dose KL retain their own panel and reference definitions.
- Reuse: MCQA signed mean-margin recovery excludes absolute mean gaps below 1e-9; Sentiment uses max(mean target gap, 1e-6); IOI uses one minus mean pointwise clean-normalized absolute error with floor 1e-6. They are not interchangeable with one minus rel_tar.

Sample standard deviations use ddof 1. Threshold decisions use unrounded values. Formal thresholds are operating choices, and the six-axis metrics do not by themselves establish the joint nonlinear target/preservation/energy certificate. The reported six-axis results do not assert that this joint certificate was evaluated.
