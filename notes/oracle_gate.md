# Why the oracle gate is unlearnable

An oracle that picks, per row, whichever of two models is closer to the truth scores far
above either model or their blend. This documents why that headroom cannot be claimed, so
nobody spends another day on it.

All numbers: pooled OOF Spearman, 10 randomised day-grouped fold seeds, ridge and LightGBM
on the full time-ordered feature set.

## The headroom is real, and partly transferable

| | score | gain over blend |
|---|---|---|
| ridge | 0.5548 | |
| LightGBM | 0.5363 | |
| blend 0.7/0.3 | 0.5654 | — |
| oracle, labels from the *same* seed | 0.6826 | +0.1172 |
| oracle, labels from a *different* seed | 0.6269 | **+0.0615** |
| oracle, majority label across seeds | 0.6533 | +0.0879 |

Foreign-seed labels still gain +0.06, so roughly half the oracle is stable row-level
structure rather than fold noise. Label stability across seeds is 0.586 where independent
coin flips would give ~0.246.

## It is not the obvious artifact

The first suspicion is that picking the better model per row is really selecting on the
target. It is not:

| oracle label correlates with | rho |
|---|---|
| TARGET | +0.004 |
| target rank | +0.004 |
| \|TARGET\| | +0.051 |
| \|ridge prediction\| | +0.141 |
| \|model disagreement\| | −0.100 |
| is_FR | −0.085 |

## Everything that tried to learn it lost

| gate | gain | wins |
|---|---|---|
| logistic on features + predictions (nested) | −0.0033 soft, −0.0225 hard | 0/10 |
| HistGB depth 2 (nested) | −0.0037 soft, −0.0257 hard | 0/10 |
| time-neighbour label borrowing, k=3 | −0.0084 | 0/10 |
| time-neighbour label borrowing, k=20 | −0.0024 | 2/10 |
| error-magnitude regression, inverse weighting | −0.0036 to −0.0087 | 0–1/10 |

Gate classification accuracy is **0.506–0.514** — chance. The label also does not cluster in
recovered time (lag-1 autocorrelation +0.030 FR, +0.003 DE, against a shuffled control of
−0.024), which kills the otherwise promising idea of borrowing labels from temporally
adjacent training days.

## The reason

Two measurements together explain it:

- a model's own rank error **is** partly predictable: rho = **+0.220**
- the two models' errors correlate at rho = **+0.613**

So we can identify where both models struggle, but that is useless for gating, because they
struggle in the *same places*. A gate needs the *difference* in errors, and once the shared
component is removed the residual is dominated by model-specific variation that is not a
tractable function of the 1494 rows available.

The structure is real and stable per row, but it is effectively the difference of two fitted
error surfaces — a legitimate function of X, just far too high-frequency to estimate at this
sample size. More data would be needed, not a better gate.

## Related: is there anything left for a time-series foundation model?

No. With the ordering recovered, the target's own temporal structure is:

    FR: L1 −0.026  L2 −0.003  L3 +0.036  L5 −0.034  L10 −0.032
    DE: L1 −0.211  L2 −0.112  L3 −0.019  L5 +0.001  L10 −0.032

The entire signal is two lags of mild mean reversion in Germany, and nothing in France.
Explicit target lags capture it (+0.0029, 13/15) and neighbour features already capture most
of the same thing. A foundation model pretrained on billions of series would recover the same
AR(2) and no more — there is nothing else in the series for it to find.
