# Ensembling, stacking and mixture-of-experts

Goal: beat **BASELINE B = 0.3005 +/- 0.0067** (50/50 blend of z-scored per-country ridge and
z-scored TabPFN v3, pooled out-of-fold Spearman over 15 randomised grouped-CV seeds).

Reference points reproduced exactly by this work:

| model | pooled OOF Spearman (15 seeds) |
|---|---|
| BASELINE A - per-country `RidgeCV`, rank target, FR=3 feats | 0.2909 +/- 0.0069 |
| BASELINE B - 0.5 z(ridge) + 0.5 z(TabPFN v3) | 0.3006 +/- 0.0067 |

## Evaluation protocol

Everything below is scored on **identical fold assignments** (`P.make_folds(seed=0..14)`),
so every comparison is paired: `delta` is the mean per-seed difference against BASELINE B,
`sd` its standard deviation, `w` the number of seeds where the candidate won, `t` the paired
t-statistic. Marginal noise is +/-0.007; paired noise is far smaller, so a delta of +0.002 with
15/15 wins is real but negligible, and a delta of -0.010 with 0/15 is a real loss.

**Nesting.** Every level-2 model, gate and blend weight is fitted *inside* the outer fold, on
level-1 predictions that are themselves out-of-fold with respect to the level-2 training rows.
The trick used to make that affordable: for a 5-fold partition, the honest predictions that
outer fold `k` needs on its 4 training folds come from models trained on the complement of
`{j,k}`. That set is symmetric in `j` and `k`, so only `C(5,2)=10` extra level-1 fits per
country per seed are needed, each serving two outer folds.

Sanity check on leakage: no stacked configuration scored above 0.302. A jump to ~0.35+ would
have meant the level-2 fit was seeing its own validation rows.

## 1. Two structural findings that constrain every ensemble

### 1a. The pooled metric already gets the FR/DE calibration right - do not touch it

The metric ranks FR and DE rows together, so the *relative* spread of the two countries'
predictions matters. The two countries are genuinely different: as a fraction of the pooled
target rank, FR has sd 0.266 and DE 0.316 - DE occupies the extremes, FR crowds the middle.

The raw per-country ridge predictions have sd 0.056 (FR) and 0.119 (DE), a ratio of **2.14**.
Sweeping an oracle rescaling of the DE block (chosen on the full OOF, so an upper bound):

| DE scale on per-country z-scored predictions | pooled Spearman | paired vs raw |
|---|---|---|
| 0.6 | 0.2660 | -0.0249 |
| 1.0 (per-country z-scoring) | 0.2815 | -0.0094 |
| 1.5 | 0.2886 | -0.0023 |
| 2.0 | 0.2906 | -0.0003 |
| 3.0 | 0.2900 | -0.0009 |
| **raw (ratio 2.14, no rescaling)** | **0.2909** | **0** |

The optimum *is* the raw scale. There is zero headroom in cross-country calibration, and any
per-country renormalisation costs about 0.009. Confirmed independently: mapping each country's
prediction rank onto that country's empirical quantile of the pooled target rank (a fold-safe,
very-low-capacity calibration) scores 0.2825 vs 0.2909, **paired -0.0084, 0/15 wins, t = -12.1**.

Consequence: every blend must be normalised **pooled**, never per country. This single rule
explains most of the losses in the results table.

### 1b. Cross-country transfer as a feature loses

FR and DE rows of a day are bit-identical in the features, so the DE model can be evaluated on
an FR row. Feeding the (nested, inner-OOF) DE-model prediction to the FR model as a 4th feature
is a very cheap form of transfer - a supervised 1-D compression of all 29 features for a country
that can only afford 3.

| variant (ridge only, 15 seeds, own paired control 0.2919) | score | paired | wins |
|---|---|---|---|
| FR model gets the DE model's prediction | 0.2890 | -0.0029 | 0/15 (t = -7.3) |
| both countries get the other's prediction | 0.2890 | -0.0029 | 0/15 (t = -8.1) |

## 2. Regime analysis: where does each model win?

Per-row error is `|pooled rank of prediction - pooled rank of TARGET|`, averaged over the 15
fold seeds. Lower is better; `v3_adv = err(ridge) - err(v3)`, so positive means TabPFN wins.

