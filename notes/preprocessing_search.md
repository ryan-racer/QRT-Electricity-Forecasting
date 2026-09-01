# Preprocessing / target / sample-weighting search

Scope: everything *except* the model class. Ridge stays fixed; what changes is how the
features are cleaned, how the target is shaped, how rows are weighted, and how the
regularisation strength is chosen.

## Protocol

* Harness: `experiments/harness.py`. `run(fit_predict, seeds)` returns one pooled OOF
  Spearman per seed; a seed randomises the DAY_ID -> fold assignment via `P.make_folds`.
* Every comparison is **paired on identical fold assignments**: variant and baseline see the
  same seeds, and the reported `delta` is the mean of per-seed differences, with `d_sd` its
  standard deviation and `se = d_sd/sqrt(n)`. `wins` counts seeds where the variant beat the
  baseline. n >= 15 everywhere, n = 30 for anything that mattered.
* Everything fitted (medians, KNN imputers, scalers, PCA, alphas, target ranks, weights) is
  fitted **inside the fold**. Variants that additionally read `X_test` are labelled
  TRANSDUCTIVE; test labels are never touched.
* **Baseline** (reproduced exactly): per-country `RidgeCV(alphas=logspace(-2,4,40))` on the
  fold-train rank-transformed target; FR = 3 features (FR_WINDPOW, GAS_RET, CARBON_RET),
  DE = all 29. **0.2909 +/- 0.0069** over seeds 0-14.

### On the noise floor

The brief says "anything inside +/-0.007 is noise". That is right for the *unpaired* number:
seed-to-seed sd of the baseline is 0.0069, so two independent 15-seed runs differ by ~0.002
on the mean by chance alone. But the paired deltas have `d_sd` of 0.001-0.005, an order of
magnitude smaller, because the fold assignment is held fixed. A +0.002 delta with `se`=0.0003
and 30/30 wins is a real effect. It is also a *small* one, and the tables below report both
the effect and its size so neither gets confused for the other.

Separately: absolute scores are not comparable across seed blocks. Seeds 0-14 give a baseline
of 0.2909, seeds 100-129 give 0.2897, seeds 200-229 give 0.2902. Only deltas within a block
mean anything.

### Guarding against CV-selection overfitting

Roughly 180 variants were scanned. Picking the max of 180 noisy numbers is optimistic, so
every candidate winner was re-run on **fresh seeds** that had not been used to select it
(100-129), and the final combination was run on a second fresh block (200-229). Fresh seeds
control for fold-assignment luck. They do **not** control for reusing the same 1494 rows, so
the headline numbers below are still upper-ish estimates; the fold-safe variant (E) is the
one with no tuned constant in it at all.

---

## Headline result

| pipeline | score | delta vs baseline | d_sd | se | wins |
|---|---|---|---|---|---|
| BASELINE (RidgeCV, 29 DE feats, rank y) | 0.2902 | - | - | - | - |
| **A+B+C+D (all four winners stacked)** | **0.3024** | **+0.0122** | 0.0047 | 0.0009 | **30/30** |
| E+B+C (fully fold-safe, nothing tuned) | 0.2991 | +0.0089 | 0.0054 | 0.0010 | 30/30 |

Seeds 200-229, never used to select any component. Components:

* **A** - replace `RidgeCV` with a **fixed `alpha=10`**
* **B** - drop the **6 weather columns** (TEMP/WIND/RAIN) from the DE feature set
* **C** - target = **centred-rank^0.5** instead of plain rank
* **D** - transductive **KNN imputation (k=5) on standardised features**
* **E** - fold-safe alternative to A: alpha chosen inside the fold by inner-CV **Spearman**

---

## 1. Regularisation strength -- the single biggest effect (`b7`, `b12`)

`RidgeCV` selects alpha by leave-one-out GCV, i.e. by **MSE on the rank target**. The metric
is Spearman. Those disagree, and the disagreement is worth ~0.006.

Diagnostic: the alpha `RidgeCV` actually picks is **median 143 for FR, 70 for DE**
(range 24-203). The Spearman-optimal alpha is around **10**.

| variant | score | delta | d_sd | se | wins | verdict |
|---|---|---|---|---|---|---|
| BASELINE RidgeCV (LOO-GCV) | 0.2893 | - | - | - | - | - |
| fixed alpha=0.1 | 0.2926 | +0.0033 | 0.0032 | 0.0006 | 25/30 | win |
| fixed alpha=1 | 0.2939 | +0.0046 | 0.0030 | 0.0006 | 28/30 | win |
| fixed alpha=3 | 0.2945 | +0.0052 | 0.0028 | 0.0005 | 29/30 | win |
| **fixed alpha=10** | **0.2948** | **+0.0056** | 0.0024 | 0.0004 | **30/30** | **best, broad plateau** |
| fixed alpha=30 | 0.2938 | +0.0045 | 0.0019 | 0.0004 | 30/30 | win |
| fixed alpha=50 | 0.2925 | +0.0033 | 0.0018 | 0.0003 | 30/30 | win |
| fixed alpha=100 | 0.2894 | +0.0001 | 0.0020 | 0.0004 | 12/30 | = where RidgeCV lands |
| fixed alpha=300 | 0.2833 | -0.0060 | 0.0028 | 0.0005 | 1/30 | loses |
| alpha FR=100, DE=10 | 0.2952 | +0.0060 | 0.0025 | 0.0005 | 30/30 | win (see below) |
| alpha FR=1, DE=100 | 0.2887 | -0.0005 | 0.0021 | 0.0004 | 9/30 | noise |
| inner-CV alpha by MSE (grouped) | 0.2897 | -0.0012 | 0.0017 | 0.0004 | 5/15 | noise/slightly worse |
| **inner-CV alpha by SPEARMAN (grouped)** | 0.2935 | **+0.0026** | 0.0029 | 0.0007 | 11/15 | **win, fold-safe** |
| alphas logspace(-4,6,60) / (0,6,40) / (1,5,40) | 0.2909 | ~0.0000 | <0.001 | - | - | grid choice is irrelevant |

Reading the FR/DE cross-grid: **the whole effect is DE's**. FR at alpha 1, 10 or 100 is
identical to 4 decimal places (FR has 3 features, it cannot overfit). DE at alpha 10 vs 100
is the entire +0.006. The 29-feature / 643-row DE model is being **over**-regularised by GCV,
not under-regularised.

Notes:
* Widening or shifting the alpha grid does nothing; the problem is the *criterion*, not the grid.
* alpha 1-50 is a broad plateau, so "10" is not a knife-edge fit. Still, it is a constant tuned
  on this CV. The honest, tuning-free version is **E** (inner-CV Spearman), which recovers
  about half the gain (+0.0026 to +0.0031) and is noisier (`d_sd` 0.003-0.004) because 643 rows
  make an inner Spearman estimate wobbly.
* With a *fixed* alpha, ridge is no longer invariant to target scale
  (`b(cy, a) = c * b(y, a/c^2)`). Any target transform must therefore be standardised to unit
  sd inside the fold before a fixed alpha means anything. `RidgeCV` is scale-invariant, so the
  baseline was unaffected; the combination runs all standardise.

## 2. Missingness / feature drop (`b6`, `b11`)

**Imputation strategy is a complete no-op.** Median / mean / pooled-across-countries / MICE
all land within +/-0.0002 of each other. This confirms and extends the existing finding.

| variant | score | delta | d_sd | se | wins | verdict |
|---|---|---|---|---|---|---|
| pooled-country median impute | 0.2910 | +0.0001 | 0.0001 | 0.0000 | 10/15 | no-op |
| pooled-country mean impute | 0.2908 | -0.0001 | 0.0002 | 0.0001 | 3/15 | no-op |
| IterativeImputer (MICE, 5 it) | 0.2908 | -0.0001 | 0.0007 | 0.0002 | 6/15 | no-op |
| IterativeImputer (MICE, 10 it) | 0.2908 | -0.0001 | 0.0007 | 0.0002 | 6/15 | no-op |

Dropping high-NaN columns from the **DE** model does help, but the mechanism is not missingness:

| variant | score | delta | d_sd | se | wins | verdict |
|---|---|---|---|---|---|---|
| DE: drop 1 highest-NaN col | 0.2907 | -0.0002 | 0.0031 | 0.0008 | 5/15 | noise |
| DE: drop 3 highest-NaN cols | 0.2942 | +0.0033 | 0.0037 | 0.0009 | 13/15 | win |
| DE: drop 5 highest-NaN cols | 0.2973 | +0.0065 | 0.0039 | 0.0010 | 14/15 | win |
| **DE: drop 6 highest-NaN cols** | **0.2992** | **+0.0083** | 0.0041 | 0.0011 | 14/15 | **win** |
| DE: drop 8 highest-NaN cols | 0.2987 | +0.0078 | 0.0045 | 0.0012 | 14/15 | win |
| DE: drop ALL 9 any-NaN cols | 0.2904 | +0.0012 | 0.0051 | 0.0009 | 19/30 | **gain collapses** |
| DE: drop the 6 weather cols by name | 0.2958 | +0.0065 | 0.0037 | 0.0007 | 29/30 | win |
| DE: drop weather + DE_NET_EXPORT | 0.2965 | +0.0072 | 0.0044 | 0.0008 | 29/30 | win |
| DE: drop NET_EXPORT cols only | 0.2916 | +0.0023 | 0.0023 | 0.0004 | 24/30 | small win |
| DE feats = DE-prefixed only (14) | 0.2986 | +0.0077 | 0.0049 | 0.0013 | 15/15 | win |
| DE feats = FR-prefixed only (12) | 0.2089 | -0.0820 | 0.0067 | 0.0017 | 0/15 | destroys it |
| DE feats = commodity only (3) | 0.1087 | -0.1822 | 0.0104 | 0.0027 | 0/15 | destroys it |
| DE feats = complete-case cols (20) | 0.2908 | -0.0000 | 0.0052 | 0.0013 | 7/15 | no-op |

Two things kill the "missingness is the problem" story:

1. **Drop-9 (every NaN-bearing column) gives +0.0012, drop-8 gives +0.0078.** The ninth column
   is `DE_FR_EXCHANGE`, which is genuinely predictive for DE. If missingness were the cause,
   dropping more NaN columns would keep helping.
2. **The tie-break is arbitrary.** The 6 highest-NaN columns are DE_NET_EXPORT plus five of the
   six weather columns, and all six weather columns share an identical 6.29% NaN rate. "Top 6"
   vs "top 7" is a coin flip that moves the score by 0.002.

**Control experiment - random 6-column drops** (8 draws, 30 seeds each):

| variant | delta | wins |
|---|---|---|
| random drop #0 | -0.0048 | 1/30 |
| random drop #1 | -0.0055 | 2/30 |
| random drop #2 | **+0.0071** | 30/30 |
| random drop #3 | -0.0128 | 0/30 |
| random drop #4 | +0.0022 | 27/30 |
| random drop #5 | -0.0030 | 4/30 |
| random drop #6 | -0.0077 | 0/30 |
| random drop #7 | -0.0035 | 3/30 |
| **mean / sd of the 8 random drops** | **-0.0035 / 0.0056** | |

The weather drop (+0.0065) sits ~1.8 sd above the random-drop mean, so it is better than
chance -- but **one random draw out of eight matched it**. The honest conclusion is: the DE
design matrix has too many columns for 643 rows, almost any well-chosen 6-column removal
helps, and "weather" is a reasonable but not uniquely correct choice.

Fold-safe in-fold selection by univariate |Spearman| does *not* reliably beat it:
drop 6 weakest = +0.0015 (19/30), drop 10 = +0.0011 (17/30), drop 15 = +0.0027 (20/30) --
all inside noise once the selection is honest.

## 3. Target variants (`b1`)

| variant | score | delta | d_sd | se | wins | verdict |
|---|---|---|---|---|---|---|
| BASELINE per-country rank | 0.2909 | - | - | - | - | - |
| **tanh-squashed robust z** | 0.2933 | **+0.0024** | 0.0014 | 0.0004 | 15/15 | small win |
| **centred-rank^0.5** | 0.2931 | **+0.0022** | 0.0013 | 0.0003 | 14/15 | small win |
| rank winsorised @0.20 | 0.2924 | +0.0015 | 0.0009 | 0.0002 | 14/15 | small win |
| rank winsorised @0.15 | 0.2923 | +0.0014 | 0.0007 | 0.0002 | 14/15 | small win |
| rank winsorised @0.10 | 0.2920 | +0.0011 | 0.0005 | 0.0001 | 15/15 | small win |
| POOLED rank (FR+DE ranked together) | 0.2922 | +0.0013 | 0.0022 | 0.0006 | 12/15 | small win |
| rank binned k=2 (median split) | 0.2922 | +0.0013 | 0.0030 | 0.0008 | 9/15 | noise |
| centred-rank^0.75 | 0.2920 | +0.0011 | 0.0005 | 0.0001 | 14/15 | small win |
| sign(y)\|y\|^0.2 | 0.2920 | +0.0012 | 0.0024 | 0.0006 | 10/15 | noise |
| rank winsorised @0.30 | 0.2916 | +0.0007 | 0.0016 | 0.0004 | 10/15 | noise |
| rank binned k=10 | 0.2915 | +0.0006 | 0.0015 | 0.0004 | 10/15 | noise |
| rank winsorised @<=0.05 | 0.2909-0.2911 | ~0 | <0.0003 | - | - | no-op |
| rank binned k>=20 | 0.2906-0.2909 | ~0 | <0.001 | - | - | no-op |
| sign(y)\|y\|^0.4 | 0.2891 | -0.0018 | 0.0028 | 0.0007 | 4/15 | loses |
| sqrt-rank (arcsine) | 0.2887 | -0.0022 | 0.0009 | 0.0002 | 0/15 | loses |
| centred-rank^1.5 | 0.2876 | -0.0032 | 0.0010 | 0.0003 | 0/15 | loses |
| raw winsorised +/-1sd | 0.2866 | -0.0043 | 0.0034 | 0.0009 | 2/15 | loses |
| gaussian rank (van der Waerden) | 0.2864 | -0.0045 | 0.0013 | 0.0003 | 0/15 | loses (confirms prior) |
| rank binned k=3 | 0.2863 | -0.0046 | 0.0023 | 0.0006 | 0/15 | loses |
| sign(y)\|y\|^0.5 | 0.2863 | -0.0046 | 0.0034 | 0.0009 | 2/15 | loses |
| centred-rank^2 | 0.2845 | -0.0064 | 0.0018 | 0.0005 | 0/15 | loses |
| logit rank | 0.2843 | -0.0066 | 0.0021 | 0.0005 | 0/15 | loses |
| sign(y)\|y\|^0.7 | 0.2779 | -0.0130 | 0.0056 | 0.0014 | 0/15 | loses |
| raw winsorised +/-2sd | 0.2740 | -0.0169 | 0.0046 | 0.0012 | 0/15 | loses |
| yeo-johnson on TARGET | 0.2638 | -0.0271 | 0.0041 | 0.0010 | 0/15 | loses badly |
| raw target (= sign\|y\|^1) | 0.2573 | -0.0336 | 0.0091 | 0.0024 | 0/15 | loses badly |
| sign(y)\|y\|^1.5 | 0.1924 | -0.0985 | 0.0139 | 0.0036 | 0/15 | loses badly |
| sign(y)\|y\|^2 | 0.1063 | -0.1846 | 0.0123 | 0.0032 | 0/15 | loses badly |

The `sign(y)|y|^p` grid is the cleanest picture in the whole study: score rises monotonically
as p falls from 2.0 to ~0.2 and then flattens. Plain rank sits at the top of that curve, and
everything that *further* compresses the middle (centred-rank^0.5, winsorising at 10-20%,
tanh of a robust z-score) buys another ~0.002. Everything that *expands* the tails relative to
rank -- gaussian rank, logit, arcsine, centred-rank^p for p>1 -- loses. The direction is
unambiguous: **push more mass toward the middle, never less.**

Pooled vs per-country ranking is a small win (+0.0013 to +0.0015) but noisy; see section 6.

## 4. Sample weighting -- uniformly harmful (`b2`)

**Not one weighting scheme beat the baseline.** 41 variants, best delta +0.0002.

| variant | score | delta | d_sd | se | wins | verdict |
|---|---|---|---|---|---|---|
| w = exp(0.5*(rank\|y\| - 0.5)) | 0.2911 | +0.0002 | 0.0012 | 0.0003 | 8/15 | noise |
| downweight top 0.5% \|y\| -> w=0.5 | 0.2909 | -0.0000 | 0.0004 | 0.0001 | 8/15 | no-op |
| downweight top 1% \|y\| -> w=0 | 0.2895 | -0.0014 | 0.0018 | 0.0005 | 2/15 | loses |
| w = knn_dist(X)^-0.5 | 0.2898 | -0.0011 | 0.0009 | 0.0002 | 1/15 | loses |
| w = knn_dist(X)^+0.5 | 0.2892 | -0.0017 | 0.0014 | 0.0004 | 1/15 | loses |
| w = 1/(1+0.25\|z\|) | 0.2895 | -0.0014 | 0.0013 | 0.0003 | 1/15 | loses |
| w = exp(-0.25\|z\|) | 0.2889 | -0.0020 | 0.0016 | 0.0004 | 1/15 | loses |
| downweight top 2% \|y\| -> w=0 | 0.2879 | -0.0030 | 0.0019 | 0.0005 | 1/15 | loses |
| w = 1/(1+0.5\|z\|) | 0.2876 | -0.0033 | 0.0019 | 0.0005 | 0/15 | loses |
| downweight top 5% \|y\| -> w=0 | 0.2873 | -0.0036 | 0.0020 | 0.0005 | 0/15 | loses |
| edge-focus w = 1 - N(rank; .5, .4) | 0.2876 | -0.0033 | 0.0031 | 0.0008 | 1/15 | loses |
| w = 1 + 0.5\|z\| (UPweight extremes) | 0.2847 | -0.0062 | 0.0029 | 0.0008 | 0/15 | loses |
| w = kde_y(y)^-0.25 (inverse y-density) | 0.2852 | -0.0057 | 0.0027 | 0.0007 | 0/15 | loses |
| centre-focus w = N(rank; .5, .25) | 0.2850 | -0.0059 | 0.0031 | 0.0008 | 0/15 | loses |
| w = exp(-1.0\|z\|) | 0.2760 | -0.0148 | 0.0038 | 0.0010 | 0/15 | loses badly |
| downweight top 10% \|y\| -> w=0 | 0.2787 | -0.0122 | 0.0031 | 0.0008 | 0/15 | loses badly |
| w = kde_y(y)^-1.0 | 0.2288 | -0.0621 | 0.0114 | 0.0029 | 0/15 | loses badly |

The tempting hypothesis -- "the top 1% of |TARGET| carries 31% of the squared-error weight, so
downweight it" -- is **already fully solved by the rank transform**. On a rank target those
days are just the top few rank positions; they carry ordinary weight, and they are still real
observations. Downweighting them a second time throws away data. Both directions lose, and
symmetrically, which is the signature of an already-correct weighting.

Weighting is genuinely a dead end here. So is inverse-density weighting in either y or X.

## 5. Outlier handling and robust losses (`b3`, `b9`)

| variant | score | delta | d_sd | se | wins | verdict |
|---|---|---|---|---|---|---|
| drop top 0.5% \|y\| at fit | 0.2910 | +0.0001 | 0.0010 | 0.0002 | 7/15 | no-op |
| drop top 5% \|y\| at fit | 0.2910 | +0.0001 | 0.0019 | 0.0005 | 8/15 | no-op |
| drop top 1% \|y\| at fit | 0.2901 | -0.0008 | 0.0018 | 0.0005 | 5/15 | noise |
| drop top 2% \|y\| at fit | 0.2894 | -0.0015 | 0.0022 | 0.0006 | 3/15 | noise |
| drop top 10% \|y\| at fit | 0.2864 | -0.0045 | 0.0031 | 0.0008 | 2/15 | loses |
| drop top 20% \|y\| at fit | 0.2771 | -0.0137 | 0.0040 | 0.0010 | 0/15 | loses |
| drop top 5% X-outliers (leverage) | 0.2926 | +0.0017 | 0.0025 | 0.0007 | 11/15 | noise |
| drop top 2% X-outliers (leverage) | 0.2908 | -0.0001 | 0.0020 | 0.0005 | 8/15 | no-op |
| drop top 10% X-outliers (leverage) | 0.2881 | -0.0028 | 0.0036 | 0.0009 | 2/15 | loses |
| drop top 2/5% X-outliers (robust Mahalanobis) | 0.2911/0.2912 | +0.0002/+0.0003 | 0.002-0.003 | - | 9-10/15 | no-op |
| drop top 10% X-outliers (Mahalanobis) | 0.2869 | -0.0040 | 0.0033 | 0.0008 | 2/15 | loses |
| drop top 2% X-outliers (IsolationForest) | 0.2874 | -0.0035 | 0.0026 | 0.0007 | 1/15 | loses |
| one-step IRLS Huber weights (delta=1.345) | 0.2916 | +0.0007 | 0.0012 | 0.0003 | 10/15 | noise |
| one-step IRLS Huber weights (delta=1.0) | 0.2911 | +0.0002 | 0.0017 | 0.0004 | 10/15 | noise |
| one-step IRLS Huber weights (delta=2.0) | 0.2908 | -0.0001 | 0.0003 | 0.0001 | 5/15 | no-op |
| trim 10% worst ridge residual | 0.2856 | -0.0030 | 0.0036 | 0.0011 | 2/10 | loses |
| trim 10% worst residual, 2 passes | 0.2851 | -0.0035 | 0.0042 | 0.0013 | 2/10 | loses |
| trim 20% worst ridge residual | 0.2809 | -0.0077 | 0.0044 | 0.0014 | 1/10 | loses |

Robust regressors on the rank target (10 seeds, features standardised; baseline 0.2886):

| variant | score | delta | d_sd | se | wins | verdict |
|---|---|---|---|---|---|---|
| Huber(eps=1.35, alpha=1.0) | 0.2916 | +0.0030 | 0.0038 | 0.0012 | 8/10 | see below |
| Huber(eps=2.0, alpha=1e-2) | 0.2910 | +0.0024 | 0.0032 | 0.0010 | 7/10 | see below |
| Huber(eps=1.35, alpha=1e-2) | 0.2903 | +0.0017 | 0.0039 | 0.0012 | 8/10 | noise |
| RANSAC (Ridge base) | 0.2886 | +0.0000 | 0.0000 | 0.0000 | 0/10 | never trips; = baseline |
| QuantileRegressor L1 (q=0.5) | 0.2842 | -0.0044 | 0.0057 | 0.0018 | 3/10 | loses |
| **Huber(eps=1.05, alpha=1e-2)** (most robust) | 0.2816 | **-0.0071** | 0.0064 | 0.0020 | 2/10 | loses |
| TheilSen | 0.2077 | -0.0809 | 0.0233 | 0.0074 | 0/10 | loses badly |

Note the ordering: **the more robust the loss, the worse the score.** Huber at eps=2.0 (nearly
least-squares) beats Huber at eps=1.35 beats Huber at eps=1.05 (most robust), and TheilSen --
the most robust estimator tried -- collapses to 0.2077. The small positive Huber deltas are
not robustness at all; they are section 1 again, since a light `alpha` on standardised features
is far less shrinkage than RidgeCV's 70-143. RANSAC's inlier test never trips on a bounded
rank target, so it returns the base ridge exactly.

Same story as section 4, for the same reason. On a bounded, uniformly-spread rank target there
are no vertical outliers left to remove -- the transform already removed them. Every exclusion
rule is either a no-op (small q) or a data-loss penalty (large q). The apparent +0.0017 from
"drop 5% leverage" is 11/15 wins with `se`=0.0007 and does not survive as anything but noise.

## 6. Multi-task / pooling FR and DE (`b4`, `b8`)

FR and DE rows of a day are bit-identical in every feature. That makes pooling *look*
attractive and it is **catastrophic**.

| variant | score | delta | d_sd | se | wins | verdict |
|---|---|---|---|---|---|---|
| BASELINE (two separate models) | 0.2909 | - | - | - | - | - |
| stacked 29 feats, interaction scale s=1.0 | 0.2751 | -0.0158 | 0.0055 | 0.0014 | 0/15 | loses |
| stacked 29 feats, s=0 (one shared model) | 0.2154 | -0.0755 | 0.0074 | 0.0019 | 0/15 | loses badly |
| stacked 29 feats, s=10 (nearly separate) | 0.2695 | -0.0213 | 0.0064 | 0.0016 | 0/15 | loses |
| stacked, POOLED rank y, s=1.0 | 0.2794 | -0.0115 | 0.0050 | 0.0013 | 0/15 | loses |
| stacked FR-3 feats only, s=1.0 | 0.2016 | -0.0892 | 0.0071 | 0.0018 | 0/15 | loses badly |
| rank-blend 25% shared + 75% baseline | 0.2869 | -0.0040 | 0.0024 | 0.0006 | 0/15 | loses |
| rank-blend 50/50 | 0.2725 | -0.0184 | 0.0042 | 0.0011 | 0/15 | loses |
| **train on the OTHER country (diagnostic)** | **0.1057** | -0.1852 | 0.0077 | 0.0020 | 0/15 | see below |

The diagnostic is the informative one: a DE-trained model scored on FR (and vice versa) gets
**0.1057**, barely a third of the baseline. Identical features, completely different
coefficient vectors. There is no shared structure to borrow -- the two markets respond to the
same inputs in genuinely different ways -- so every degree of partial pooling costs score
monotonically. `s=10` still loses 0.021 partly because the stacked design gives FR all 29
features instead of its 3; that confound does not change the conclusion, since `s=0` and the
cross-country diagnostic are unconfounded and both collapse.

**Cross-country prediction calibration** deserves separate mention because it looked promising
and mostly is not. FR and DE marginals genuinely differ (KS D=0.135, p=2.5e-6; mean pooled rank
FR 0.4935 vs DE 0.5093), so per-country rank targets do discard a real cross-country offset.

| variant | score | delta | d_sd | se | wins | verdict |
|---|---|---|---|---|---|---|
| DE prediction offset +0.02 (hand-tuned) | 0.2947 | +0.0038 | 0.0012 | 0.0003 | 15/15 | win, but tuned |
| DE prediction offset +0.05 | 0.2945 | +0.0036 | 0.0021 | 0.0006 | 14/15 | win, but tuned |
| country offset from fold-train pooled mean rank | 0.2919 | +0.0010 | 0.0025 | 0.0006 | 11/15 | noise |
| ... same, x2 | 0.2901 | -0.0008 | 0.0054 | 0.0014 | 7/15 | noise |
| ... from fold-train pooled median rank | 0.2888 | -0.0021 | 0.0043 | 0.0011 | 5/15 | noise |
| DE prediction gain x1.25 | 0.2907 | -0.0002 | 0.0006 | 0.0001 | 4/15 | no-op |
| DE prediction gain x0.5 | 0.2830 | -0.0079 | 0.0019 | 0.0005 | 0/15 | loses |
| per-country calib: affine to fold-train mean/sd | 0.2841 | -0.0068 | 0.0028 | 0.0007 | 0/15 | loses |
| per-country calib: z-score | 0.2822 | -0.0086 | 0.0027 | 0.0007 | 0/15 | loses |
| per-country calib: empirical quantile of TARGET | 0.2818 | -0.0091 | 0.0027 | 0.0007 | 0/15 | loses |
| per-country calib: uniform ranks | 0.2769 | -0.0140 | 0.0032 | 0.0008 | 0/15 | loses |
| DE prediction offset -0.05 | 0.2638 | -0.0271 | 0.0026 | 0.0007 | 0/15 | loses |

The hand-tuned +0.02 offset works (15/15) and it is not an accident -- 0.02 is almost exactly
the 0.0158 gap in mean pooled rank. But the **fold-safe estimate of that same offset only
delivers +0.0010 at 11/15**, i.e. the offset cannot be estimated accurately enough from
fold-train data to be worth having. Treat the +0.0038 as tuned-on-CV and unusable.

More interesting: **full recalibration hurts, badly and consistently.** Mapping each country's
predictions onto the empirical TARGET quantiles -- the textbook-correct way to put two
countries on a common scale -- costs 0.009. The reason is that the raw ridge output's *spread*
is informative: shrinkage pulls uncertain rows toward the middle, and that is exactly right
when the two countries' rows have to interleave. Forcing the predicted marginal to be uniform
(or to match the target marginal) destroys that shrinkage and pushes low-confidence rows into
the tails. **Do not rank-normalise predictions per country before pooling.**

Using the POOLED rank as the training target (section 3, +0.0013/+0.0015) is the one version
of this idea that survives, and it is small.

## 7. Transductive use of X_test (`b5`, `b10`)

Legitimate -- test features are given, test labels are never read. Mostly worthless.

| variant | score | delta | d_sd | se | wins | verdict |
|---|---|---|---|---|---|---|
| TRANSDUCTIVE median: fold-train + X_test | 0.2909 | +0.0000 | 0.0001 | 0.0000 | 9/15 | exact no-op |
| TRANSDUCTIVE median: + X_test + X_valid | 0.2909 | +0.0000 | 0.0000 | 0.0000 | 8/15 | exact no-op |
| median: fold-train + X_valid only | 0.2909 | +0.0000 | 0.0001 | 0.0000 | 6/15 | exact no-op |
| **TRANSDUCTIVE KNN-impute k=5 (+scaler)** | 0.2932 | **+0.0023** | 0.0010 | 0.0003 | 15/15 | small win |
| TRANSDUCTIVE KNN-impute k=15 (+scaler) | 0.2929 | +0.0020 | 0.0008 | 0.0002 | 15/15 | small win |
| TRANSDUCTIVE PCA-whiten k=29 | 0.2930 | +0.0021 | 0.0025 | 0.0006 | 10/15 | noise |
| TRANSDUCTIVE PCA k=29 (rotation only) | 0.2920 | +0.0011 | 0.0009 | 0.0002 | 13/15 | small; = scaler effect |
| in-fold PCA k=29 (no test) | 0.2919 | +0.0010 | 0.0009 | 0.0002 | 13/15 | same without test |
| TRANSDUCTIVE quantile-normal features | 0.2878 | -0.0031 | 0.0065 | 0.0017 | 4/15 | loses (confirms prior) |
| TRANSDUCTIVE PCA k=20 | 0.2832 | -0.0077 | 0.0028 | 0.0007 | 0/15 | loses |
| TRANSDUCTIVE PCA k=10 | 0.2724 | -0.0185 | 0.0044 | 0.0011 | 0/15 | loses |
| TRANSDUCTIVE PCA k=5 | 0.2620 | -0.0289 | 0.0055 | 0.0014 | 0/15 | loses |
| TRANSDUCTIVE PCA-whiten k=5 | 0.2610 | -0.0299 | 0.0056 | 0.0014 | 0/15 | loses |

**Ablation of the KNN-imputer win** (this is the important row -- the naive read is wrong):

| variant | score | delta | d_sd | se | wins |
|---|---|---|---|---|---|
| BASELINE | 0.2909 | - | - | - | - |
| median impute + StandardScaler (no KNN, no test) | 0.2919 | +0.0010 | 0.0009 | 0.0002 | 13/15 |
| KNN-impute k=5, **in-fold only**, scaled | 0.2925 | +0.0017 | 0.0009 | 0.0002 | 15/15 |
| KNN-impute k=5, transductive, scaled | 0.2932 | +0.0023 | 0.0010 | 0.0003 | 15/15 |
| KNN-impute k=5, transductive, **unscaled** | 0.2920 | +0.0011 | 0.0007 | 0.0002 | 14/15 |

Decomposition of the +0.0023: about **0.0010 is just standardising the features** (which with
a fixed alpha grid changes ridge's per-feature penalty -- i.e. it is another regularisation
effect, section 1 again), about **0.0007 is KNN vs median imputation**, and about **0.0006 is
the extra test rows**. Transduction is real but worth well under a thousandth. Do not
oversell it.

Dimensionality reduction is a clear loss at every k below full rank; PCA k=29 is a rotation,
so its +0.0011 is the standardisation effect and nothing else.

## 8. Other target/row manipulations that failed (`b9`)

| variant | score | delta | d_sd | se | wins | verdict |
|---|---|---|---|---|---|---|
| kNN target smoothing k=15, a=0.25 | 0.2901 | -0.0008 | 0.0010 | 0.0003 | 3/15 | loses |
| kNN target smoothing k=5, a=0.5 | 0.2860 | -0.0049 | 0.0029 | 0.0007 | 0/15 | loses |
| kNN target smoothing k=30, a=0.5 | 0.2854 | -0.0055 | 0.0018 | 0.0005 | 0/15 | loses |
| subagged ridge frac=0.8, B=30 | 0.2892 | -0.0017 | 0.0016 | 0.0004 | 1/15 | loses |
| bagged ridge, Poisson weights, B=30 | 0.2867 | -0.0042 | 0.0025 | 0.0007 | 1/15 | loses |
| subagged ridge frac=0.5, B=30 | 0.2851 | -0.0058 | 0.0032 | 0.0008 | 1/15 | loses |
| bagged ridge, Poisson weights, B=10 | 0.2839 | -0.0070 | 0.0046 | 0.0012 | 1/15 | loses |

Bagging a linear model that is already ridge-regularised just adds variance from the resampling
without reducing any; averaging over bootstrap fits of a convex-in-y estimator buys nothing.

## 9. Combination on fresh seeds 200-229 (`b13`)

| variant | score | delta | d_sd | se | wins | verdict |
|---|---|---|---|---|---|---|
| BASELINE RidgeCV / 29 DE feats / rank y | 0.2902 | - | - | - | - | - |
| A  fixed alpha=10 | 0.2967 | +0.0065 | 0.0025 | 0.0004 | 30/30 | win |
| B  DE drops the 6 weather cols | 0.2970 | +0.0068 | 0.0035 | 0.0006 | 29/30 | win |
| C  target = centred-rank^0.5 | 0.2930 | +0.0027 | 0.0018 | 0.0003 | 28/30 | small win |
| C' target = rank winsorised @0.20 | 0.2916 | +0.0014 | 0.0014 | 0.0002 | 25/30 | small win |
| D  transductive KNN-impute k=5 + scaler | 0.2926 | +0.0024 | 0.0012 | 0.0002 | 29/30 | small win |
| E  fold-safe alpha by inner Spearman | 0.2933 | +0.0031 | 0.0036 | 0.0007 | 25/30 | win, untuned |
| A+C | 0.2981 | +0.0079 | 0.0033 | 0.0006 | 29/30 | win |
| E+B | 0.2986 | +0.0084 | 0.0048 | 0.0009 | 28/30 | win |
| E+B+C | 0.2991 | +0.0089 | 0.0054 | 0.0010 | 30/30 | **win, fully fold-safe** |
| A+B | 0.3011 | +0.0109 | 0.0040 | 0.0007 | 30/30 | win |
| A+B+C | 0.3020 | +0.0118 | 0.0046 | 0.0008 | 30/30 | win |
| **A+B+C+D** | **0.3024** | **+0.0122** | 0.0047 | 0.0009 | **30/30** | **best** |
| [B given A] = (A+B) - A | | +0.0044 | 0.0035 | 0.0006 | 27/30 | partly redundant |
| [C given A+B] = (A+B+C) - (A+B) | | +0.0009 | 0.0014 | 0.0003 | 23/30 | mostly absorbed |

A and B are **partly redundant**, as expected: both reduce the effective complexity of the
29-feature DE model. B alone is +0.0068; B on top of alpha=10 is +0.0044. They still stack to
+0.0109, more than either alone. C is largely absorbed once A and B are in (+0.0027 alone,
+0.0009 on top), which fits the same story -- the target-shape tweaks were a mild
regularisation effect too.

## Recommended pipelines

Fully fold-safe, no constant tuned on this CV (**E+B+C, 0.2991, +0.0089, 30/30**):

```python
import numpy as np, pandas as pd, sys
sys.path.insert(0, 'src'); import qrt_prep as P
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

FR_FEATS = ["FR_WINDPOW", "GAS_RET", "CARBON_RET"]
FEATS    = P.feature_columns(train)
WEATHER  = [c for c in FEATS if c.split("_")[-1] in ("TEMP", "WIND", "RAIN")]
DE_FEATS = [c for c in FEATS if c not in WEATHER]          # B: 23 of 29

def target(y):                                              # C: centred-rank^0.5, unit sd
    r = P.rank_transform(y) - 0.5
    v = np.sign(r) * np.sqrt(np.abs(r))
    return (v - v.mean()) / v.std()

def fit_predict(T, V, country):
    cols = FR_FEATS if country == "FR" else DE_FEATS
    Xt, Xv = P.impute(T, V, cols=cols)
    Xt, Xv, y = Xt.values, Xv.values, target(T.TARGET.values)

    # E: pick alpha inside the fold by SPEARMAN, not by GCV/MSE. RidgeCV's LOO-GCV
    # lands near alpha=70-140; the Spearman optimum for the 29-feature DE model is ~10.
    inner = list(GroupKFold(n_splits=5).split(Xt, y, groups=T.DAY_ID.values))
    best = (-2.0, None)
    for a in np.logspace(-1, 3, 17):
        o = np.zeros(len(y))
        for i, j in inner:
            o[j] = Ridge(alpha=a).fit(Xt[i], y[i]).predict(Xt[j])
        s = P.pooled_spearman(o, y)
        if s > best[0]:
            best = (s, a)
    return Ridge(alpha=best[1]).fit(Xt, y).predict(Xv)
```

Best-scoring, but with two constants (alpha=10, the weather column list) chosen on this CV
(**A+B+C+D, 0.3024, +0.0122, 30/30**):

```python
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def fit_predict_tuned(T, V, country, test):
    cols = FR_FEATS if country == "FR" else DE_FEATS         # B
    # D: KNN imputation on standardised features, fitted on fold-train + X_test.
    #    Transductive but label-free. Worth ~+0.0024, of which ~0.0010 is just the scaler.
    tt   = test[test.COUNTRY == country][cols]
    fitX = pd.concat([T[cols], tt])
    sc   = StandardScaler().fit(fitX.fillna(fitX.median()))
    im   = KNNImputer(n_neighbors=5).fit(pd.DataFrame(sc.transform(fitX), columns=cols))
    Xt, Xv = im.transform(sc.transform(T[cols])), im.transform(sc.transform(V[cols]))
    # A: alpha=10 flat. Sits on a broad 1-50 plateau, so it is not a knife-edge fit.
    #    Note a FIXED alpha is not scale-invariant, hence target() standardises to unit sd.
    return Ridge(alpha=10.0).fit(Xt, target(T.TARGET.values)).predict(Xv)
```

## What actually matters, in order

1. **The 29-feature DE model is over-regularised by GCV and over-parameterised for 643 rows.**
   Fixing either (alpha=10, or dropping 6 columns) is worth ~0.006-0.007; fixing both is worth
   ~0.011. Everything else in this study is a rounding error next to this.
2. **The rank transform already did all the heavy-tail work.** Every attempt to do it again --
   sample weighting, outlier exclusion, robust losses, density weighting, target smoothing,
   bagging -- is neutral at best and usually negative. This is the strongest negative result
   here: ~70 variants, not one real win, and within each family the score falls monotonically
   as the robustness knob is turned up (weight decay rate, exclusion fraction, Huber epsilon,
   all the way to TheilSen at 0.2077).
3. **Do not pool FR and DE.** Not as a stacked model, not with interactions, not as a blend,
   and do not recalibrate their predictions onto a common marginal before pooling.

## Verification

`experiments/verify_recipes.py` runs both code blocks above **verbatim as written in this file**
and re-scores them, including on a third seed block (300-329) not used anywhere else:

| pipeline | seeds 200-229 | seeds 300-329 |
|---|---|---|
| BASELINE | 0.2903 | 0.2929 |
| RECIPE 1 (E+B+C, fold-safe) | 0.2991 (+0.0088, 30/30) | 0.3020 (+0.0091, 28/30) |
| RECIPE 2 (A+B+C+D, tuned) | 0.3024 (+0.0121, 30/30) | 0.3046 (+0.0117, 30/30) |

Both replicate on the independent block. (Compare only within a column: the two baselines
differ by 0.0026 purely from fold-assignment luck, which is exactly the point of pairing.)

## Reproducing

`experiments/harness.py` plus `b1`..`b13`, `b3b`, and `verify_recipes.py`; each writes a JSON
of the same rows. Nothing here has been committed.
