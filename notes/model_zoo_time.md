# Model zoo, re-run on the ID-ordered time features

Everything in `notes/model_zoo.md` was measured in the 0.29 regime. The `DAY_ID`-decoy
finding moved the target to ~0.52, and the coordinator was right to be suspicious: **the
headline conclusion of that document inverts, and four of its five confirmed sub-findings
fail to transfer.** This file supersedes it.

## TL;DR

1. **"Added capacity always loses" is dead.** The *identical* LightGBM configuration that
   lost by 0.0125 on the base features now **beats ridge by +0.0232 (15/15 seeds)**,
   scoring 0.5463 against 0.5231. HistGradientBoosting flips the same way, -0.0274 ->
   +0.0114. The rule was never about model class; it was about signal-to-noise. With 58
   informative time features there is finally enough signal to support interactions.
2. **The diversity story inverts with it.** In the 0.29 regime, boosting was worthless
   alone and valuable only as a blend partner. Now it is the best standalone model, and
   *ridge* is the member you drop: the best combination found is
   **`0.5*z(LightGBM) + 0.5*z(ARDRegression)` with no ridge at all -- 0.5593, +0.0362,
   15/15**, confirmed on 30 unseen fold seeds.
3. **The biggest surprise is a linear model.** `ARDRegression`, worth +0.0009 (noise) on
   29 features, is worth **+0.0164 (15/15)** on 87. Automatic relevance determination is
   doing feature selection that ridge's single global alpha cannot: most of the 58 new
   columns are noise, and ARD prunes them per-coefficient. Sparsity was harmful in the old
   regime and is essential in this one -- an exact reversal.
4. **What did NOT transfer, from my own prior document.** `Ridge-std` +0.0013 -> **-0.0040
   (0/15)**; `Huber` +0.0020 -> **-0.0273 (0/15)**; `PLS` +0.0014 -> **-0.0275 (0/15)**;
   `C1` FR-ridge/DE-spline +0.0035 -> **-0.0016 (5/15)**; `C2` FR-ridge/DE-KRR +0.0038 ->
   **-0.0109 (0/15)**. The France/Germany asymmetry finding is gone with them. Only
   `Ridge-winsor` survives, and weakly: +0.0022 -> +0.0021 (11/15).
5. **Kernel and instance methods got much worse, not better.** `SVR-rbf` -0.0137 ->
   -0.0538; `kNN-std-dist` -0.0379 -> -0.0829. They cannot cope with 87 columns of mixed
   scale where most are noise. Their OOF correlation with ridge fell (0.85 -> 0.78,
   0.81 -> 0.76), but that is now the signature of a broken model rather than a useful one.

## Protocol

* **Features.** `TO.add_time_features(train, test, F, lags=(1,), windows=(7,))` -> 58 new
  columns (first difference at lag 1, and value-minus-trailing-7-day-mean, for each of the
  29 base features). FR uses `SEL + NEW` (61 columns), DE uses `F + NEW` (87).
* **Baselines.** Per-country `RidgeCV(alphas=np.logspace(-2,4,40))` on median-imputed raw
  features with a within-country rank-transformed target:
  * **time-ridge = 0.5231 +/- 0.0061** (15 seeds) -- the paired reference for every row
    below, since every candidate uses the same feature set.
  * **time-ridge + neighbour(k=3) = 0.5277 +/- 0.0076** -- reproduces the coordinator's
    0.5282. Used as the reference for section 4 only.
* **Pairing.** Identical `P.make_folds(train.DAY_ID, seed=s)` assignments, seeds 0..14.
  `delta` is the mean of the 15 per-seed differences, `sd(delta)` their sd, `wins/15` the
  count of seeds where the candidate is higher.
* **Leakage.** Time features are built from X only (train and test), with `.shift(1)` inside
  every rolling window, so they are computable at submission time and contain no target;
  they are therefore built once, outside the fold loop. Everything target-dependent --
  imputation medians, scaling, winsorisation cut-points, the rank transform, and all
  hyperparameter selection -- is fitted inside the fold on that country's training rows.
  The neighbour-target features in section 4 are the one genuinely leak-prone piece and are
  rebuilt per fold with sources restricted to that fold's training rows of that country,
  with the identity match dropped.
* **Sanity check on the jump.** 0.29 -> 0.52 is large enough to deserve suspicion. It is not
  leakage: the added columns are differences and rolling deviations of X, the fold split is
  still grouped on DAY_ID, and the ridge that produces 0.5231 is the same estimator that
  produces 0.2909 on the base features. The gain comes from the features, and the
  organisers supply test X, so it survives to submission.
* **One transductive detail, named rather than hidden.** `add_time_features` clips the
  derived columns at 8 sd, and that sd is computed over the whole train+test derived frame,
  i.e. outside the fold. It is a feature-only statistic with no target in it, so it cannot
  leak label information, and it is computable at submission time since test X is supplied.
  It is still a global constant rather than a fold-local one; anyone who wants the estimate
  to be airtight can move it inside the fold, at the cost of clipping test rows with a
  train-only sd.

## 1. Full sweep on time features, paired against time-ridge (seeds 0..14)

| model | config | score | delta vs ridge | sd(delta) | wins/15 | corr(ridge) | FR rho | DE rho |
|---|---|---|---|---|---|---|---|---|
| `LGBM-leaves4` | identical config to the 0.29-regime run | 0.5463 | +0.0232 | 0.0072 | 15/15 | 0.881 | 0.2426 | 0.7847 |
| `ARDRegression` | default (was +0.0009, best DE linear) | 0.5395 | +0.0164 | 0.0042 | 15/15 | 0.926 | 0.2666 | 0.7677 |
| `HGB-shallow` | identical config to the 0.29-regime run | 0.5345 | +0.0114 | 0.0070 | 14/15 | 0.856 | 0.2329 | 0.7787 |
| `Spline+Ridge` | cubic B-splines per feature, n_knots tuned inner, RidgeCV head | 0.5286 | +0.0055 | 0.0052 | 13/15 | 0.934 | 0.2251 | 0.7684 |
| `Ridge-winsor` | features clipped to fold-train 2/98 pct (was +0.0022, 25/30) | 0.5251 | +0.0021 | 0.0028 | 11/15 | 0.969 | 0.2137 | 0.7718 |
| `C1-FRridge-DEspline` | ridge FR + splines DE (was +0.0035, 22/30) | 0.5215 | -0.0016 | 0.0032 | 5/15 | 0.963 | 0.2053 | 0.7684 |
| `POOLED-LGBM-leaves15` | pooled, COUNTRY feature, leaves 15 / 600 rounds (2x the rows) | 0.5214 | -0.0017 | 0.0077 | 6/15 | 0.803 | 0.2122 | 0.7614 |
| `Ridge-std` | RidgeCV on z-scored features (was +0.0013, 27/30) | 0.5191 | -0.0040 | 0.0029 | 0/15 | 0.976 | 0.2053 | 0.7690 |
| `C2-FRridge-DEkrr` | ridge FR + kernel ridge DE (was +0.0038, 25/30) | 0.5122 | -0.0109 | 0.0034 | 0/15 | 0.966 | 0.2053 | 0.7581 |
| `Huber` | alpha tuned inner (was +0.0020, 10/15) | 0.4958 | -0.0273 | 0.0065 | 0/15 | 0.928 | 0.1953 | 0.7600 |
| `PLS` | n_components tuned inner (was +0.0014, 9/15) | 0.4956 | -0.0275 | 0.0061 | 0/15 | 0.921 | 0.1929 | 0.7597 |
| `POOLED-HGB-shallow` | pooled, COUNTRY feature, HistGB depth 2 | 0.4864 | -0.0367 | 0.0076 | 0/15 | 0.770 | 0.1505 | 0.7303 |
| `POOLED-LGBM-leaves4` | pooled, COUNTRY feature, same leaves-4 config as the per-country winner | 0.4738 | -0.0493 | 0.0070 | 0/15 | 0.739 | 0.1340 | 0.7429 |
| `SVR-rbf` | C/gamma tuned inner, identical grid | 0.4693 | -0.0538 | 0.0108 | 0/15 | 0.777 | 0.2084 | 0.7278 |
| `kNN-std-dist` | z-scored feats, distance weights, k tuned | 0.4402 | -0.0829 | 0.0097 | 0/15 | 0.761 | 0.1721 | 0.7115 |
| `POOLED-ridge` | one RidgeCV over FR+DE with a COUNTRY indicator (sanity check) | 0.3951 | -0.1280 | 0.0044 | 0/15 | 0.702 | 0.1339 | 0.6820 |


## 2. Diversity map: pairwise OOF Spearman between models

Averaged over the 15 seeds. Compare with the 0.29 regime, where the same
boosting/SVR/kNN members sat at 0.78-0.85 against ridge.

| | `ridge` | `ARDRegression` | `C1-FRridge-DEspline` | `C2-FRridge-DEkrr` | `HGB-shallow` | `Huber` | `LGBM-leaves4` | `PLS` | `POOLED-HGB-shallow` | `POOLED-LGBM-leaves15` | `POOLED-LGBM-leaves4` | `POOLED-ridge` | `Ridge-std` | `Ridge-winsor` | `SVR-rbf` | `Spline+Ridge` | `kNN-std-dist` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ridge` | 1.000 | 0.926 | 0.963 | 0.966 | 0.856 | 0.928 | 0.881 | 0.921 | 0.770 | 0.803 | 0.739 | 0.702 | 0.976 | 0.969 | 0.777 | 0.934 | 0.761 |
| `ARDRegression` | 0.926 | 1.000 | 0.903 | 0.904 | 0.834 | 0.907 | 0.858 | 0.893 | 0.751 | 0.781 | 0.714 | 0.655 | 0.912 | 0.910 | 0.747 | 0.882 | 0.715 |
| `C1-FRridge-DEspline` | 0.963 | 0.903 | 1.000 | 0.976 | 0.877 | 0.900 | 0.904 | 0.914 | 0.794 | 0.825 | 0.764 | 0.690 | 0.984 | 0.980 | 0.782 | 0.966 | 0.782 |
| `C2-FRridge-DEkrr` | 0.966 | 0.904 | 0.976 | 1.000 | 0.857 | 0.906 | 0.883 | 0.920 | 0.778 | 0.805 | 0.749 | 0.689 | 0.991 | 0.979 | 0.781 | 0.942 | 0.772 |
| `HGB-shallow` | 0.856 | 0.834 | 0.877 | 0.857 | 1.000 | 0.796 | 0.962 | 0.809 | 0.818 | 0.871 | 0.773 | 0.625 | 0.864 | 0.879 | 0.757 | 0.899 | 0.739 |
| `Huber` | 0.928 | 0.907 | 0.900 | 0.906 | 0.796 | 1.000 | 0.813 | 0.919 | 0.704 | 0.757 | 0.664 | 0.640 | 0.914 | 0.905 | 0.769 | 0.864 | 0.678 |
| `LGBM-leaves4` | 0.881 | 0.858 | 0.904 | 0.883 | 0.962 | 0.813 | 1.000 | 0.828 | 0.837 | 0.887 | 0.798 | 0.646 | 0.891 | 0.908 | 0.771 | 0.926 | 0.764 |
| `PLS` | 0.921 | 0.893 | 0.914 | 0.920 | 0.809 | 0.919 | 0.828 | 1.000 | 0.712 | 0.759 | 0.675 | 0.629 | 0.926 | 0.916 | 0.751 | 0.880 | 0.708 |
| `POOLED-HGB-shallow` | 0.770 | 0.751 | 0.794 | 0.778 | 0.818 | 0.704 | 0.837 | 0.712 | 1.000 | 0.867 | 0.957 | 0.816 | 0.784 | 0.793 | 0.664 | 0.807 | 0.737 |
| `POOLED-LGBM-leaves15` | 0.803 | 0.781 | 0.825 | 0.805 | 0.871 | 0.757 | 0.887 | 0.759 | 0.867 | 1.000 | 0.829 | 0.668 | 0.811 | 0.824 | 0.760 | 0.851 | 0.714 |
| `POOLED-LGBM-leaves4` | 0.739 | 0.714 | 0.764 | 0.749 | 0.773 | 0.664 | 0.798 | 0.675 | 0.957 | 0.829 | 1.000 | 0.882 | 0.755 | 0.764 | 0.632 | 0.775 | 0.742 |
| `POOLED-ridge` | 0.702 | 0.655 | 0.690 | 0.689 | 0.625 | 0.640 | 0.646 | 0.629 | 0.816 | 0.668 | 0.882 | 1.000 | 0.697 | 0.693 | 0.562 | 0.680 | 0.702 |
| `Ridge-std` | 0.976 | 0.912 | 0.984 | 0.991 | 0.864 | 0.914 | 0.891 | 0.926 | 0.784 | 0.811 | 0.755 | 0.697 | 1.000 | 0.989 | 0.777 | 0.950 | 0.780 |
| `Ridge-winsor` | 0.969 | 0.910 | 0.980 | 0.979 | 0.879 | 0.905 | 0.908 | 0.916 | 0.793 | 0.824 | 0.764 | 0.693 | 0.989 | 1.000 | 0.785 | 0.961 | 0.779 |
| `SVR-rbf` | 0.777 | 0.747 | 0.782 | 0.781 | 0.757 | 0.769 | 0.771 | 0.751 | 0.664 | 0.760 | 0.632 | 0.562 | 0.777 | 0.785 | 1.000 | 0.803 | 0.691 |
| `Spline+Ridge` | 0.934 | 0.882 | 0.966 | 0.942 | 0.899 | 0.864 | 0.926 | 0.880 | 0.807 | 0.851 | 0.775 | 0.680 | 0.950 | 0.961 | 0.803 | 1.000 | 0.794 |
| `kNN-std-dist` | 0.761 | 0.715 | 0.782 | 0.772 | 0.739 | 0.678 | 0.764 | 0.708 | 0.737 | 0.714 | 0.742 | 0.702 | 0.780 | 0.779 | 0.691 | 0.794 | 1.000 |

## 3. Blends with time-ridge (pooled-z, seeds 0..14)

`(1-w)*z(ridge) + w*z(candidate)`. Weights are scanned, so treat as an upper bound.

| candidate | corr(ridge) | w=0.25 | delta | w=0.40 | delta | w=0.50 | delta |
|---|---|---|---|---|---|---|---|
| `LGBM-leaves4` | 0.881 | 0.5402 | +0.0172 (15/15) | 0.5463 | +0.0232 (15/15) | 0.5487 | +0.0257 (15/15) |
| `HGB-shallow` | 0.856 | 0.5382 | +0.0151 (15/15) | 0.5426 | +0.0195 (15/15) | 0.5439 | +0.0208 (15/15) |
| `POOLED-LGBM-leaves15` | 0.803 | 0.5403 | +0.0172 (15/15) | 0.5439 | +0.0208 (15/15) | 0.5437 | +0.0206 (15/15) |
| `ARDRegression` | 0.926 | 0.5328 | +0.0097 (15/15) | 0.5368 | +0.0137 (15/15) | 0.5387 | +0.0156 (15/15) |
| `POOLED-HGB-shallow` | 0.770 | 0.5338 | +0.0107 (15/15) | 0.5314 | +0.0083 (15/15) | 0.5270 | +0.0039 (13/15) |
| `POOLED-LGBM-leaves4` | 0.739 | 0.5327 | +0.0096 (15/15) | 0.5279 | +0.0049 (14/15) | 0.5218 | -0.0012 (7/15) |
| `Spline+Ridge` | 0.934 | 0.5291 | +0.0060 (15/15) | 0.5313 | +0.0082 (15/15) | 0.5321 | +0.0090 (15/15) |
| `SVR-rbf` | 0.777 | 0.5288 | +0.0057 (15/15) | 0.5250 | +0.0019 (10/15) | 0.5199 | -0.0032 (3/15) |
| `Ridge-winsor` | 0.969 | 0.5256 | +0.0025 (15/15) | 0.5266 | +0.0035 (15/15) | 0.5270 | +0.0039 (15/15) |
| `C1-FRridge-DEspline` | 0.963 | 0.5257 | +0.0026 (15/15) | 0.5263 | +0.0032 (15/15) | 0.5262 | +0.0031 (14/15) |
| `Ridge-std` | 0.976 | 0.5237 | +0.0006 (12/15) | 0.5237 | +0.0006 (11/15) | 0.5233 | +0.0003 (11/15) |
| `C2-FRridge-DEkrr` | 0.966 | 0.5229 | -0.0001 (8/15) | 0.5221 | -0.0010 (3/15) | 0.5211 | -0.0019 (1/15) |
| `PLS` | 0.921 | 0.5216 | -0.0015 (3/15) | 0.5186 | -0.0044 (2/15) | 0.5159 | -0.0072 (0/15) |
| `Huber` | 0.928 | 0.5212 | -0.0019 (1/15) | 0.5181 | -0.0050 (0/15) | 0.5155 | -0.0076 (0/15) |
| `kNN-std-dist` | 0.761 | 0.5210 | -0.0021 (3/15) | 0.5132 | -0.0099 (0/15) | 0.5053 | -0.0177 (0/15) |
| `POOLED-ridge` | 0.702 | 0.5134 | -0.0097 (0/15) | 0.4971 | -0.0260 (0/15) | 0.4834 | -0.0397 (0/15) |


## 4. Ensembles (seeds 0..14)

Members and weights are chosen on these seeds, so treat as an upper bound;
section 5 re-scores the survivors on fold seeds never used for selection.

| ensemble (pooled-z weights) | score | delta vs time-ridge | sd(delta) | wins/15 |
|---|---|---|---|---|
| ridge alone (reference) | 0.5231 | +0.0000 | 0.0000 | 0/15 |
| LGBM alone | 0.5463 | +0.0232 | 0.0072 | 15/15 |
| 0.60 ridge + 0.40 LGBM  (the 0.29-regime recipe) | 0.5463 | +0.0232 | 0.0030 | 15/15 |
| 0.50 ridge + 0.25 LGBM + 0.25 SVR (the 0.29-regime best) | 0.5434 | +0.0203 | 0.0032 | 15/15 |
| 0.50 ridge + 0.50 LGBM | 0.5487 | +0.0257 | 0.0038 | 15/15 |
| equal ridge + LGBM + ARD | 0.5539 | +0.0308 | 0.0030 | 15/15 |
| equal ridge + LGBM + ARD + HGB | 0.5550 | +0.0319 | 0.0039 | 15/15 |
| 0.50 LGBM + 0.50 ARD  (no ridge) | 0.5593 | +0.0362 | 0.0044 | 15/15 |
| equal LGBM + ARD + HGB  (no ridge) | 0.5569 | +0.0338 | 0.0050 | 15/15 |
| equal LGBM + ARD + Spline  (no ridge) | 0.5554 | +0.0323 | 0.0040 | 15/15 |

## 5. Confirmation on 30 fresh fold seeds (15..44)

Members and weights fixed before running.

fresh seeds 15..44 (n=30)   time-ridge 0.5245 +/- 0.0082

| configuration (fixed in advance) | score | delta vs time-ridge | sd(delta) | wins/30 |
|---|---|---|---|---|
| ridge | 0.5245 | +0.0000 | 0.0000 | 0/30 |
| ARDRegression alone | 0.5376 | +0.0131 | 0.0052 | 30/30 |
| LGBM alone | 0.5452 | +0.0207 | 0.0068 | 30/30 |
| HGB alone | 0.5309 | +0.0065 | 0.0065 | 26/30 |
| 0.50 LGBM + 0.50 ARD (no ridge) | 0.5577 | +0.0332 | 0.0044 | 30/30 |
| equal LGBM + ARD + HGB (no ridge) | 0.5543 | +0.0298 | 0.0050 | 30/30 |
| equal ridge + LGBM + ARD + HGB | 0.5532 | +0.0287 | 0.0037 | 30/30 |

## 6. Temporal-neighbour features (section 4 of the coordinator's brief)

Scored against the ridge+neighbour(k=3) baseline of 0.5277, not plain time-ridge.

| model | score | delta vs ridge+nb | sd(delta) | wins/15 | corr(ridge+nb) |
|---|---|---|---|---|---|
| `ridge + neighbour(k=3)` | 0.5277 | +0.0046 vs plain time-ridge | 0.0053 | 13/15 | 1.000 |
| `NB-HGB-shallow` | 0.5368 | +0.0091 | 0.0087 | 12/15 | 0.849 |
| `NB-LGBM-leaves4` | 0.5486 | +0.0209 | 0.0077 | 15/15 | 0.874 |

## 7. What changed, and why

**The regime change is a signal-to-noise change, not a model-class change.** On the base
features, ridge's pooled OOF was 0.29 and the honest spread between the best and worst
sensible model was ~0.08. On the time features ridge is 0.52 and the spread is ~0.11, but
the *ordering* has flipped: the families that were penalised for capacity are now rewarded
for it. Two mechanisms, both visible in the tables:

1. *There is now enough signal to identify interactions.* 58 of the 87 DE columns are
   first differences and deviations-from-trailing-mean. Electricity price moves are driven
   by things like "residual load jumped **and** gas is above its recent regime", which is a
   product term a linear model cannot represent. With 0.29 of signal, estimating those
   interactions cost more variance than they bought. With 0.52, they pay: LightGBM
   -0.0125 -> +0.0232, HistGB -0.0274 -> +0.0114, splines -0.0009 -> +0.0055.
2. *There are now enough columns for relevance weighting to matter.* Going from 29 to 87
   features means most columns are noise. Ridge's single global alpha must shrink the good
   ones to control the bad ones; `ARDRegression`'s per-coefficient prior does not. That is
   worth +0.0164 with no nonlinearity at all, and it is the single cleanest reversal in the
   sweep: sparsity was *harmful* on 29 features and is *essential* on 87.

**The whole 0.29 -> 0.52 jump is Germany, and France is still the open problem.**
Per country, ridge goes FR 0.2084 -> 0.2265 (+0.018) and DE 0.3598 -> 0.7630 (+0.403).
The recovered ordering essentially solves Germany and barely touches France. That makes
the France column the place to look, and there is a large result sitting in it:
**`ARDRegression` scores FR rho 0.2666 against time-ridge's 0.2265, +0.040** -- by a wide
margin the largest France gain in either document, and more than double what the time
features themselves bought France. LightGBM manages FR 0.2426. France has 61 columns and
851 rows, of which 58 columns are derived and mostly irrelevant to it; ridge's single alpha
has to shrink everything to survive that, and ARD does not. Anyone working on France next
should start from ARD, not ridge.

**Boosting and ARD are complementary, and ridge is redundant to both.** LightGBM's OOF
correlates 0.881 with ridge and 0.858 with ARD; ARD correlates 0.926 with ridge. So the
tree model contributes nonlinearity, ARD contributes a better-selected linear part, and
ridge contributes a strictly worse version of what ARD already provides. Dropping ridge
from the ensemble *improves* it: `LGBM + ARD` at 0.5593 beats `ridge + LGBM + ARD` at
0.5539 and `ridge + LGBM + ARD + HGB` at 0.5550. On fresh seeds, `0.5 LGBM + 0.5 ARD`
is +0.0332 (30/30) against time-ridge's +0.0000.

**The winner's pooled-with-COUNTRY shape does not reproduce here -- a clean negative.**
A single model over FR+DE with COUNTRY as a feature loses in every configuration tried:
pooled RidgeCV -0.1280 (a country dummy can only move the intercept, so this is expected),
pooled LightGBM at the per-country winner's settings -0.0493, pooled HistGB -0.0367, and
the best pooled tree found -- LightGBM with 15 leaves and 600 rounds, given the extra
capacity the doubled row count allows -- only reaches parity at -0.0017 (6/15). Doubling
the rows does not compensate for forcing one function to serve two price series. If the
winner's CatBoost really was pooled, the gain must have come from elsewhere in his
pipeline, not from pooling itself.

**Everything that made kNN and SVR interesting is gone.** Both were mediocre-but-diverse in
the 0.29 regime and were worth +0.004 to +0.006 in a blend. On 87 columns they are simply
broken: SVR-rbf -0.0538, kNN -0.0829. Their correlation with ridge fell (0.85 -> 0.78,
0.81 -> 0.76), which in the old document was the marker of a useful ensemble member and
here is just the signature of a model that has lost the plot. Low correlation is only
worth having when the model is still tracking the target.

**My old blend recipe survives in direction but not in composition.** The 0.29-regime
best -- `0.50 ridge + 0.25 LightGBM + 0.25 SVR-rbf` -- still gains here, +0.0203 (15/15),
because the LightGBM half is now genuinely strong. But it is beaten by LightGBM on its own
(+0.0232) and comfortably beaten by `LGBM + ARD` (+0.0362), because a quarter of its weight
is on SVR-rbf, which is now the second-worst model in the sweep. The general lesson
transfers; the specific recipe does not.

**Neighbour-target features and boosting overlap but do not cancel.** Ridge gains +0.0046
from the k=3 neighbour features (0.5231 -> 0.5277). LightGBM gains only +0.0023 from them
(0.5463 -> 0.5486), and stays +0.0209 ahead of ridge+neighbour. Both are reading the same
recovered ordering, so roughly half of what the neighbour features give a linear model is
already available to a tree through the differenced columns.

## 8. Exact constructors

```python
import sys; sys.path.insert(0, "src")
import numpy as np, qrt_prep as P, qrt_timeorder as TO
from scipy.stats import rankdata

train, test = P.build_frames()
F   = P.feature_columns(train)
SEL = ["FR_WINDPOW", "GAS_RET", "CARBON_RET"]
tr, te, NEW = TO.add_time_features(train, test, F, lags=(1,), windows=(7,))
CF, CD = SEL + NEW, F + NEW                    # FR 61 cols, DE 87 cols

# per fold, per country: median-impute from the fold's training rows, rank the target
med = t[cols].median()
Xtr = np.nan_to_num(t[cols].fillna(med).to_numpy(float))
Xva = np.nan_to_num(v[cols].fillna(med).to_numpy(float))
ytr = rankdata(t.TARGET.values) / len(t)
```

### 8.1 Best single model: LightGBM, per country (+0.0207, 30/30 on fresh seeds)

Raw (unscaled) imputed features. This is byte-for-byte the config that *lost* by 0.0125
on the base features -- nothing about it was retuned for the new regime.

```python
import lightgbm as lgb
lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=4,
                  min_child_samples=40, subsample=0.7, subsample_freq=1,
                  colsample_bytree=(1.0 if country == "FR" else 0.4),
                  reg_lambda=20.0, random_state=seed, n_jobs=1, verbose=-1)
```

### 8.2 Best linear model: ARDRegression, per country (+0.0131, 30/30 on fresh seeds)

Needs z-scored features (`prep="std"`); it is scale-sensitive in a way RidgeCV is not.
No hyperparameters, no inner CV, 2 seconds for all 15 seeds.

```python
from sklearn.linear_model import ARDRegression
ARDRegression()                      # on (X - mu) / sd, fitted inside the fold
```

### 8.3 BEST OVERALL: drop ridge, blend LightGBM with ARD (+0.0332, 30/30 on fresh seeds)

```python
z = lambda v: (v - v.mean()) / v.std()          # global, never per country
pred = 0.5 * z(lgbm_pred) + 0.5 * z(ard_pred)   # 0.5577 on 30 unseen fold seeds
```

## 9. Out-of-fold bundle

`notes/oof_zoo_time.pkl` is a `dict[str, list[np.ndarray]]`, 15 OOF vectors per model
(fold seeds 0..14, in order), each of length 1494 aligned to `P.build_frames()[0]` row
order. Members: `ridge_time`, `ridge_time_nb`, `LGBM-leaves4`, `ARDRegression`,
`HGB-shallow`, `Spline+Ridge`, `SVR-rbf`, `kNN-std-dist`, `Ridge-winsor`,
`NB-LGBM-leaves4`, `NB-HGB-shallow`.

```python
import pickle, numpy as np
from scipy.stats import spearmanr
oof = pickle.load(open("notes/oof_zoo_time.pkl", "rb"))
z = lambda v: (v - v.mean()) / v.std()
best = [0.5 * z(l) + 0.5 * z(a)
        for l, a in zip(oof["LGBM-leaves4"], oof["ARDRegression"])]   # 0.5593 on seeds 0..14
```

`SVR-rbf` and `kNN-std-dist` are kept in the bundle only so the negative result is
reproducible; do not put them in an ensemble.
