# Reconstructing the true day order

**Status: SOLVED — exactly, not approximately.**

The chronological order of all 1,216 anonymised days is recoverable in closed form from the
`ID` column. `DAY_ID` is a shuffle; `ID` is *not*. The mapping is

```
POSITION t  =  ID_FR - 932            (every day has an FR row; ID_FR in 932..2147)
ID_DE       =  t - 284                (a DE row exists iff t >= 284)
```

`t` runs 0..1215 and is a bijection. The map is written to
[`day_order.csv`](day_order.csv) as `DAY_ID, POSITION, SPLIT, ID_FR, ID_DE`.

Downstream consequence, measured below: a plain ridge on
`[feature levels + 1-day feature changes]` scores **0.504 pooled OOF Spearman**
versus **0.275** for the same ridge on levels alone. That is the whole
"winners reach 0.5+, feature-only models plateau at 0.30" gap.

---

## 1. How the ID structure was found

Grouping rows by `DAY_ID` and looking at `max(ID) - min(ID)` for the 932 days that carry both an
FR and a DE row gives **exactly 1216 for every one of them** — 1216 being the number of distinct
days. That is not a coincidence of a shuffle. Unpacking it:

| | ID range | n |
|---|---|---|
| DE rows | 0 .. 931 | 932 |
| FR rows | 932 .. 2147 | 1216 |

so `ID` is two stacked blocks, each block internally ordered by the same underlying day index,
offset by 1216. Sorting days by `ID_FR` orders them in time.

I found this *after* two days of blind seriation work (section 4), by checking whether the `ID`
column leaked anything — the cheap check that should have come first.

## 2. Proof that ID order is real time

Every one of these validators is a column or a structure that plays **no part** in defining the
order (the order comes from `ID` alone, which is not a feature).

### 2a. Lag-1 autocorrelation vs a shuffled null (200 permutations)

| column | lag-1 along ID order | shuffled null | z |
|---|---|---|---|
| DE_NUCLEAR | **0.982** | 0.000 ± 0.027 | 35.8 |
| FR_NUCLEAR | **0.980** | -0.004 ± 0.030 | 33.0 |
| FR_CONSUMPTION | **0.965** | -0.001 ± 0.027 | 36.2 |
| DE_SOLAR | **0.882** | 0.002 ± 0.027 | 32.1 |
| DE_WIND | **0.811** | 0.002 ± 0.031 | 26.1 |
| FR_WIND | **0.784** | 0.001 ± 0.032 | 24.8 |
| FR_TEMP | **0.754** | -0.000 ± 0.028 | 27.3 |
| DE_TEMP | **0.737** | -0.003 ± 0.032 | 22.9 |
| DE_RAIN | **0.303** | -0.001 ± 0.030 | 10.1 |
| FR_RAIN | **0.201** | 0.001 ± 0.031 | 6.6 |
| GAS_RET | 0.180 | -0.001 ± 0.029 | 6.3 |
| COAL_RET | -0.052 | -0.003 ± 0.029 | -1.7 |
| CARBON_RET | 0.074 | -0.000 ± 0.033 | 2.3 |
| \|GAS_RET\| | **0.210** | -0.003 ± 0.028 | 7.6 |
| \|COAL_RET\| | **0.159** | -0.002 ± 0.029 | 5.6 |
| \|CARBON_RET\| | **0.171** | -0.003 ± 0.029 | 6.1 |

This is exactly the signature of real daily data:

* Nuclear availability and consumption are near-unit-root (0.96–0.98).
* Temperature **anomalies** decay fast and realistically:
  L1 0.74 → L2 0.50 → L3 0.37 → L5 0.20 → L7 0.12 → L14 0.09 → L30 -0.05.
* Rainfall is much less persistent than temperature (0.20–0.30), as it must be.
* **Signed commodity returns have ~zero autocorrelation at every lag** while their
  **absolute values cluster** (0.16–0.21 at L1, decaying to 0 by L14). No ordering fitted to
  the physical features could manufacture that pattern; it is the volatility-clustering
  signature of genuine financial returns placed in true calendar order.

### 2b. Missingness is perfectly contiguous

Nothing about missingness was used to construct the order. Under ID order every missingness
pattern collapses to a **single run**:

| marker | positions | n | runs |
|---|---|---|---|
| DE row absent (no German target) | 0–283 | 284 | **1** |
| DE_NET_EXPORT missing | 0–170 | 171 | **1** |
| FR_NET_EXPORT missing | 0–93 | 94 | **1** |
| DE_FR_EXCHANGE missing | 0–33 | 34 | **1** |
| all 6 weather cols missing | 725–791 | 67 | **1** |

A nested "series come online" ladder at the start of the sample, plus one 67-day weather-feed
outage in the middle. Under a random ordering the expected number of runs for the 67-day weather
block alone is ≈ 63; observing 1 has probability ~1e-100.

### 2c. Physical time series behave correctly

* **Annual period.** Fitting `a + b·cos(2πt/P) + … + trend` and scanning `P`:
  DE_SOLAR peaks at **P = 250.00**, FR_SOLAR 250.75, FR_CONSUMPTION 250.50 index steps,
  with R² = 0.72, 0.71, 0.72. So 1216 days / 250 ≈ **4.86 years**, i.e. the sample is
  *exchange trading days* (~252/yr), not calendar days. Solar's fitted maximum sits ~98 steps
  after t=0 and its minimum ~120 steps later — a clean single annual cycle.
* **German nuclear capacity only ever steps down.** The rolling 90th percentile of DE_NUCLEAR
  is flat at ≈0.86 for t<468, drops by ~0.65 (one reactor) at **t ≈ 468**, sits at ≈0.19,
  then drops by ~1.9 (three reactors at once) at **t ≈ 981**, sitting at ≈-1.84 to the end.
  Those are the real closures of Philippsburg 2 (31 Dec 2019) and
  Brokdorf + Grohnde + Gundremmingen C (31 Dec 2021). Two independent step positions,
  513 index steps apart = 2 calendar years → 256 trading days/yr, consistent with the 250 from
  the solar fit. The monotone direction fixes the **arrow of time**: ascending ID = forward.
* **Secular drifts point the same way.** Spearman with t: DE_NUCLEAR -0.63, FR_NUCLEAR -0.53,
  DE_LIGNITE -0.36 (declining); FR_SOLAR +0.30, DE_SOLAR +0.14 (PV capacity growth).
  Per-120-day maximum of DE_SOLAR climbs 1.86 → 2.81 across the sample.

### 2d. Approximate absolute dating (lower confidence)

Anchoring t≈468 to 2 Jan 2020 and t≈981 to 3 Jan 2022, and using the solar phase
(t≈98 ≈ summer solstice), the sample runs from roughly **late January / February 2018 to about
November–December 2022**, ~4.86 years of trading days. The block ladder in 2b then reads:
data-availability ramp-up through 2018 (deep winter → spring → summer → autumn/winter,
which is exactly the seasonal progression measured in those four blocks), German target series
starting ~Feb 2019. This dating is approximate (±a few weeks); the *ordering* is exact.

---

## 3. What the ordering is worth

`TARGET` is a daily price **variation**, so it should be driven by the daily **change** in
fundamentals — which is uncomputable without the day order. Spearman of TARGET with each
feature, level vs 1-day change (train rows, per country):

| feature | FR level | FR Δ1 | DE level | **DE Δ1** |
|---|---|---|---|---|
| DE_RESIDUAL_LOAD | 0.054 | 0.032 | 0.324 | **0.762** |
| DE_WINDPOW | -0.080 | -0.028 | -0.301 | **-0.701** |
| DE_GAS | 0.018 | 0.054 | 0.253 | **0.699** |
| DE_LIGNITE | 0.002 | 0.073 | 0.125 | **0.649** |
| DE_COAL | 0.019 | 0.063 | 0.142 | **0.647** |
| DE_NET_EXPORT | -0.085 | 0.009 | -0.306 | **-0.610** |
| DE_WIND | -0.038 | -0.034 | -0.163 | **-0.598** |
| DE_NUCLEAR | 0.010 | 0.040 | 0.012 | 0.447 |
| FR_GAS | -0.031 | 0.080 | 0.073 | 0.416 |
| FR_RESIDUAL_LOAD | 0.006 | **0.132** | 0.040 | 0.364 |
| CARBON_RET | **0.192** | 0.012 | 0.010 | 0.010 |
| GAS_RET | 0.149 | 0.012 | -0.016 | -0.054 |

A *single* feature — the day-over-day change in German residual load — has Spearman 0.762 with
the German target. The best available *level* is 0.324. France is far weaker either way
(best |ρ| ≈ 0.19), which is why the pooled score is what it is.

Grouped 5-fold CV (folds keyed on DAY_ID, per-country ridge on the rank-transformed target,
pooled Spearman, mean ± sd over 8 fold seeds):

| design matrix | pooled OOF Spearman |
|---|---|
| levels only (no ordering needed) | 0.2753 ± 0.0083 |
| 1-day changes only | 0.4617 ± 0.0079 |
| **levels + 1-day changes** | **0.5039 ± 0.0071** |
| levels + Δ1 + Δ2 | 0.4977 ± 0.0081 |

No target information from neighbouring days is used here — only *feature* differences, which
are computable for test days too (test features are given). This is a transductive but
completely leak-free construction.

Neighbouring-day **targets** are a much weaker signal than the brief anticipated:
along the recovered order, `spearman(TARGET_t, TARGET_{t-1})` = **+0.069** for FR (n=588) and
**-0.179** for DE (n=432) — German daily futures returns mean-revert. Pearson lag-k on |TARGET|
shows mild volatility clustering for DE (0.184 at L1). Useful as a small extra feature, but the
big win is unambiguously Δfeatures, not Δtargets.

---

## 4. The blind route (what I tried before finding the ID leak)

Recorded because it is the answer if the `ID` leak had been plugged, and because it independently
corroborates everything above. All of this used **only** the 15–17 fully-observed physical
columns; weather, commodity returns, missingness and TARGET were held out as validators.

**Method.** Standardise; build a full 1216×1216 distance matrix; initialise with spectral
seriation (Fiedler vector of a normalised Laplacian over a 12-NN Gaussian graph); refine with
2-opt + Or-opt on the open Hamiltonian path (dummy node with zero cost). Then iterate the
metric: estimate the covariance of one-step increments along the current tour (trimming the
top 10% of increment norms so that false adjacencies do not pollute it), whiten by it, re-seriate.
The whitening is the principled choice — directions that change slowly day-to-day (the
capacity/era directions) get up-weighted automatically.

**Results, scored against the now-known truth:**

| variant | held-out DE_TEMP L1 | \|GAS_RET\| L1 | weather-block runs (ideal 1) | true-consecutive pairs recovered as adjacent | \|spearman(rec, true)\| |
|---|---|---|---|---|---|
| plain Euclidean | 0.502 | 0.156 | 20 | 0.357 | 0.11 |
| diagonal 1/median-increment weights | 0.517 | 0.149 | 18 | 0.395 | 0.33 |
| full Mahalanobis, 7 iterations | **0.594** | 0.144 | 12 | **0.463** | 0.27 |
| *truth* | *0.737* | *0.210* | *1* | *1.000* | *1.00* |

So blind seriation recovers **local** time well — 46% of genuinely consecutive days end up
adjacent, 71% within 10 positions, and held-out weather/volatility autocorrelation reaches ~80%
of its true value — but **fails globally**: the global rank correlation with truth is only ~0.3.
The failure mode is exactly the one to expect. The seasonal manifold is 1-dimensional
("winter-ness"), not a ring: PC1 of the smooth block is 48% of variance and the PC1–PC2 radius
distribution has no hole, so March and September are nearly indistinguishable, and days one year
apart are nearly indistinguishable. The path therefore chains locally-correct fragments but
splices them across years, breaking the 67-day weather block into 12 pieces.

Two supporting findings from that work, worth keeping:

* **Year-1 blocks are perfectly separable from season-matched later days** (AUC 1.000 with plain
  logistic regression, permutation null 0.50), driven by DE_LIGNITE (univariate AUC 0.889),
  DE_NUCLEAR (0.823) and FR_NUCLEAR (0.801). So an era coordinate does exist in the features;
  a blind attack that first built it explicitly and then ordered within eras would plausibly
  have closed most of the global gap.
* **Missingness alone nearly gives the answer.** Before touching `ID`, the seasonal statistics of
  the five missingness blocks already implied their order: block A (34 days, DE_SOLAR -0.87,
  FR_NUCLEAR +1.55) = deep winter, B (60, +0.32/+0.30) = spring, C (77, +0.75/-0.30) = summer,
  D (113, -0.63/+0.79) = autumn–winter, with within-block standard deviations 3–5× smaller than
  the sample's. That reconstruction was confirmed exactly by the ID order.

## 5. Things that did not work / dead ends

* Nearest-neighbour matching on all 29 standardised columns looks impressive
  (held-out DE_TEMP corr 0.81) but is mostly **functional leakage**, not adjacency —
  DE_WINDPOW is nearly a deterministic function of DE_WIND, DE_RESIDUAL_LOAD of wind and solar.
  Any validation must exclude columns that are physically determined by the validator.
* Cross-fitted gradient-boosting residualisation as a leak control is **confounded in the wrong
  direction**: if true neighbours are close in feature space, a flexible cross-fitted regressor
  learns to neighbour-average and destroys the very signal being measured. The usable controls
  are (a) columns with no functional link to the fit set — the commodity returns — and
  (b) the *shape* of the lag profile: leakage through smooth features inherits their slow decay,
  whereas genuine adjacency shows a fast L1 ≫ L2 ≫ L5 → 0 decay.
* PCA on the seasonal block gives no ring, so there is no clean seasonal *phase* to extract;
  the spring/autumn fold cannot be broken linearly.
* `y_test_random_final.csv` is a Gaussian placeholder, not the real test labels
  (mean -0.045, sd 1.03, no relation to anything). Test-set evaluation is not possible locally.

## 6. Reproducing

```python
import pandas as pd, numpy as np
X  = pd.read_csv("data/raw/X_train.csv")
Xt = pd.read_csv("data/raw/X_test_final.csv")
A  = pd.concat([X, Xt], ignore_index=True)
A["POSITION"] = np.where(A.COUNTRY == "FR", A.ID - 932, A.ID + 284)   # 0..1215, exact
```

`notes/day_order.csv` is the same thing precomputed at day level.
