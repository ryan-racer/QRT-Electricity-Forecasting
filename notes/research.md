# QRT / ENS Challenge 97 — "Can you explain the price of electricity?"
## Competitive research: what other solvers actually did

Compiled 2026-09-01. Method: exhaustive GitHub API sweep (24 solution repos cloned and
read), ENS/Collège de France pages, French- and English-language web search, plus local
replication experiments against `data/raw/`.

Everything below is sourced. Where a claim is my own measurement rather than someone
else's, it is tagged **[measured here]**.

---

## 1. Source inventory

### 1.1 Solution repositories (all cloned and read)

| Repo | Reported score | What it actually does |
|---|---|---|
| [chantomkit/QRT2023](https://github.com/chantomkit/QRT2023) | **Public LB 0.295 / Private LB 0.251** | **Best documented public/private pair found.** 3-way split (FR-paired / DE-paired / "exotic" FR-only days), KS-test-driven feature selection, degree-2 interactions, target = global `rank()`, 9 model families, bootstrap-bagged prediction aggregation. |
| [hrandrIAga/ChallengeData_QubeRT](https://github.com/hrandrIAga/ChallengeData_QubeRT) | **0.2799, 50th/737 (final)** | **Model-based imputation** of every NaN column via per-column XGBoost (R²=0.92 for `DE_FR_EXCHANGE`, 0.96 for `FR_WIND`), then one pooled `RandomForestRegressor(n_estimators=200, min_samples_split=5, min_samples_leaf=6)`. Drops `RAIN`/`TEMP` entirely. |
| [97continuum/QRT---Electricity-Prices---2023](https://github.com/97continuum/QRT---Electricity-Prices---2023) (mirror: [talhajamal11](https://github.com/talhajamal11/QRT---Electricity-Prices---2023)) | 0.2749, 77/866 | Pooled LightGBM, grid-searched on **MSE** (not Spearman). No per-country model. Cubic-spline interpolation for NaNs. Weakest method of the high scorers — shows 0.27 is reachable with very little. |
| [hugoser10/explaining-electricity-prices](https://github.com/hugoser10/explaining-electricity-prices) | 0.2358 (final test) | Best *methodology* writeup. Merit-order domain analysis (French). **Clean spark/dark spread features.** Per-country models, rank target, 5 feature-selection schemes (VIF/MDI/permutation/SHAP/RFE) each calibrated by Kneedle elbow on Spearman CV. Reports CV: **DE 0.3921 (rank target), FR 0.2251**. |
| [lukasvoss/forecast_electricity_futures_QRT_challenge](https://github.com/lukasvoss/forecast_electricity_futures_QRT_challenge) | "top 9%" | Country-specific models: **ElasticNet for FR, LightGBM for DE**. Large engineered feature set (spreads, ratios, renewables bundles, commodity×load interactions). Reports LGBM val: **DE 0.4227, FR 0.0756**; FR ElasticNet 0.2038. |
| [MorganREN/Algothon_IC_2023](https://github.com/MorganREN/Algothon_IC_2023) (`QRT/public_data/QRT-Final.ipynb`) | 0.278 local; "highest score for this challenge" at Imperial Algothon | Per-country **linear SVR** (`kernel="linear", C=20, epsilon=0.511`) on the **top-10 features by \|corr\|** only. Extreme feature reduction + robust loss. |
| [PouyaFarivar/Predicting-24h-Changes-in-Electricity-Futures](https://github.com/PouyaFarivar/Predicting-24h-Changes-in-Electricity-Futures) | not reported | Only repo with a **differentiable Spearman objective** — `torchsort.soft_rank` wired into a custom LightGBM `fobj`. Code reproduced in §3.8. |
| [bncha/QRT-Electricity-ENS](https://github.com/bncha/QRT-Electricity-ENS) | not reported | `SPREAD_{feat} = DE_{feat} - FR_{feat}` for 11 base features, LightGBM. |
| [garou404/QRT-ENS-Data-Challenge-2023](https://github.com/garou404/QRT-ENS-Data-Challenge-2023) | in-sample only | Useful negative result: PolynomialFeatures(deg 2) → train 0.619 / holdout **−0.08** on FR. Degree-2 expansion on raw features is catastrophic on FR. |
| [justin-ayivi/ML_Electricity_Price](https://github.com/justin-ayivi/ML_Electricity_Price) | 0.284 CV | Winsorisation + ratio features. |
| Others read, nothing above benchmark worth extracting: [Romain-Berton](https://github.com/Romain-Berton/Data-Challenge-QRT-2023-Predicting-Electricity-Price), [charlesdezons](https://github.com/charlesdezons/QRT_Electricity_Prices_Challenge), [Buzon-coder](https://github.com/Buzon-coder/QRT-Electricity-price), [PedroPaesD](https://github.com/PedroPaesD/ElectricityPrice-QRT), [aa6dcc](https://github.com/aa6dcc/QRT-Electricity-Pricing-Hackathon), [taognt](https://github.com/taognt/Data_Challenge_QRT), [gpelleri](https://github.com/gpelleri/electricity_qrt), [nish0699](https://github.com/nish0699/Electricity-Price---QRT-Data-Challange---Team3) (CV 0.066–0.093), [FayssalHaddad](https://github.com/FayssalHaddad/ENS_Data_Challenge_By_QRT), [GeorgeA1357](https://github.com/GeorgeA1357/QRT-2023-Challenge), [paul-ada](https://github.com/paul-ada/qrt-electricity-challenge), [RJ-McReady](https://github.com/RJ-McReady/ENS-Data-Challenge-Electricity-Prices-Forecast), [Aurelien10-coder](https://github.com/Aurelien10-coder/Data_Challenge_QRT), [garou404](https://github.com/garou404/QRT-ENS-Data-Challenge-2023). | | |

### 1.2 Official / institutional sources

- [Challenge page](https://challengedata.ens.fr/challenges/97) — benchmark = plain linear regression, NaN→0, COUNTRY dropped, **public score 15.86%**.
- [Collège de France — 2023 winners' presentations](https://www.college-de-france.fr/en/agenda/seminar/learning-and-generation-by-random-sampling/presentations-of-the-2023-challenge-winners) (31 Jan 2024). Winners of *this* challenge: **Mohamed Ali Ben Amara** (1st), "mb" (2nd), **Amine Abdelkader** (3rd). Only an audio recording exists; **no slides, no code, no method description is public**. I could not find any writeup by any of the three.
- [Collège de France — Challenges 2023](https://www.college-de-france.fr/en/challenges-2023).
- [Challenge presentation video](https://www.youtube.com/watch?v=ZUCRlCx0wYE) — this is the *launch* video by QRT (presenter: Eduardo Peynetti), not a solution talk.
- [ENS Terms of Use](https://challengedata.ens.fr/terms_of_use) — **critical, see §4**: 2 submissions / 24h / team; test set split public/private; private board updated only 15 June and 15 December; **"public data freely accessible by any Participant may be used"**.

### 1.3 What does *not* exist

Searched hard, found nothing: no Medium/Substack writeup, no LinkedIn method post, no Reddit
r/MachineLearning thread, no Kaggle discussion, no arXiv paper citing this dataset, no
readable ENS forum. There is a [Kaggle data mirror](https://www.kaggle.com/datasets/emilefrotier/qrt-data-challenge)
with no public notebooks. **Assume no top-3 solution has ever been published.** Every score
above 0.30 in the wild is unexplained.

---

## 2. The single biggest structural find: the "exotic" FR-only segment

From `chantomkit`'s `data_processing.ipynb`, **verified locally [measured here]**:

```
TRAIN: 851 days. 643 days have BOTH FR and DE rows. 208 days have ONLY an FR row.
TEST:  365 days. 289 days have BOTH.              76 days have ONLY an FR row.
```

There is no DE-only day anywhere. So the data is **three** populations, not two:
`FR_paired (643)`, `DE_paired (643)`, `FR_exotic (208)`.

The missingness pattern separates them perfectly **[measured here]**:

| Segment | Columns with NaN |
|---|---|
| FR_paired / DE_paired | `DE_RAIN`, `FR_RAIN`, `DE_WIND`, `FR_WIND`, `DE_TEMP`, `FR_TEMP` (47 each) |
| **FR_exotic** | `DE_FR_EXCHANGE`, `FR_DE_EXCHANGE` (25); `FR_NET_EXPORT`, `FR_NET_IMPORT` (70); `DE_NET_EXPORT`, `DE_NET_IMPORT` (124) |

`chantomkit` ran KS tests on 532 degree-2 features: **0** differ between FR_paired and
DE_paired (confirming the bit-identical-features observation), but **403 of 532** differ
between exotic and non-exotic. The exotic rows are a genuinely different distribution.

Feature→target Spearman by segment **[measured here]** — the drivers really do differ:

| Feature | FR_paired | DE_paired | FR_exotic |
|---|---|---|---|
| `DE_RESIDUAL_LOAD` | 0.067 | **0.324** | 0.015 |
| `DE_WINDPOW` | −0.079 | **−0.301** | −0.090 |
| `DE_NET_EXPORT` | −0.053 | **−0.306** | −0.201 |
| `COAL_RET` | 0.015 | −0.021 | **0.193** |
| `FR_NUCLEAR` | 0.054 | −0.009 | **−0.169** |
| `DE_SOLAR` | 0.006 | 0.021 | **0.156** |
| `CARBON_RET` | 0.185 | 0.010 | **0.208** |
| `GAS_RET` | **0.151** | −0.016 | 0.142 |

`COAL_RET`, `FR_NUCLEAR` and `DE_SOLAR` carry signal on exotic FR days and essentially none
on paired FR days. This is genuinely new information relative to a plain FR/DE split.

**Caveat [measured here]:** naively fitting a third RidgeCV on the 208 exotic rows *hurt*
(0.2801 → 0.2746). The segment is real but too small for an independent model. See §3.1 for
the framing that should work instead.

---

## 3. Concrete techniques, ranked by expected value

Ranking accounts for what `src/qrt_softrank.py` and the existing pipeline already do
(per-country ridge on a rank target, ~0.29 pooled OOF with FR reduced to 3 features).

### 3.1 Exploit the exotic segment — real structure, but I could not yet convert it
Rather than a separate model on 208 rows:
- Add a binary `IS_EXOTIC` / `HAS_DE_COUNTERPART` column.
- Add per-column missingness indicators (`MISS_DE_NET_EXPORT`, etc.).
- Fit **one FR model on all 851 FR rows** but interact the exotic flag with the features whose
  sign flips (`COAL_RET`, `FR_NUCLEAR`, `DE_SOLAR`): `x * IS_EXOTIC`.

**Honest status — three attempts, all null or negative [measured here]:**

| attempt | result |
|---|---|
| separate RidgeCV on the 208 exotic rows (3-segment) | 0.2801 → 0.2746 (**negative**) |
| 3-segment on top of bag-20 | 0.2779 → 0.2728 (**negative**) |
| `IS_EXOTIC` + 5 interaction terms, bag-20, all features | 0.2779 → 0.2780 (**null**) |
| missingness indicator columns | 0.2801 → 0.2771 (**null**) |

So: the segment is unambiguously real (§2 — 208/76 rows, disjoint missingness pattern, 403/532
features distributionally different, drivers that flip sign), but **no naive exploitation of it
has paid yet.** The untested version is the one that matches how FR actually behaves: apply it
*inside* a reduced FR feature set (your FR=3 configuration) rather than bolted onto 30
features, where 5 extra interaction columns are lost in the noise. Treat this as the most
promising *unexploited* structure in the data, not as a banked win.
Origin: segment discovered by `chantomkit`; interaction framing and all measurements are mine.

### 3.2 Bagging + quantile pooling — small, cheap, *measured*, variance-reducing
Two independent stackable wins **[measured here]**, all-features per-country ridge on rank target:

| Config | Pooled OOF Spearman (8 seeds) |
|---|---|
| per-country ridge, uniform-rank pooling | 0.2713 ± 0.0073 |
| + quantile pooling | 0.2757 ± 0.0074 |
| + bag-30 (rank-averaged) | 0.2745 ± 0.0052 |
| **+ both** | **0.2801 ± 0.0055** |

- **Bagging**: fit 30 RidgeCVs on 70% subsamples, convert each prediction to ranks, average
  the ranks. `chantomkit` measured the same effect much larger on a holdout: DE ridge
  0.361 → **0.417** with `n_bootstrap=100, bootstrap_fraction=0.7`. Note the σ drop from
  0.0073 → 0.0052 — this makes CV decisions more reliable, which is worth as much as the score.
- **Quantile pooling**: the metric is pooled over FR+DE, so how you splice the two
  per-country score vectors matters. Instead of mapping each country's predictions to
  uniform [0,1] ranks, map them through **that country's own empirical target quantiles**:
  ```python
  q = rankdata(pred_country) / (len(pred_country) + 1)
  out[country_idx] = np.quantile(y_train_country, q)
  ```
  Oracle check **[measured here]**: perfect within-country ranks pooled as uniform[0,1] score
  only **0.9916**; pooled through own-country quantiles score **1.0000**. A simulation at
  realistic skill (ρ_FR=0.22, ρ_DE=0.40) gives uniform 0.2992, z-score 0.2980,
  quantile **0.3077**. Free ~+0.008, and it is not noise — it is a genuine metric artifact.

### 3.3 Clean spark / clean dark spread features — strong domain prior
From `hugoser10/data_prep.py`, the most defensible feature engineering anyone did:
```python
GAS_EFFICIENCY, GAS_EMISSION_FACTOR   = 0.5, 0.4
COAL_EFFICIENCY, COAL_EMISSION_FACTOR = 0.5, 1.0
df['MARGINAL_GAS']  = GAS_EFFICIENCY  * df['GAS_RET']  + GAS_EMISSION_FACTOR  * df['CARBON_RET']
df['MARGINAL_COAL'] = COAL_EFFICIENCY * df['COAL_RET'] + COAL_EMISSION_FACTOR * df['CARBON_RET']
df.drop(columns=['GAS_RET', 'COAL_RET', 'CARBON_RET'], inplace=True)
```
Rationale (merit order): the clearing price is set by the marginal thermal plant's cost,
which is fuel cost / efficiency **plus** the CO₂ cost of its emission factor. A CCGT is
~50% efficient at ~0.4 tCO₂/MWh; hard coal ~50% at ~1.0 tCO₂/MWh. So these two composites
are the actual economic drivers, and `GAS_RET`/`COAL_RET`/`CARBON_RET` individually are
noisy projections of them. `hugoser10`'s SHAP/PDP analysis found `MARGINAL_GAS` and
`MARGINAL_COAL` were the **top two features for France** — exactly the segment where you have
only 3 significant features. Worth trying both as replacements and as additions.
**[measured here]**: as a pure addition to an all-feature model, +0.007 in one config,
neutral in another. The real test is inside your reduced-FR feature set.

### 3.4 Model-based imputation of the missing columns
`hrandrIAga` (0.2799) trained a per-column XGBoost to fill each NaN column from the
always-complete columns. Reported R²: `DE_FR_EXCHANGE` 0.92 (vs 0.82 for linear),
`FR_WIND` 0.96. Also note the free identity **`FR_DE_EXCHANGE = −DE_FR_EXCHANGE`** exactly,
so one of them is pure redundancy.
Fit on complete rows only (1276 of 1494), apply to train and test.
**[measured here]**: ExtraTrees-based version gave 0.2745 → 0.2785, i.e. ≈ +0.004 over
plain bagging but no better than quantile pooling alone. Modest; matters more if you keep
the exchange columns for the exotic rows rather than dropping them.

### 3.5 Aggressive per-country feature reduction + robust linear loss
`MorganREN` (best-in-challenge at the Imperial Algothon) used only the **top 10 features by
|correlation| per country**, then `SVR(kernel="linear", C=20, epsilon=0.511)`. The large
epsilon makes it an ε-insensitive (robust) regression — it ignores residuals inside a band,
which suits fat-tailed price returns. This is a different robustness axis from ridge and
worth an ensemble slot even if it doesn't win alone. `chantomkit` and `hugoser10` both also
found Huber competitive. Your FR=3-feature result is the same discovery from another angle.
**[measured here]**: on all features with bag-20, `LinearSVR` scored 0.2760 vs ridge 0.2779
(no gain), and `SVR(kernel='rbf')` collapsed to 0.2137. The value, if any, is in the
*combination* MorganREN used — top-10 features **and** a robust loss — not in the loss alone.
Low priority.

### 3.6 Cross-country spread and ratio features
Independently invented by `bncha`, `lukasvoss` and `97continuum`. Cheap to test:
```python
SPREAD_{f}       = DE_{f} - FR_{f}          # for CONSUMPTION, GAS, COAL, HYDRO, NUCLEAR,
                                            # SOLAR, WINDPOW, RAIN, WIND, TEMP, RESIDUAL_LOAD
{c}_NET_BALANCE  = {c}_NET_IMPORT - {c}_NET_EXPORT
{c}_RENEWABLES   = {c}_WINDPOW + {c}_SOLAR + {c}_HYDRO
{c}_RESIDUAL_RATIO = {c}_RESIDUAL_LOAD / ({c}_CONSUMPTION + 1e-6)
EXCHANGE_TOTAL   = DE_FR_EXCHANGE + FR_DE_EXCHANGE     # ≈ 0, so a data-quality flag
```
plus commodity×load interactions (`CARBON_RET * FR_RESIDUAL_LOAD`, `GAS_RET * FR_CONSUMPTION`).
**[measured here]**: the plain DE−FR spread block was *negative* on its own (0.2676 → 0.2410).
Do not add the whole block. The economically motivated few (`{c}_NET_BALANCE`,
`{c}_RESIDUAL_RATIO`, and the commodity×residual-load interactions, which encode "how much
does a fuel price move matter *given* how much thermal generation is needed") are the ones to
test individually.

### 3.7 Feature selection calibrated *on Spearman CV*, not on a fixed threshold
`hugoser10`'s framework, which produced their best numbers (DE 0.3921 / FR 0.2251), sweeps a
selection threshold and picks it by the **Kneedle elbow of the Spearman-CV curve**, with the
selection redone *inside each fold*. Five selectors: VIF (multicollinearity), MDI, permutation
importance, SHAP, RFE-with-Ridge. Their winner on both countries was `vif_rfe_ridge`
(VIF prune → RFE with a Ridge estimator). Also: rank target beat raw target on **every**
model × country combination they tested, corroborating your +0.04.

### 3.8 Differentiable Spearman surrogate — likely a dead end, already tested
`PouyaFarivar` implements it as a LightGBM custom objective via `torchsort`:
```python
def spearman_loss_lgb(ytrue, ypred):
    def corrcoef(target, pred):
        pred_n   = pred   - pred.mean();   pred_n   = pred_n   / pred_n.norm()
        target_n = target - target.mean(); target_n = target_n / target_n.norm()
        return -1 * (pred_n * target_n).sum()
    def differentiable_spearman(target, pred, regularization="l2", regularization_strength=1.0):
        pred = torchsort.soft_rank(pred, regularization=regularization,
                                   regularization_strength=regularization_strength)
        return corrcoef(target, pred / pred.shape[-1])
    ypred_th = torch.tensor(ypred.reshape(1, -1), requires_grad=True)
    ytrue_th = torch.tensor(ytrue.reshape(1, -1))
    loss = differentiable_spearman(ytrue_th, ypred_th, regularization_strength=1e-2)
    g = torch.autograd.grad(loss, ypred_th)[0].cpu().detach().numpy()
    return g[0], np.ones(g.shape)[0]   # gradient, unit Hessian
```
**Your `src/qrt_softrank.py` already tested this family (JAX sigmoid soft-rank) and it lost:
0.2714 vs 0.2904 for ridge on a rank target.** The `torchsort` variant differs only in the
soft-rank construction (isotonic-regression projection vs pairwise sigmoid), so expect the
same outcome. Recorded for completeness — do not re-spend time here.

### 3.9 Negative results worth not repeating
- **Degree-2 polynomial expansion on raw features**: `garou404` got FR train 0.619 → holdout
  **−0.08**. `chantomkit` only made it work by pairing it with heavy KS-drift filtering and
  correlation-based pruning down to ~75 of 532 features.
- **GBDT tuned on MSE**: `97continuum` grid-searched LightGBM/XGB on `neg_mean_squared_error`.
  Their prediction std was 0.31 vs a target std of 1.03 — massively shrunk, which MSE rewards
  and Spearman does not care about. If you use GBDT at all, tune on Spearman.
- **A single pooled model**: consistently ~0.25–0.27 ceiling across repos.
- **Whole DE−FR spread block** **[measured here]**: −0.027.

---

## 4. Are the 0.5+ scores legitimate?

**Verdict: most likely a genuine ~0.35–0.42 signal inflated by heavy public-leaderboard
selection. Not de-anonymisation. External data is legal but there is no evidence anyone used it.**

**Fact 1 — the platform invites LB overfitting.** [ENS Terms of Use](https://challengedata.ens.fr/terms_of_use):
2 submissions per 24 h per team; the test set is split public/private; *"by default, the best
submission on the public set is taken for rankings"*; the private board is refreshed only
twice a year. The challenge opened January 2023 and **is still open**. That is ~1,900
submissions available to a participant who has been probing since launch.

**Fact 2 — the noise is large enough.** With 654 test rows and a public half of ~327, the
standard error of a Spearman estimate is ≈ 1/√326 ≈ **0.055**. The maximum of ~2,000
submissions drawn around a true skill τ sits at roughly τ + 3.4σ ≈ τ + 0.19. A true 0.42
therefore produces a public 0.61 without any cheating. On the private side, taking the max
over 866 teams gives ≈ +2.9σ ≈ +0.16, so a private top of 0.53 is consistent with a best
true skill around **0.37–0.40**.

**Fact 3 — the de-anonymisation hypothesis predicts the wrong number.** `DAY_ID` is shuffled
but the features are real normalised physical quantities (FR/DE load, generation by fuel,
cross-border flows). All of that is free on the ENTSO-E Transparency Platform, and the ENS
rules **explicitly permit public external data**. So a determined solver could in principle
match each `DAY_ID` to a calendar date and then look up the actual EEX/EPEX baseload futures
settlement change — which is *the target itself*. That attack yields ≈ **1.0**, not 0.61.
Nobody scores near 1.0. Therefore the top scores are almost certainly **not** de-anonymised
targets. (A partial/noisy match is conceivable but would be a strange amount of work to stop
halfway.) I found no post, repo or paper describing such an attack for this challenge.

**Corroborating evidence from the honest repos**: `chantomkit` is the only source with both
numbers — **public 0.295, private 0.251**. A −0.044 public→private drop on a mid-table
submission is exactly the selection-noise signature, and it also means *your CV should be
compared against private-style numbers, not public ones*.

**Practical consequences**
1. Your ~0.30 local CV is **not** far off the real frontier. Realistic upside is ~+0.05–0.10,
   not +0.30. Do not chase 0.61.
2. Treat the public LB as noisy with σ≈0.055. Do **not** select models on it. A 0.02 public
   gain is meaningless; trust repeated-seed OOF CV instead.
3. Since the private board only updates 15 June / 15 December, an LB-probing strategy is
   available to you too but is a poor use of effort relative to variance reduction.
4. External public data (ENTSO-E) is *legal*. You have said you will not de-anonymise, and
   the evidence says it wouldn't explain the leaderboard anyway.

---

## 5. Local replication results (all [measured here])

Per-country RidgeCV on a per-country rank target, 5-fold, mean ± sd over repeated seeds,
pooled OOF Spearman on the 1494 train rows. All features, no selection (so the absolute
level is below your FR-reduced 0.2904 — read the *deltas*).

```
baseline: per-COUNTRY ridge, raw y            0.2074 ± 0.0136
+ rank target                                 0.2676 ± 0.0074   (+0.060  confirms your +0.04)
+ 3-segment (FR/DE/FR_EXO), naive             0.2481 ± 0.0144   (-0.020  NEGATIVE, see 3.1)
+ marginal fuel feats (spark/dark)            0.2554 ± 0.0123
+ DE-FR spread block                          0.2410 ± 0.0170   (-0.027  NEGATIVE)

per-COUNTRY, uniform pooling                  0.2713 ± 0.0073
+ quantile pooling                            0.2757 ± 0.0074   (+0.004)
+ bag-30 rank-averaged                        0.2745 ± 0.0052   (+0.003, sd -29%)
+ both                                        0.2801 ± 0.0055   (+0.009)
+ missingness indicators                      0.2771 ± 0.0053   (nothing)
+ model-based imputation (ExtraTrees)         0.2785 ± 0.0062   (nothing on top of the above)
3-segment + bag30 + quantile                  0.2746 ± 0.0058   (still negative)

ridge + bag20 (reference)                     0.2779 ± 0.0062
+ IS_EXOTIC + 5 interaction terms             0.2780 ± 0.0051   (null)
+ 3-segment                                   0.2728 ± 0.0048   (negative)
LinearSVR + bag20                             0.2760 ± 0.0065   (no gain over ridge)
RBF-SVR + bag20                               0.2137 ± 0.0082   (bad)
ridge top-8 by |spearman| + quantile pool     0.2677 ± 0.0061   (worse than all-features;
                                                                 your permutation-null FR=3
                                                                 selection is better)
```

Oracle / metric-artifact checks:
```
COUNTRY dummy alone, pooled Spearman                       0.0271   (country offset is worth ~nothing)
perfect within-country ranks -> uniform[0,1] pooling       0.9916   (pooling loses 0.008 of a perfect model)
perfect within-country ranks -> own-country quantiles      1.0000
simulated rho_FR=0.22 rho_DE=0.40: uniform / z / quantile  0.2992 / 0.2980 / 0.3077
```

---

## 6. Reference numbers from other solvers, for calibration

| Source | FR | DE | Pooled |
|---|---|---|---|
| hugoser10 CV (rank target, vif_rfe_ridge) | 0.2251 | **0.3921** | — |
| lukasvoss validation (LGBM) | 0.0756 | **0.4227** | 0.2491 |
| lukasvoss validation (ElasticNet, FR) | 0.2038 | — | — |
| chantomkit validation, bootstrap-bagged ridge | 0.175 | **0.417** | — |
| chantomkit validation, "exotic" segment | — | — | 0.24–0.31 |
| garou404 5-fold CV, plain linear | 0.092 | 0.301 | — |
| **Consensus** | **~0.20–0.23 is the FR ceiling** | **~0.39–0.42 is the DE ceiling** | ~0.28–0.30 |

Everyone lands in the same place: **DE is a solved, well-behaved problem (~0.40) and FR is the
bottleneck (~0.22)**. Marginal effort is far better spent on France. `hugoser10`'s PDP analysis
explains why: DE responds to a sharp `DE_WINDPOW` threshold with homogeneous ICE curves, while
FR's `MARGINAL_GAS` effect is strongly context-dependent (heterogeneous ICE), i.e. FR has a
regime-switching structure a single linear model cannot capture. That, plus the exotic segment,
is the case for §3.1.

---

## 7. General literature: maximising Spearman, ranking losses, small tabular data

Sourced from Numerai's public research (Numerai is scored on rank correlation on small-ish
tabular data — the closest public analogue), Kaggle Spearman-metric competitions, and the
soft-ranking literature. Several claims were re-measured directly on `data/raw/`
**[measured here]**.

### 7.1 A correlation loss is a no-op for linear models — this closes §3.8
For a linear model, maximising `pearson(rank(y), Xb)` and least-squares regression of
`rank(y)` on `X` give the **same predictions up to a positive affine transform**. Verified
numerically: cosine between the two coefficient vectors = 1.0, Spearman between the two
prediction vectors = 1.0. Since Spearman is invariant to monotone maps, they score identically.

**Consequence: ridge on a rank-transformed target *is already* the correlation objective.**
There is no headroom left in "optimise Spearman directly" for a linear model. A soft-rank
loss only adds ranking on the *prediction* side, which matters solely when predictions have
outliers — better fixed with a robust loss. This explains cleanly why `src/qrt_softrank.py`
lost to plain ridge, and why the `torchsort` variant in §3.8 would too.
Also note from [torchsort](https://github.com/teddykoker/torchsort)'s own numbers: at
`regularization_strength=1e-3` the gradient is **identically zero**; at `1e-2` gradients
appear but accuracy degrades. Numerai's
[differentiable-Spearman thread](https://forum.numer.ai/t/differentiable-spearman-in-pytorch-optimize-for-corr-directly/2287)
reports it performing poorly on MLPs and "disaster" results in LightGBM.

### 7.2 Learning-to-rank objectives are actively harmful here **[measured here]**
There is only one query group, so LambdaRank's position-discounting is meaningless:

| objective | pooled OOF Spearman |
|---|---|
| LightGBM regression on rank target | 0.2457 |
| LightGBM `lambdarank`, 8 bins | 0.1453 |
| LightGBM `lambdarank`, 32 bins | 0.1385 |
| LightGBM `rank_xendcg`, 8 bins | 0.1643 |

NDCG is top-heavy; Spearman weights the whole distribution symmetrically — wrong objective.
Corroborated by Numerai's [Learning to Rank thread](https://forum.numer.ai/t/learning-to-rank/454),
five years of attempts with no win. **Do not try `rank:pairwise` / `lambdarank`.**

### 7.3 Target transform: uniform rank beats gaussian rank **[measured here]**
Ridge, 29 features, pooled OOF:

| target transform | score |
|---|---|
| raw target | 0.2385 |
| **uniform rank** `rankdata(v)/len(v)` | **0.2739** |
| gaussian rank `norm.ppf(...)` | 0.2715 |
| winsorise at 2 sd | 0.2553 |

`norm.ppf` re-inflates exactly the tails the rank transform was suppressing. Keep uniform rank.

### 7.4 Model family sweep — bagged ridge wins **[measured here]**
Uniform-rank target:

| model | score |
|---|---|
| ridge α=10 | 0.2777 |
| ridge α=100 | 0.2794 |
| ridge α=10 + QuantileTransformer features | 0.2779 |
| PLS n=3 / n=6 | 0.2667 / 0.2780 |
| Huber | 0.2763 |
| **`BaggingRegressor(Ridge(), n_estimators=25, max_samples=0.7, max_features=0.7)`** | **0.2836 ± 0.0058** |

Bagged ridge is the best single model **and roughly halves seed variance** (±0.0058 vs
±0.0095 for plain ridge). This independently confirms §3.2 and `chantomkit`'s bootstrap
aggregation. **PLS, Huber, gaussian-rank features and QuantileTransformer all add nothing** —
skip them.

From Numerai, the one transferable GBDT hyperparameter is **`colsample_bytree ≈ 0.1`**: with
correlated low-SNR features it forces the ensemble to spread bets instead of re-exploiting the
few strongest. Their `min_data_in_leaf=10000` is safe only at millions of rows — do not copy.
**Numerai "era boosting" is refuted** — absent from every version of the official repo,
disowned by its author, and the circulated code has real bugs. See
[Era Boosted Models](https://forum.numer.ai/t/era-boosted-models/189).

### 7.5 Ensembling: rank globally, never per-country **[measured here]**
Averaging 4 models (ridge α=10, ridge α=300, PLS-6, bagged ridge):

| scheme | score |
|---|---|
| raw average | 0.2805 |
| z-score average | 0.2807 |
| global rank average | 0.2802 |
| **per-country rank average** | **0.2643** |

Per-country rank averaging is clearly worse — it destroys cross-country calibration. This
**contradicts Numerai's "always rank per era before averaging" rule**, which applies only
because Numerai scores per-era. Our metric is one pooled Spearman over all 654 test rows, so
Numerai's per-era advice does not transfer. Note also that no ensemble beat the single bagged
ridge — the candidate models are too correlated to help.

### 7.6 Post-processing / discretisation does not transfer
The [Google QUEST 1st-place code](https://github.com/oleg-yaroshevskiy/quest_qa_labeling/blob/master/step11_final/blending_n_postprocessing.py)
buckets predictions to match the training target's discrete value frequencies. That worked
because QUEST targets had heavy ties. Our target is continuous and tie-free, so `rank(y)` is a
full permutation and introducing ties is neutral-to-harmful in expectation. More generally:
**Spearman is invariant to monotone maps, so no within-group post-processing can ever help.**
The only exploitable post-processing is *between-group* — §7.7.

### 7.7 Cross-country calibration: real, but a substitute for bagging **[measured here]**
The one place post-processing can pay, because it is the only thing that is not a
within-group monotone map. Two free parameters on FR relative to DE:
```python
p[fr] = (p[fr] - p[fr].mean()) * scale + p[fr].mean() + shift
```
Tuned on one half, evaluated on the held-out half, 120 splits: **0.2737 → 0.2830, +0.0093,
winning 108/120.** Best parameters `scale ≈ 0.6, shift ≈ −0.03` — i.e. **shrink the weaker
country (FR) toward the pooled centre.** Theory agrees: confident extremes coming from a
low-skill group hurt a pooled ranking.

**Critical caveat: it does not stack with bagging.** Bagged ridge + calibration gave a
held-out delta of **−0.0024 (52/120 wins)**. Bagging already shrinks the noisier country's
predictions automatically. They are substitutes — pick one, and bagging is better-behaved.

This is the same lever as the quantile pooling in §3.2 (my measurement there, +0.004 on top of
bagging, is within noise of zero). **Honest summary: cross-country calibration is a genuine
+0.005–0.01 effect that is largely absorbed by bagging.** Use bagged ridge and do not expect
to collect both.

### 7.8 Sources
[torchsort](https://github.com/teddykoker/torchsort) ·
[Numerai differentiable Spearman](https://forum.numer.ai/t/differentiable-spearman-in-pytorch-optimize-for-corr-directly/2287) ·
[Numerai Learning to Rank](https://forum.numer.ai/t/learning-to-rank/454) ·
[QUEST 1st place post-processing](https://github.com/oleg-yaroshevskiy/quest_qa_labeling/blob/master/step11_final/blending_n_postprocessing.py) ·
[numerai-tools scoring.py](https://github.com/numerai/numerai-tools/blob/master/numerai_tools/scoring.py) ·
[Era Boosted Models](https://forum.numer.ai/t/era-boosted-models/189) ·
[Ubiquant 3rd place](https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/hyd-3rd-place-solution-5-seeds-ensemble-transforme) ·
[Hull Tactical CV diagnostic](https://www.kaggle.com/competitions/hull-tactical-market-prediction/discussion/608088) ·
[Concordance-correlation loss vs MSE](https://arxiv.org/abs/2003.10724)

---

## 8. Ranked shortlist

Ordered by measured evidence, strongest first.

1. **Bagged ridge** — `BaggingRegressor(Ridge(), n_estimators=25-100, max_samples=0.7,
   max_features=0.7)`, per country, rank target. **Measured best single model (0.2836, §7.4)**,
   confirmed independently in §3.2, and the largest effect anyone else reports
   (`chantomkit` holdout: DE ridge 0.361 → 0.417 with 100 bootstraps at 70%). It also roughly
   halves seed variance, which makes every subsequent CV decision more trustworthy. And it
   subsumes the calibration gain in (2). **Do this first.**
2. **Cross-country calibration** — two parameters, `p[FR] = (p[FR] - mean) * scale + mean + shift`,
   tuned on train folds; best ≈ `scale 0.6, shift -0.03`, i.e. shrink France toward the pooled
   centre. **+0.0093 held out, winning 108/120 splits (§7.7).** Equivalent to the quantile
   pooling in §3.2. **Caveat: does not stack with (1)** — bagging already shrinks the noisier
   country. Use this only if you stay on plain ridge.
3. **Clean spark / clean dark spread features** (§3.3) —
   `MARGINAL_GAS = 0.5*GAS_RET + 0.4*CARBON_RET`, `MARGINAL_COAL = 0.5*COAL_RET + 1.0*CARBON_RET`,
   dropping the three raw commodity returns. `hugoser10`'s SHAP/PDP found these the **top two
   features for France**, the segment where you are down to 3 significant features. Untested
   inside a reduced FR set — that is the experiment to run.
4. **Model-based imputation** of the exchange/weather NaNs (§3.4) — `hrandrIAga`'s 0.2799 route.
   Per-column XGBoost/ExtraTrees fitted on the 1276 complete rows; R² 0.92 for
   `DE_FR_EXCHANGE`, 0.96 for `FR_WIND`. Measured +0.004; modest on its own, but the exotic FR
   rows are precisely the NaN-heavy ones, so it compounds with (5). Free win: use the exact
   identity `FR_DE_EXCHANGE = -DE_FR_EXCHANGE`.
5. **The exotic FR segment, done properly** (§2, §3.1) — the only genuinely new structural
   information found anywhere, sitting on the half of the problem that is the bottleneck for
   every solver (FR ~0.22 vs DE ~0.40). But three naive exploitations measured null or
   negative. Only worth another attempt *inside* your reduced FR feature set. High variance:
   either the best idea here or nothing.

**Do not spend time on:** soft-rank / differentiable-Spearman losses (§7.1 — mathematically a
no-op for linear models, already refuted in your own `qrt_softrank.py`); `lambdarank` /
`rank:pairwise` (§7.2 — 0.14 vs 0.25); gaussian-rank target, PLS, Huber, QuantileTransformer
features (§7.4); prediction discretisation (§7.6); per-country rank averaging in ensembles
(§7.5); LinearSVR/RBF-SVR (§3.5); degree-2 polynomial expansion on raw features (§3.9); the
whole DE-FR spread block (§3.6); chasing the 0.61 public leaderboard (§4).
