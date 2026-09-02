# QRT Electricity Price Challenge

My solution to the [QRT / ENS "Can you explain the price of electricity?"](https://challengedata.ens.fr/challenges/97) data challenge.

**Final public score: 0.5658 Spearman, rank 6 / 1157.** The provided benchmark scores 0.1587.

The task: explain the daily change in 24h baseload electricity futures for France and Germany from same-day weather, generation, consumption and commodity data. The metric is Spearman rank correlation over the pooled test set.

## How the score was built

Every step below was measured with paired comparisons on identical fold assignments (randomised, day-grouped 5-fold, 15 to 20 seeds), and only kept if it won on nearly every seed. Pooled out-of-fold Spearman:

| step | CV | what it is |
|---|---|---|
| benchmark (pooled OLS, no country) | 0.194 | as provided |
| per-country models | 0.236 | see discovery 1 |
| rank-transform the target | 0.290 | see discovery 2 |
| **recover the true day ordering from `ID`** | **0.524** | see discovery 3 |
| cumulative commodity returns for France | 0.546 | see discovery 4 |
| neighbouring days' targets | 0.552 | Germany mean-reverts (lag-1 −0.21) |
| shrink France's prediction spread | 0.558 | see discovery 5 |
| LightGBM in the blend | 0.569 | see discovery 6 |
| pooled CatBoost in the blend | **0.572** | diversity member, OOF corr 0.84 with ridge |

Final CV 0.5716 ± 0.0050. Public leaderboard 0.5658, so the harness was well calibrated.

## The discoveries

Almost all of the score came from understanding the data rather than from model choice. In rough order of importance:

**3. The `ID` column encodes calendar order.** `DAY_ID` is genuinely shuffled, and I initially concluded from that that no time structure existed. But `ID` is not shuffled. Sorted by `ID`, French nuclear output autocorrelates at 0.980 and consumption at 0.965, against ~0.00 under `DAY_ID`. The tell that it's real time rather than an artifact: physical *levels* autocorrelate at 0.87–0.98 while commodity *returns* sit near zero, which is exactly how real daily data behaves. The exact mapping is two stacked blocks over one chronological index: France is `ID − 932`, Germany is `ID + 284`, and a day's two rows differ by exactly 1216. The organisers acknowledged this in their winners' seminar. The target is a price *change*, so first differences of the fundamentals along that ordering are what actually drive it: the lag-1 change in German residual load alone correlates 0.76 with the German target, against 0.32 for the best level. This one step took CV from 0.29 to 0.52.

**1. A day's French and German rows are bit-identical.** All 32 feature columns match exactly between the FR and DE row of the same day; only `COUNTRY` and the target differ, and the two targets correlate at just 0.12. A model without country can't tell them apart, which is why the benchmark is capped where it is. Everything is modelled per country.

**2. Least squares fights the metric.** The top 1% of days by |target| carry 31% of the squared-error weight but 1% of the rank positions. Rank-transforming the target before fitting is worth +0.04 and costs nothing, since Spearman only sees ranks.

**4. France is priced off levels, Germany off changes.** With time features Germany reached 0.77 within-country while France sat at 0.23, and nothing moved it. The reason is physical: France is nuclear-heavy, so its marginal price is set by the fuel cost of the marginal thermal unit, which is a *level*. The commodity columns are already daily returns, so differencing them is a second difference of the wrong thing. Summing them instead (trailing 5- and 20-day cumulative returns) reconstructs the level move, and lifts France from 0.23 to 0.28. `CARBON_RET_cum5` is the strongest single French feature in the whole dataset.

**5. The two countries should not be on the same scale.** The metric ranks FR and DE in one pool, and Germany is far better predicted. Compressing France's predictions toward their own mean by half lets the reliable German predictions occupy the extremes of the pooled ranking. This is the opposite of normalising the two countries onto a common scale, which loses badly (−0.048).

**6. "Added capacity always loses" was true, then false.** Under the original 29 features, every flexible model lost to ridge: gradient boosting scored 0.21 against 0.29, and a differentiable soft-Spearman objective did no better. I took that as a property of the dataset. It was a property of the signal-to-noise ratio. Once the time features existed, the byte-identical LightGBM config flipped from −0.013 to +0.023 against ridge. Fine-tuning a tabular foundation model showed the same split: it helped Germany and hurt France.

The France restriction is a smaller version of the same lesson. A permutation test at 5% family-wise error passes 9 of 29 features for Germany but only 3 for France, and France with all 29 overfits badly. So France gets `FR_WINDPOW`, `GAS_RET`, `CARBON_RET` and the derived features; Germany gets everything.

## The final model

Everything above assembled. This is exactly what `src/make_submission.py` runs.

**Features.** Along the recovered day ordering, per country:

| group | construction | FR | DE |
|---|---|---|---|
| base | the raw columns after dropping 3 exact sign-flips | 3 | 29 |
| time | lag-1 difference and deviation from the 7-day trailing mean, for all 29 base columns | 58 | 58 |
| cumulative returns | trailing 5- and 20-day sums of `GAS_RET`, `COAL_RET`, `CARBON_RET` | 6 | — |
| neighbour targets | weighted mean, mean absolute value, std, gap, and before/after means of the 3 nearest training days' targets | 6 | 6 |
| | **total** | **73** | **93** |

Trailing windows use `shift(1)`, so a day never enters its own window. Neighbour features only ever draw on training-set targets, and a day is never its own neighbour.

**Target.** Rank-transformed to (0, 1] before fitting, separately per country.

**Models.** Three, all fit on the rank target with median imputation:

| model | scope | settings |
|---|---|---|
| RidgeCV | per country | alpha chosen by leave-one-out over `logspace(-2, 4, 40)` |
| LightGBM | per country | `num_leaves=4, n_estimators=400, learning_rate=0.05, min_child_samples=20, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.7` |
| CatBoost | one model on both countries, with a country indicator | `depth=6, iterations=400, learning_rate=0.05, l2_leaf_reg=6` |

The pooled CatBoost is weaker alone (0.546 against 0.568 for the per-country blend) but its out-of-fold predictions correlate with ridge at only 0.84, so it adds a different view.

**Blend.** Each model's predictions are z-scored over the full test set, then combined as

```
0.8 × (0.7 × ridge + 0.3 × lightgbm) + 0.2 × catboost   =   0.56 / 0.24 / 0.20
```

Every LightGBM weight from 0.2 to 0.4 won on 20 of 20 seeds, and CatBoost at 0.1 or 0.2 won on 15 of 16; the values used sit in the middle of those plateaus rather than at the peaks.

**Post-processing.** France's predictions are shrunk toward their own mean by a factor of 0.5. Germany's are left alone.

**Validation.** 5-fold cross-validation grouped on `DAY_ID` so a day's two rows never straddle a split, with the group-to-fold assignment randomised over 16 to 20 seeds. Scored as one pooled Spearman over all out-of-fold predictions, the same quantity the leaderboard computes. Each component was compared against the previous best on identical fold assignments and kept only if it won on nearly every seed.

## What didn't work

Recorded so I don't retry them. FR−DE spread features, seasonal phase terms recovered from solar, calendar features (day-of-week, annual phase) built from the recovered ordering, missing-value indicators, row deletion, feature standardisation, feature rank transforms, gaussianised targets, sample weighting, robust losses, ElasticNet, partial pooling across countries, second differences, lagged levels, rolling feature volatility, EWMA deviations, longer cumulative windows, and every fitted stack, gate or weight optimiser. A per-row oracle choosing between ridge and LightGBM would score 0.68, and about half of that is stable structure, but a gate to learn it gets chance accuracy: the two models' errors correlate at 0.61, so knowing where one struggles tells you nothing about which to trust.

Time-series foundation models have nothing to find here either. With the ordering recovered, the target's own autocorrelation is two lags of mild mean reversion in Germany and nothing in France, which the neighbour-target features already capture.

## Reproduce it

```bash
uv sync
uv run python src/make_submission.py      # writes submissions/blend3_final.csv
```

Runs in under a minute on CPU. The output is byte-identical to the submitted file.

The exploratory work is in two notebooks. `notebooks/eda.ipynb` establishes discoveries 1 and 2 and the validation harness; `notebooks/data_prep.ipynb` settles preprocessing choices by measurement. Note that the EDA notebook's section 1.3 concludes no time structure exists, which was correct about `DAY_ID` and wrong about the dataset; discovery 3 came later, and I've left the original reasoning in place with a note rather than rewriting it.

## Layout

```
data/raw/               the four challenge CSVs
docs/challenge.md       the challenge statement
notebooks/
  eda.ipynb             exploratory analysis
  data_prep.ipynb       preprocessing decisions, each measured
  benchmark_qrt_en.ipynb  the organisers' benchmark, unchanged
src/
  qrt_prep.py           loading, validation, randomised grouped folds, scoring
  qrt_timeorder.py      the ID ordering, difference and cumulative-return features
  qrt_temporal.py       neighbouring-day target features, with a leakage guard
  make_submission.py    the full pipeline, end to end
submissions/
  blend3_final.csv      the submitted file
  benchmark_qrt.csv     the benchmark, for reference
```

Python 3.14, managed with `uv`. Main dependencies: pandas, scikit-learn, LightGBM, CatBoost, JAX (used for the permutation-null feature selection in the EDA).
