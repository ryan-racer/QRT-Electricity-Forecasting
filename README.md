# Explaining Electricity Prices

**QRT × ENS Data Challenge** — explain daily changes in French and German electricity futures from same-day weather, generation, consumption and commodity data. Scored by Spearman rank correlation over the pooled test set.

![Spearman](https://img.shields.io/badge/Spearman-0.5658-2ea44f?style=flat-square)
![Rank](https://img.shields.io/badge/rank-6%20%2F%201157-1f6feb?style=flat-square)
![Python](https://img.shields.io/badge/python-3.14-3776ab?style=flat-square)

Nearly all of the score came from understanding the data rather than from picking models. The single largest step was noticing that the `ID` column, unlike `DAY_ID`, was never shuffled.

## Quick start

```bash
git clone https://github.com/ryan-racer/QRT-Electricity-Forecasting
cd QRT-Electricity-Forecasting
uv sync
uv run python src/make_submission.py      # → submissions/blend3_final.csv
```

The output is byte-identical to the submitted file. Runs in about 15 seconds on CPU.

## Results

Pooled out-of-fold Spearman, day-grouped 5-fold, 15–20 randomised seeds. Each step was kept only if it beat the previous one on nearly every seed, compared on identical folds.

```
model each country separately  0.236  █████████████████████
rank-transform the target      0.290  █████████████████████████
recover the day ordering (ID)  0.524  ██████████████████████████████████████████████
cumulative returns, France     0.546  ████████████████████████████████████████████████
neighbouring days' targets     0.552  ████████████████████████████████████████████████
shrink France's spread         0.558  █████████████████████████████████████████████████
+ LightGBM                     0.569  ██████████████████████████████████████████████████
+ pooled CatBoost              0.572  ██████████████████████████████████████████████████
```

Final CV **0.5716 ± 0.0050** → public leaderboard **0.5658**.

## What I found

1. **`ID` encodes calendar order.** `DAY_ID` is shuffled; `ID` is not. Sorted by `ID`, French nuclear output autocorrelates at 0.98 against ~0 under `DAY_ID`. Since the target is a price *change*, day-over-day differences of the fundamentals are what drive it. This one step took CV from 0.29 to 0.52.
2. **A day's French and German rows are bit-identical.** Only `COUNTRY` and the target differ, and the two targets correlate at just 0.12. Everything is modelled per country.
3. **Least squares fights the metric.** 1% of days carry 31% of the squared-error weight. Rank-transforming the target is worth +0.04 for free.
4. **France is priced off levels, Germany off changes.** France is nuclear-heavy, so its price follows fuel *cost*. Summing the commodity returns over 5 and 20 days reconstructs that level and lifts France from 0.23 to 0.28.
5. **Don't put the two countries on one scale.** Germany is far better predicted, so halving France's prediction spread lets the reliable German predictions occupy the extremes of the pooled ranking. Normalising both onto a common scale loses 0.048.
6. **"Added capacity always loses" was true, then false.** Under the original 29 features every flexible model lost to ridge. Once the time features existed, the identical LightGBM config flipped from −0.013 to +0.023 against it. Signal-to-noise, not model class.

## The model

Exactly what `src/make_submission.py` runs.

| features, per country | FR | DE |
|---|---:|---:|
| base columns (3 sign-flip duplicates dropped) | 3 | 29 |
| lag-1 difference + deviation from 7-day trailing mean, all 29 columns | 58 | 58 |
| 5- and 20-day cumulative sums of the three commodity returns | 6 | — |
| level / volatility / spread of the 3 nearest training days' targets | 6 | 6 |
| **total** | **73** | **93** |

France gets three base columns because a permutation test passes only 3 of 29 for it (9 for Germany); with all 29 it overfits badly.

| model | scope | settings |
|---|---|---|
| RidgeCV | per country | α by LOO over `logspace(−2, 4, 40)` |
| LightGBM | per country | `num_leaves=4 · n_estimators=400 · lr=0.05 · min_child_samples=20 · λ=1 · subsample=0.8 · colsample=0.7` |
| CatBoost | both countries, with a country flag | `depth=6 · iterations=400 · lr=0.05 · l2=6` |

All three fit the rank-transformed target with median imputation. Predictions are z-scored and blended

```
0.8 × ( 0.7·ridge + 0.3·lightgbm ) + 0.2 × catboost     →   0.56 / 0.24 / 0.20
```

then France's predictions are shrunk halfway toward their own mean. Trailing windows use `shift(1)` so a day never sees itself; neighbour features draw only on training targets and a day is never its own neighbour.

## Setup

**Requirements:** Python 3.14 and [uv](https://docs.astral.sh/uv/). Everything else is pinned in `uv.lock`.

```bash
uv sync                      # creates .venv with all dependencies, including the notebook tooling
```

**Data.** The four challenge CSVs (`X_train`, `y_train`, `X_test_final`, `y_test_random_final`) are included in `data/raw/`. They come from the [challenge page](https://challengedata.ens.fr/challenges/97); nothing else is needed.

## Usage

**Generate the submission**

```bash
uv run python src/make_submission.py
```

**Reproduce the cross-validation score**

```bash
uv run python src/validate.py              # 5 seeds, about a minute
uv run python src/validate.py --seeds 16   # the 0.5716 quoted above, about 3 minutes
```

**Run the notebooks.** Register the environment as a Jupyter kernel once, then open them in VS Code or JupyterLab and pick that kernel:

```bash
uv run python -m ipykernel install --user --name qrt-elec --display-name "QRT (.venv)"
uv run --with jupyterlab jupyter lab       # or open the .ipynb files in VS Code
```

To execute a notebook headlessly and refresh its outputs:

```bash
uv run python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=qrt-elec notebooks/eda.ipynb
```

**Use the pieces from your own code**

```python
import sys; sys.path.insert(0, "src")
import qrt_prep as P, qrt_timeorder as TO, qrt_temporal as TT

train, test = P.build_frames()                                  # raw CSVs, sign-flip columns dropped
P.validate(train, test)                                         # re-asserts every structural fact
train, test, cols = TO.add_time_features(train, test, P.feature_columns(train))
folds = P.make_folds(train.DAY_ID, seed=0)                      # day-grouped, randomised
```

## Project structure

```
src/
  make_submission.py      the full pipeline, raw CSVs → submission
  validate.py             the cross-validation harness behind the score
  qrt_timeorder.py        the ID ordering; difference and cumulative-return features
  qrt_temporal.py         neighbouring-day target features, with a leakage guard
  qrt_prep.py             loading, validation, randomised grouped folds, scoring
notebooks/
  eda.ipynb               exploratory analysis: findings 2 and 3, and the validation harness
  data_prep.ipynb         preprocessing decisions, each one measured
submissions/
  blend3_final.csv        the submitted file
data/raw/                 the four challenge CSVs
docs/challenge.md         the challenge statement
```

The EDA notebook's §1.3 concludes no time structure exists — right about `DAY_ID`, wrong about the dataset. Finding 1 came later; the original reasoning is left in place with a note rather than rewritten.

<details>
<summary><b>What didn't work</b> — recorded so nothing gets retried</summary>
<br>

FR−DE spread features · seasonal phase from solar · calendar features from the recovered ordering · missing-value indicators · row deletion · feature standardisation · feature rank transforms · gaussianised targets · sample weighting · robust losses · ElasticNet · partial pooling across countries · second differences · lagged levels · rolling feature volatility · EWMA deviations · longer cumulative windows · every fitted stack, gate or weight optimiser.

A per-row oracle choosing between ridge and LightGBM would score 0.68, and about half of that is stable row-level structure — but any gate trained to learn it lands at chance accuracy. The two models' errors correlate at 0.61: knowing where one struggles says nothing about which to trust.

Time-series foundation models have nothing to find here either. With the ordering recovered, the target's own autocorrelation is two lags of mild mean reversion in Germany and nothing in France, which the neighbour-target features already capture.

</details>
