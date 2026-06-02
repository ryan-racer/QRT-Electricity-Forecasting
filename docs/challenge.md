# QRT ENS Data Challenge 2023: Electricity Price Explanation

Can you explain the price of electricity?

Provider: Qube Research & Technologies (QRT)

## Challenge Context

Every day, many factors impact the price of electricity. Local weather variations affect both electricity generation and demand. Long-term phenomena such as global warming can also have a significant influence. Geopolitical events, such as the war in Ukraine, may affect commodity prices, which are key inputs in electricity generation.

Each country also relies on a particular energy mix, including nuclear, solar, hydro, gas, coal, and other sources. In Europe, countries may import or export electricity with their neighbors through dynamic markets. These factors make electricity price modeling complex.

## Challenge Goal

The aim is to model electricity price changes using weather, energy, commodity, and commercial data for France and Germany. The problem is designed to explain electricity price changes with simultaneous variables, so it is not a forecasting problem.

The target is the daily price variation of 24-hour electricity baseload futures contracts in France and Germany. These contracts allow a buyer or seller to receive or deliver electricity at a specified price and future maturity. Short-term electricity futures are financial instruments that reflect the expected future price of electricity under current market conditions.

## Provider

Qube Research & Technologies (QRT) is a global quantitative and systematic investment manager operating across asset classes. QRT was established in 2018 and supports coding initiatives and academic projects that promote mathematics and science education.

## Contact

Questions about the challenge can be sent to:

`qrtdatachallenge@qube-rt.com`

## Winners

1. Mohamed Ali Ben Amara
2. mb
3. Amine Abdelkader

## Evaluation Metric

Submissions are scored with Spearman's correlation between the participant output and the actual daily price changes over the test sample.

## Data Description

The challenge provides three main CSV datasets:

- `X_train`: training input data
- `y_train`: training target data
- `X_test`: test input data

The local data files are stored in [`../data/raw`](../data/raw):

- `X_train.csv`
- `y_train.csv`
- `X_test_final.csv`
- `y_test_random_final.csv`

`X_train` and `X_test` contain the same explanatory variables over different time periods. The `ID` columns in `X_train` and `y_train` match. The same identifier logic applies to the test data.

The training data contains 1,494 rows. The test data contains 654 rows.

## Input Columns

The input datasets contain 35 columns.

### Identifiers

- `ID`: unique row identifier associated with a day and country
- `DAY_ID`: anonymized day identifier
- `COUNTRY`: country identifier, where `DE` is Germany and `FR` is France

### Commodity Price Variations

- `GAS_RET`: European gas
- `COAL_RET`: European coal
- `CARBON_RET`: carbon emissions futures

### Weather Measures

For each country `x`:

- `x_TEMP`: temperature
- `x_RAIN`: rainfall
- `x_WIND`: wind

### Energy Production Measures

For each country `x`:

- `x_GAS`: natural gas
- `x_COAL`: hard coal
- `x_HYDRO`: hydro reservoir
- `x_NUCLEAR`: daily nuclear production
- `x_SOLAR`: photovoltaic
- `x_WINDPOW`: wind power
- `x_LIGNITE`: lignite

### Electricity Use Metrics

For each country `x`:

- `x_CONSUMPTION`: total electricity consumption
- `x_RESIDUAL_LOAD`: electricity consumption after renewable energy production
- `x_NET_IMPORT`: imported electricity from Europe
- `x_NET_EXPORT`: exported electricity to Europe

Cross-country exchange columns:

- `DE_FR_EXCHANGE`: total daily electricity exchange between Germany and France
- `FR_DE_EXCHANGE`: total daily electricity exchange between France and Germany

## Output Columns

The output datasets contain two columns:

- `ID`: unique row identifier corresponding to the input identifiers
- `TARGET`: daily price variation for 24-hour electricity baseload futures

Submission files must contain exactly these columns, with `ID` values corresponding to the `ID` column of `X_test`.

## Benchmark

The benchmark model is a simple linear regression with light data cleaning:

- Missing values are filled with `0`.
- The `COUNTRY` column is dropped.
- A single model is trained for both France and Germany.

The public score obtained by this benchmark was 15.86%.

The local benchmark notebook is available at [`../notebooks/benchmark_qrt_en.ipynb`](../notebooks/benchmark_qrt_en.ipynb).
