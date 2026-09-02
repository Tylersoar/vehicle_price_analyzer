[README.md](https://github.com/user-attachments/files/31740244/README.md)
# Used Car Price Analysis

A supervised regression pipeline over ~90,000 UK used car listings, built to answer a specific question: **how does a car's value actually decay with age?**

The short answer is that depreciation is approximately exponential, not linear — roughly 36% of value is lost in the first three years and 63% by year eight. That finding is what motivated the log-price transform used in the models below.

## Dataset

Nine manufacturer CSVs from the [100,000 UK Used Car Dataset](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) on Kaggle: Audi, BMW, Ford, Hyundai, Mercedes, Skoda, Toyota, Vauxhall and VW.

Each row is a listing with year, mileage, engine size, transmission, fuel type, road tax, mpg and price. The files are loaded separately and tagged with a `brand` column, because model names are not unique across manufacturers

## Cleaning

Decisions worth calling out:

- **Duplicates are dropped before the train/test split.** The same listing appearing in both sets is a silent form of leakage that inflates test scores without any visible error.
- **`engineSize == 0` is treated as missing**, not as a real value — no car has a zero-litre engine.
- **Rows with a model year in the future are removed** (the raw data contains at least one 2060 listing).
- `transmission == 'Other'` is dropped as uninformative, and the `tax(£)` column header is normalised, since only some manufacturer files use that form.

`Car Age` is then derived as the reference year minus the model year, and the raw year column is dropped — age is the meaningful predictor, and using year directly would tie the model to the moment the data was collected.

## Method

Categorical features (brand, model, fuel type, transmission) are one-hot encoded. The target is log-transformed, and the models are trained to predict `log(price)`.

Two models are compared on an 80/20 split:

- Linear regression, as a baseline
- Random Forest regressor (100 trees)

### Retransformation bias

Training on `log(price)` and exponentiating the predictions returns the geometric mean rather than the arithmetic mean, which systematically under-predicts. Duan's smearing estimator corrects for this.

The residuals used to compute the smearing factor are taken from **out-of-bag predictions**, not in-sample ones. This matters: a Random Forest grown to full depth nearly memorises its training data, so in-sample residuals are close to zero and the correction would come out at ~1.000 regardless of whether bias exists. Estimated out-of-bag, the factor is **1.005** — the bias is genuinely negligible here, which is a measured result rather than an assumption.

## Results

| | Linear regression | Random Forest |
|---|---|---|
| Test R² | 0.879 | **0.957** |
| MAE | £1,767 | **£1,144** |
| RMSE | £3,401 | **£2,080** |

Training R² for the Random Forest is 0.993 against a test R² of 0.957. That gap is variance — fully grown trees fitting training noise — and is expected behaviour for a bagged ensemble rather than a defect.

Note that R² is itself a comparison against predicting the mean price for every car, so no separate baseline model is needed: 0.957 means 95.7% of the variance a mean-prediction baseline leaves unexplained is accounted for.

## Depreciation findings

Partial dependence of predicted price on `Car Age`, with other features held at their average:

| Age | Predicted price | Cumulative loss |
|---|---|---|
| 0 | £21,249 | — |
| 3 | £13,570 | 36% |
| 5 | £10,984 | 48% |
| 8 | £7,829 | 63% |
| 10 | £5,091 | 76% |
| 15 | £2,845 | 87% |

The year-on-year decline is roughly constant in *percentage* terms — around 13% a year over the first decade — rather than constant in pounds. A geometric decay at 13.3% tracks the observed curve closely across that range.

This is why the log transform helps: taking logs converts a constant proportional decline into a constant additive one, which is a shape a linear model can fit. The data motivated the transform, and the transform improved the models.

It also explains the error profile. Constant percentage error means larger absolute error on expensive cars, which is why RMSE sits at roughly 1.8× MAE rather than the ~1.25× you would expect from symmetric, well-behaved errors.

## Limitations

**The curve is only meaningful up to about age 15.** Beyond age 24 there are very few training rows, and the partial dependence values there — which tick upward, suggesting appreciation — are an artefact of sparse data, not evidence of classic-car value recovery. No claim is made about that range.

**Partial dependence assumes feature independence, which does not hold.** Older cars have more miles, so the curve reflects the combined effect of ageing and accumulated mileage rather than age in isolation.

**The model under-predicts high-value cars.** There are only around 90 listings above £80,000, so those rows are scattered across leaves dominated by cheaper cars and the leaf averages pull predictions downward. Tree ensembles also cannot predict outside the range of their training targets. The under-prediction at the top end is therefore structural, not incidental.

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib
```

Place the manufacturer CSVs in `datasets/` and run:

```bash
python main.py
```

## Possible extensions

- Gradient boosting as a third model, and feature importance comparison
- Per-price-band percentage error, to test whether the model is genuinely worse on expensive cars or whether RMSE is simply reflecting scale
- A 2D partial dependence over age and mileage, to separate the two effects
- Encoding and scaling moved inside a `Pipeline` fitted on training data only, so unseen categories are handled the way they would be in deployment
