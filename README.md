# Amplo - AutoML (for Machine Data)

[![image](https://img.shields.io/pypi/v/amplo.svg)](https://pypi.python.org/pypi/amplo)
[![PyPI - License](https://img.shields.io/pypi/l/virtualenv?style=flat-square)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/badge/python-%3E%3D3.9%2C%3C4.0-blue)
![](https://tokei.rs/b1/github/nielsuit227/automl)
![](https://img.shields.io/pypi/dm/amplo)

Welcome to the Automated Machine Learning package `amplo`. Amplo's AutoML is designed specifically for machine data and
works very well with tabular time series data (especially unbalanced classification!).

Though this is a standalone Python package, Amplo's AutoML is also available on Amplo's Smart Maintenance Platform.
With a graphical user interface and various data connectors, it is the ideal place for service engineers to get started
on Predictive.

Amplo's AutoML Pipeline contains the entire Machine Learning development cycle, including exploratory data analysis,
data cleaning, feature extraction, feature selection, model selection, hyperparameter optimization, stacking,
version control, production-ready models and documentation. It comes with additional tools such as interval analysers,
drift detectors, data quality checks, etc.

## 1. Downloading Amplo

The easiest way is to install our Python package through [PyPi](https://pypi.org/project/amplo/):

```bash
pip install amplo
```

## 2. Usage

Usage is very simple with Amplo's AutoML Pipeline.

```python
from amplo import Pipeline
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression

x, y = make_classification()
pipeline = Pipeline()
pipeline.fit(x, y)
yp = pipeline.predict_proba(x)

x, y = make_regression()
pipeline = Pipeline()
pipeline.fit(x, y)
yp = pipeline.predict(x)
```

## 3. Amplo AutoML Features

### Interval Analyser

```python
from amplo.automl import IntervalAnalyser
```

Interval Analyser for Log file classification. When log files have to be classified, and there is not enough
data for time series methods (such as LSTMs, ROCKET or Weasel, Boss, etc.), one needs to fall back to classical
machine learning models which work better with lower samples. This raises the problem of which samples to
classify. You shouldn't just simply classify on every sample and accumulate, that may greatly disrupt
classification performance. Therefore, we introduce this interval analyser. By using an approximate K-Nearest
Neighbors algorithm, one can estimate the strength of correlation for every sample inside a log. Using this
allows for better interval selection for classical machine learning models.

To use this interval analyser, make sure that your logs are located in a folder of their class, with one parent folder with all classes, e.g.:

```text
+-- Parent Folder
|   +-- Class_1
|       +-- Log_1.*
|       +-- Log_2.*
|   +-- Class_2
|       +-- Log_3.*
```

### Data Processing

```python
from amplo.automl import DataProcessor
```

Automated Data Cleaning:

- Infers & converts data types (integer, floats, categorical, datetime)
- Reformats column names
- Removes duplicates columns and rows
- Handles missing values by:
  - Removing columns
  - Removing rows
  - Interpolating
  - Filling with zero's
- Removes outliers using:
  - Clipping
  - Z-score
  - Quantiles
- Removes constant columns

### Feature Processing

```python
from amplo.automl import FeatureProcessor
```

Automatically extracts and selects features. Removes Co-Linear Features.
Included Feature Extraction algorithms:

- Multiplicative Features
- Dividing Features
- Additive Features
- Subtractive Features
- Trigonometric Features
- K-Means Features
- Lagged Features
- Differencing Features
- Inverse Features
- Datetime Features

Included Feature Selection algorithms:

- Random Forest Feature Importance (Threshold and Increment)
- Predictive Power Score

### Sequencing

```python
from amplo.automl import Sequencer
```

For time series regression problems, it is often useful to include multiple previous samples instead of just the latest.
This class sequences the data, based on which time steps you want included in the in- and output.
This is also very useful when working with tensors, as a tensor can be returned which directly fits into a Recurrent Neural Network.

### Modelling

```python
from amplo.automl import Modeller
```

Runs various regression or classification models.
Includes:

- Scikit's Linear Model
- Scikit's Random Forest
- Scikit's Bagging
- Scikit's GradientBoosting
- Scikit's HistGradientBoosting
- DMLC's XGBoost
- Catboost's Catboost
- Microsoft's LightGBM
- Stacking Models

### Grid Search

```python
from amplo.grid_search import OptunaGridSearch
```

Contains three hyperparameter optimizers with extended predefined model parameters:

- Optuna's Tree-Parzen-Estimator
