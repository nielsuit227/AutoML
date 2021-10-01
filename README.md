# Amplo - AutoML (for Machine Data)
[![image](https://img.shields.io/pypi/v/amplo.svg)](https://pypi.python.org/pypi/amplo)
[![PyPI - License](https://img.shields.io/pypi/l/virtualenv?style=flat-square)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/badge/python-%3E%3D3.6%2C%3C4.0-blue)
![](https://tokei.rs/b1/github/nielsuit227/automl)

Welcome to the Automated Machine Learning package `Amplo`. Amplo's AutoML is designed specifically for machine data and 
works very well with tabular time series data (especially unbalanced classification!).

Though this is a standalone Python package, Amplo's AutoML is also available on Amplo's ML Developer Platform. 
With a graphical user interface and various data connectors, it is the ideal place for service engineers to get started 
on Predictive Maintenance development. 

Amplo's AutoML Pipeline contains the entire Machine Learning development cycle, including exploratory data analysis, 
data cleaning, feature extraction, feature selection, model selection, hyper parameter optimization, stacking, 
version control, production-ready models and documentation. 

# Downloading Amplo
The easiest way is to install our Python package through [PyPi](https://pypi.org/project/amplo/):
```commandline
pip install Amplo
```

# 2. Usage
Usage is very simple with Amplo's AutoML Pipeline. 
```python
from Amplo import Pipeline
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression


x, y = make_classification()
pipeline = Pipeline()
pipeline.fit(x, y)

x, y = make_regression()
pipeline = Pipeline()
pipeline.fit(x, y)
```

# 3. Amplo AutoML Features

## Interval Analyser
`from Amplo.AutoML import IntervalAnalyser`

Interval Analyser for Log file classification. When log files have to be classified, and there is not enough
data for time series methods (such as LSTMs, ROCKET or Weasel, Boss, etc), one needs to fall back to classical
machine learning models which work better with lower samples. This raises the problem of which samples to
classify. You shouldn't just simply classify on every sample and accumulate, that may greatly disrupt
classification performance. Therefore, we introduce this interval analyser. By using an approximate K-Nearest 
Neighbors algorithm, one can estimate the strength of correlation for every sample inside a log. Using this 
allows for better interval selection for classical machine learning models.

To use this interval analyser, make sure that your logs are located in a folder of their class, with one parent folder with all classes, e.g.:

```
+-- Parent Folder
|   +-- Class_1
|       +-- Log_1.*
|       +-- Log_2.*
|   +-- Class_2
|       +-- Log_3.*
```
## Exploratory Data Analysis
`from Amplo.AutoML import DataExplorer`

Automated Exploratory Data Analysis. Covers binary classification and regression.
It generates:
- Missing Values Plot
- Line Plots of all features
- Box plots of all features
- Co-linearity Plot
- SHAP Values
- Random Forest Feature Importance
- Predictive Power Score

Additional plots for Regression:
- Seasonality Plots
- Differentiated Variance Plot
- Auto Correlation Function Plot
- Partial Auto Correlation Function Plot
- Cross Correlation Function Plot
- Scatter Plots

## Data Processing
`from Amplo.AutoML import DataProcesser`

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

## Data Sampler
`from Amplo.AutoML import DataSampler`

This pipeline is designed to handle unbalanced classification problems. 
Aside weighted loss functions, under sampling the majority class or down sampling the 
minority class helps. Various algorithms are analysed:
- SMOTE
- Borderline SMOTE
- Random Over Sampler
- Tomek Links
- One Sided Selection
- Random Under Sampler
- Edited Nearest Neighbours
- SMOTE Tomek
- SMOTE Edited Nearest Neighbours

## Feature Processing
`from Amplo.AutoML import FeatureProcesser`

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

## Sequencing
`from Amplo.AutoML import Sequencer`

For time series regression problems, it is often useful to include multiple previous samples instead of just the latest. 
This class sequences the data, based on which time steps you want included in the in- and output. 
This is also very useful when working with tensors, as a tensor can be returned which directly fits into a Recurrent Neural Network. 

## Modelling
`from Amplo.AutoML import Modeller`

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

## Grid Search
`from Amplo.GridSearch import *`

Contains three hyper parameter optimizers with extended predefined model parameters:
- Grid Search
- Halving Random Search
- `Optuna`'s Tree-Parzen-Estimator

## Automatic Documntation
`from Amplo.AutoML import Documenter`

Contains a documenter for classification (`binary` and `multiclass` problems), as well as for regression. 
Creates a pdf report for a Pipeline, including metrics, data processing steps, and everything else to recreate the result.


