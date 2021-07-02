# SimpleRegressor

This module contains the implementation of *SimpleClassifier* and *SimpleClassifierList*.

*SimpleClassifier*
```python
>>> from sklearn.datasets import load_diabetes
>>> from simple_learn.regressors import SimpleRegressor
>>> 
>>> diabetes = load_diabetes()
>>> rgr = SimpleRegressor()
>>> rgr.fit(diabetes.data, diabetes.target)
Fitting Models: 100%|█████████████████████████████████████████| 7/7 [04:07<00:00, 35.37s/ Algorithm]
>>> rgr
{
    "Type": "SGDRegressor",
    "Training Duration": "0.004389762878417969s",
    "GridSearch Duration": "14.880131006240845s",
    "Parameters": {
        "alpha": 0.0001,
        "eta0": 0.05,
        "learning_rate": "adaptive",
        "loss": "squared_epsilon_insensitive",
        "max_iter": 10000,
        "penalty": "elasticnet"
    },
    "Metrics": {
        "Training Score": 54.81611954590818,
        "Mean Absolute Error": 43.75161649319255,
        "Mean Square Error": 2896.700648762865,
        "R-Squared": 0.5115081153983077
    }
}
```