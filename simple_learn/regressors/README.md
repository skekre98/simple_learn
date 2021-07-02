# SimpleRegressor

This module contains the implementation of *SimpleRegressor* and *SimpleRegressorList*.

*SimpleRegressor*
```python
>>> from sklearn.datasets import load_diabetes
>>> from simple_learn.regressors import SimpleRegressor
>>>
>>> diabetes = load_diabetes()
>>> rgr = SimpleRegressor()
>>> rgr.fit(diabetes.data, diabetes.target)
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

*SimpleRegressorList*
```python
>>> from sklearn.datasets import load_diabetes
>>> from simple_learn.regressors import SimpleRegressorList
>>>
>>> diabetes = load_diabetes()
>>> rgr_list.fit(diabetes.data, diabetes.target)
>>> rgr_list
{
    "Type": "SGDRegressor",
    "Rank": 1,
    "Training Duration": "0.003373861312866211s",
    "GridSearch Duration": "15.93338394165039s",
    "Parameters": {
        "alpha": 0.0001,
        "eta0": 0.1,
        "learning_rate": "adaptive",
        "loss": "squared_epsilon_insensitive",
        "max_iter": 10000,
        "penalty": "l1"
    },
    "Metrics": {
        "Training Score": 54.777144265696144,
        "Mean Absolute Error": 43.52439301067071,
        "Mean Squared Error": 2884.9267631483817,
        "R-Squared": 0.5134936321189809
    },
    "Index": 0
}
{
    "Type": "KNeighborsRegressor",
    "Rank": 2,
    "Training Duration": "0.00043511390686035156s",
    "GridSearch Duration": "0.39653992652893066s",
    "Parameters": {
        "algorithm": "ball_tree",
        "n_neighbors": 14,
        "p": 2,
        "weights": "distance"
    },
    "Metrics": {
        "Training Score": 56.2496192614498,
        "Mean Absolute Error": 0.0,
        "Mean Squared Error": 0.0,
        "R-Squared": 1.0
    },
    "Index": 1
}
{
    "Type": "HistGradientBoostingRegressor",
    "Rank": 3,
    "Training Duration": "0.4561913013458252s",
    "GridSearch Duration": "2.6251039505004883s",
    "Parameters": {
        "learning_rate": 0.1,
        "loss": "least_absolute_deviation"
    },
    "Metrics": {
        "Training Score": 56.50792962111714,
        "Mean Absolute Error": 21.52874366299438,
        "Mean Squared Error": 1190.6544749582472,
        "R-Squared": 0.7992111996004159
    },
    "Index": 2
}
...
>>> rgr = rgr_list.pop(index=2) # default index is 0
>>> rgr
{
    "Type": "HistGradientBoostingRegressor",
    "Rank": 3,
    "Training Duration": "0.4561913013458252s",
    "GridSearch Duration": "2.6251039505004883s",
    "Parameters": {
        "learning_rate": 0.1,
        "loss": "least_absolute_deviation"
    },
    "Metrics": {
        "Training Score": 56.50792962111714,
        "Mean Absolute Error": 21.52874366299438,
        "Mean Squared Error": 1190.6544749582472,
        "R-Squared": 0.7992111996004159
    },
    "Index": 2
}
```
