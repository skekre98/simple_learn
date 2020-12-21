# SimpleLearn

A python package to simplify or automate data science workflows.

## Installation

```
pip install simple-learn
```

## Primer

This package is based off [Google AutoML](https://cloud.google.com/automl) and hopes to allow people with limited machine learning knowledge to train high performance models for their specific use cases. Similar to AutoML,
SimpleLearn aims to create an automatic process of model algorithm selection, hyper parameter tuning, iterative modelling, and model assessment. Under the hood, SimpleLearn is using a greedy algorithm to select/assess the model algorithm while leveraging a grid search to tune model hyperparameters. The metrics for assessing the model can all be configured via input parameters. Keep in mind this package does NOT automate the entire process of data science and assumes you are handling tasks such as data preparation and feature engineering. A strong model algorithm cannot apologize for bad data.

### Usage

The following are examples of how to use some of the classes in SimpleLearn.

#### SimpleClassifier

```python
>>> from sklearn.datasets import load_iris
>>> from simple_learn.classifiers import SimpleClassifier
>>>
>>> iris = load_iris()
>>> clf = SimpleClassifier()
>>> clf.fit(iris.data, iris.target)
>>> clf
{
    "Type": "KNeighborsClassifier",
    "Training Duration": "0.0006814002990722656s",
    "GridSearch Duration": "0.17136621475219727s",
    "Parameters": {
        "metric": "euclidean",
        "n_neighbors": 4,
        "weights": "uniform"
    },
    "Metrics": {
        "Training Accuracy": 0.9866666666666667,
        "Jaccard Score": 0.9245283018867925,
        "F1 Score": 0.96
    }
}
```

#### SimpleClassifierList
```python
>>> from sklearn.datasets import load_iris
>>> from simple_learn.classifiers import SimpleClassifierList
>>>
>>> iris = load_iris()
>>> clf_list = SimpleClassifierList()
>>> clf_list.fit(iris.data, iris.target)
>>> clf_list
{
    "Type": "KNeighborsClassifier",
    "Rank": 1,
    "Training Duration": "0.0005269050598144531s",
    "GridSearch Duration": "0.17510604858398438s",
    "Parameters": {
        "metric": "euclidean",
        "n_neighbors": 4,
        "weights": "uniform"
    },
    "Metrics": {
        "Training Accuracy": 0.9866666666666667,
        "Jaccard Score": 0.9245283018867925,
        "F1 Score": 0.96
    },
    "Index": 0
}
{
    "Type": "DecisionTreeClassifier",
    "Rank": 2,
    "Training Duration": "0.0004031658172607422s",
    "GridSearch Duration": "0.06979990005493164s",
    "Parameters": {
        "criterion": "gini",
        "max_depth": 3
    },
    "Metrics": {
        "Training Accuracy": 0.9733333333333333,
        "Jaccard Score": 0.9486989764459243,
        "F1 Score": 0.9733226623982927
    },
    "Index": 1
}
{
    "Type": "ExtraTreeClassifier",
    "Rank": 3,
    "Training Duration": "0.00039696693420410156s",
    "GridSearch Duration": "0.11928296089172363s",
    "Parameters": {
        "criterion": "gini",
        "max_depth": 4,
        "splitter": "best"
    },
    "Metrics": {
        "Training Accuracy": 0.9666666666666667,
        "Jaccard Score": 0.9611613876319759,
        "F1 Score": 0.97999799979998
    },
    "Index": 2
}
...
>>> clf = clf_list.pop(index=2) \\ default index is 0
>>> clf
{
    "Type": "ExtraTreeClassifier",
    "Training Duration": "0.00039696693420410156s",
    "GridSearch Duration": "0.11928296089172363s",
    "Parameters": {
        "criterion": "gini",
        "max_depth": 4,
        "splitter": "best"
    },
    "Metrics": {
        "Training Accuracy": 0.9666666666666667,
        "Jaccard Score": 0.9611613876319759,
        "F1 Score": 0.97999799979998
    }
}
```
