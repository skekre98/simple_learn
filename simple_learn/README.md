# SimpleLearn

A python package to simplify or automate data science workflows.

## Installation

```
pip install simple-learn
```

## Primer

This package is based off [Google AutoML](https://cloud.google.com/automl) and hopes to allow people with limited machine learning knowledge to train high performance models for their specific use cases. Similar to AutoML,
SimpleLearn aims to create an automatic process of model algorithm selection, hyper parameter tuning, iterative modelling, and model assessment. Under the hood, SimpleLearn is using a greedy algorithm to select/assess the model algorithm while leveraging a grid search to tune model hyperparameters. The metrics for assessing the model can all be configured via input parameters. Keep in mind this package does NOT automate the entire process of data science and assumes you are handling tasks such as data preparation and feature engineering. A strong model algorithm cannot apologize for bad data.

## Usage

The following are examples of how to use some of the classes in SimpleLearn.

### SimpleClassifier

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
