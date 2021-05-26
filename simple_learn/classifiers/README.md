# SimpleClassifier

This module contains the implementation of *SimpleClassifier* and *SimpleClassifierList*.

*SimpleClassifier*
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


*SimpleClassifierList*
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
>>> clf = clf_list.pop(index=2) # default index is 0
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
