# SimpleClassifier

This module contains the implementation of *SimpleClassifier* and *SimpleClassifierList*.

### Usage

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
