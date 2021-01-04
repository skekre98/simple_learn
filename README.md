<p align="center">
    <img src="/img/logo.png">
</p>

[![Python](https://img.shields.io/pypi/pyversions/simple-learn.svg?style=plastic)](https://badge.fury.io/py/simple-learn)
[![PyPI version shields.io](https://img.shields.io/pypi/v/simple-learn.svg?kill_cache=1)](https://pypi.python.org/pypi/simple-learn/)
[![PyPI license](https://img.shields.io/pypi/l/simple-learn.svg)](https://pypi.python.org/pypi/simple-learn/)

[SimpleLearn](https://pypi.org/project/simple-learn/) is a python package that aims to create an automatic process of model algorithm selection, hyper parameter tuning, iterative modelling, and model assessment. This package is built on top of [sklearn](https://scikit-learn.org/) and leaves all the flexibility and API support available to you. Keep in mind this package does NOT automate the entire process of data science and assumes you are handling tasks such as data preparation and feature engineering. A strong model algorithm cannot apologize for bad data.

## Install

To install the current release of SimpleLearn:
```
$ pip install simple-learn
```
To update SimpleLearn to the latest version, add `--upgrade` flag to the above command.

#### *Try your first SimpleLearn program*
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

## Build Locally

You can build your most recent changes by running the following command from the root directory:
```
$ pip install -e ."[devel]"
```

You can then import the package into your own code:
```python
# With recent changes
import simple_learn
```

## Contributing

There is a lot to do so contributions are really appreciated! This is a great project for early stage developers to work with.

To begin it is recommended starting with issues labelled [good first issue](https://github.com/skekre98/simple_learn/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).


How to get started:

1. Fork the simple_learn repo.
2. Create a new branch in you current repo from the 'main' branch with issue label.
3. 'Check out' the code with Git or [GitHub Desktop](https://desktop.github.com/)
4. Check [contributing.md](CONTRIBUTING.md)
5. Prior to pushing your changes, run the following command to format/lint your code(The CI pipeline will fail if standards are not met) followed by creating a Pull Request (PR) on simple_learn.
```bash
$ pre-commit run --all-files
```
