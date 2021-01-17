# Copyright (c) 2020 Sharvil Kekre skekre98
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

# Hyper-parameter search space for grid search
model_param_map = {
    "BernoulliNB": {
        "alpha": 10.0 ** -np.arange(1, 7),
        "binarize": [*np.linspace(0.0, 1, 3)],
        "fit_prior": [True, False],
    },
    "ComplementNB": {
        "alpha": np.linspace(0.1, 1, 3),
        "fit_prior": [True, False],
        "norm": [True, False],
    },
    "DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": np.arange(3, 15),
    },
    "ExtraTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "splitter": ["random", "best"],
        "max_depth": np.arange(3, 15),
    },
    "GradientBoostingClassifier": {
        "loss": ["deviance"],
        "min_samples_split": np.linspace(0.1, 0.5, 3),
        "min_samples_leaf": np.linspace(0.1, 0.5, 3),
        "max_depth": np.arange(3, 8),
        "max_features": ["log2", "sqrt"],
        "criterion": ["friedman_mse", "mse"],
        "n_estimators": [10],
    },
    "HistGradientBoostingClassifier": {
        "loss": ["auto", "categorical_crossentropy"],
        "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        "max_depth": np.arange(3, 15),
    },
    "KNeighborsClassifier": {
        "n_neighbors": np.arange(3, 15),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "Perceptron": {
        "alpha": 10.0 ** -np.arange(1, 3),
        "tol": [1, 0.1, 0.01, 0.001, 0.0001],
    },
    "RandomForestClassifier": {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [4, 5, 6, 7, 8],
        "criterion": ["gini", "entropy"],
        "random_state": [42],
    },
    "RidgeClassifier": {
        "alpha": 10.0 ** -np.arange(1, 3),
        "tol": [1, 0.1, 0.01, 0.001, 0.0001],
    },
    "SGDClassifier": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]},
}
