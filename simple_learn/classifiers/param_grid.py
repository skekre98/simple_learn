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
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    ExpSineSquared,
    RationalQuadratic,
)

ker_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
    1.0, length_scale_bounds="fixed"
)
ker_rq = ConstantKernel(1.0, constant_value_bounds="fixed") * RationalQuadratic(
    alpha=0.1, length_scale=1
)
ker_expsine = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(
    1.0, 5.0, periodicity_bounds=(1e-2, 1e1)
)
kernel_list = [ker_rbf, ker_rq, ker_expsine]

model_param_map = {
    "BernoulliNB": {
        "alpha": 10.0 ** -np.arange(1, 7),
        "binarize": [*np.linspace(0.0, 1, 3)],
        "fit_prior": [True, False],
    },
    "CategoricalNB": {"alpha": np.linspace(0.1, 1, 3), "fit_prior": [True, False]},
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
    "GaussianProcessClassifier": {
        "kernel": kernel_list,
        "optimizer": ["fmin_l_bfgs_b"],
        "n_restarts_optimizer": [1, 2, 3],
    },
    "GradientBoostingClassifier": {
        "loss": ["deviance"],
        "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        "min_samples_split": np.linspace(0.1, 0.5, 12),
        "min_samples_leaf": np.linspace(0.1, 0.5, 12),
        "max_depth": np.arange(3, 15),
        "max_features": ["log2", "sqrt"],
        "criterion": ["friedman_mse", "mae"],
        "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
        "n_estimators": [10],
    },
    "HistGradientBoostingClassifier": {
        "loss": ["auto", "binary_crossentropy", "categorical_crossentropy"],
        "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        "max_depth": np.arange(3, 15),
    },
    "KNeighborsClassifier": {
        "n_neighbors": np.arange(3, 15),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "LinearSVC": {
        "penalty": ["l1", "l2"],
        "loss": ["hinge", "squared_hinge"],
        "dual": [True, False],
        "tol": [1, 0.1, 0.01, 0.001, 0.0001],
        "C": [0.1, 1, 10, 100],
        "multi_class": ["ovr", "crammer_singer"],
    },
    "LogisticRegression": {
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    },
    "MLPClassifier": {
        "solver": ["lbfgs"],
        "max_iter": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
        "alpha": 10.0 ** -np.arange(1, 7),
        "hidden_layer_sizes": np.arange(10, 15),
        "random_state": [0, 1, 2, 3, 4],
    },
    "PassiveAggressiveClassifier": {
        "C": [0.1, 1, 10, 100],
        "max_iter": [1, 10, 100],
        "tol": [1, 0.1, 0.01, 0.001, 0.0001],
    },
    "Perceptron": {
        "penalty": ["l1", "l2", "elasticnet"],
        "alpha": 10.0 ** -np.arange(1, 7),
        "max_iter": [1, 10, 100],
        "tol": [1, 0.1, 0.01, 0.001, 0.0001],
    },
    "RadiusNeighborsClassifier": {
        "weights": ["uniform", "distance"],
        "algorithm": ["auto"],
        "leaf_size": np.arange(20, 40),
    },
    "RandomForestClassifier": {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [4, 5, 6, 7, 8],
        "criterion": ["gini", "entropy"],
        "random_state": [42],
    },
    "RidgeClassifier": {
        "kernel": ["rbf", "linear"],
        "gamma": [1e-3, 1e-4],
        "C": [1, 10, 100, 1000],
    },
    "SGDClassifier": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10], "max_iter": [1000]},
    "SVC": {
        "C": [1, 10, 100, 1000],
        "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
    },
}
