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

import json
import time

import numpy as np
from sklearn.metrics import f1_score, jaccard_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import all_estimators

from simple_learn.classifiers.param_grid import model_param_map


class SimpleClassifier:
    def __init__(self):
        self.name = "Empty Model"
        self.sk_model = None
        self.attributes = dict()
        self.metrics = dict()
        self.gridsearch_duration = None
        self.train_duration = None

    def __str__(self):

        for k in self.attributes:
            if type(self.attributes[k]) == np.int64:
                self.attributes[k] = int(self.attributes[k])

        attr = {
            "Type": self.name,
            "Training Duration": "{}s".format(self.train_duration),
            "GridSearch Duration": "{}s".format(self.gridsearch_duration),
            "Parameters": self.attributes,
            "Metrics": self.metrics,
        }

        return json.dumps(attr, indent=4)

    def fit(self, train_x, train_y, folds=3):
        estimators = all_estimators(type_filter="classifier")
        for name, ClassifierClass in estimators:
            if name in model_param_map:
                param_grid = model_param_map[name]
                grid_clf = GridSearchCV(
                    ClassifierClass(),
                    param_grid,
                    cv=folds,
                    scoring="accuracy",
                    verbose=0,
                    n_jobs=-1,
                )
                start = time.time()
                grid_clf.fit(train_x, train_y)
                end = time.time()
                if grid_clf.best_score_ > self.metrics.get("Training Accuracy", 0.0):
                    self.metrics["Training Accuracy"] = grid_clf.best_score_
                    pred_y = grid_clf.predict(train_x)
                    self.metrics["Jaccard Score"] = jaccard_score(
                        train_y, pred_y, average="macro"
                    )
                    self.metrics["F1 Score"] = f1_score(
                        train_y, pred_y, average="macro"
                    )
                    self.sk_model = grid_clf.best_estimator_
                    self.name = name
                    self.attributes = grid_clf.best_params_
                    self.train_duration = grid_clf.refit_time_
                    self.gridsearch_duration = end - start

    def predict(self, pred_x):
        return self.sk_model.predict(pred_x)
