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

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    jaccard_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import all_estimators

from simple_learn.classifiers.param_grid import model_param_map


class SimpleClassifier:
    def __init__(self):
        self.name = "Empty Model"
        self.sk_model = None
        self.training_accuracy = 0.0
        self.metrics = dict()

    def __str__(self):
        attr = {
            "Type": self.name,
            "Training Accuracy": self.training_accuracy,
            "Metrics": self.metrics,
        }

        return json.dumps(attr, indent=4)

    def fit(self, train_x, train_y, folds=3):
        estimators = all_estimators(type_filter="classifier")
        max_accuracy = 0.0
        best_model = None
        best_name = "Empty Model"
        for name, ClassifierClass in estimators:
            if name in model_param_map:
                param_grid = model_param_map[name]
                grid_clf = GridSearchCV(
                    ClassifierClass(),
                    param_grid,
                    cv=folds,
                    scoring="accuracy",
                    n_jobs=-1,
                )
                grid_clf.fit(train_x, train_y)
                if grid_clf.best_score_ > max_accuracy:
                    max_accuracy = grid_clf.best_score_
                    best_model = grid_clf.best_estimator_
                    best_name = name
                    pred_y = grid_clf.predict(train_x)
                    self.metrics["Jaccard Score"] = jaccard_score(
                        train_y, pred_y, average="macro"
                    )
                    self.metrics["ROC Score"] = roc_auc_score(train_y, pred_y)
                    self.metrics["F1 Score"] = f1_score(
                        train_y, pred_y, average="macro"
                    )
                    self.metrics["Precision Score"] = average_precision_score(
                        train_y, pred_y
                    )

        self.name = name
        self.sk_model = best_model
        self.training_accuracy = max_accuracy

    def predict(self, pred_x):
        return self.sk_model.predict(pred_x)
