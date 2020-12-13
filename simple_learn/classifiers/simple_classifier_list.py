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
from sklearn.utils import all_estimators

from simple_learn.classifiers import SimpleClassifier
from simple_learn.classifiers.param_grid import model_param_map


class SimpleClassifierListObject:
    def __init__(self, clf, rank):
        self.clf = clf
        self.rank = rank

    def __str__(self):

        for k in self.clf.attributes:
            if type(self.clf.attributes[k]) == np.int64:
                self.clf.attributes[k] = int(self.clf.attributes[k])

        attr = {
            "Type": self.clf.name,
            "Rank": self.rank,
            "Training Duration": "{}s".format(self.clf.train_duration),
            "GridSearch Duration": "{}s".format(self.clf.gridsearch_duration),
            "Parameters": self.clf.attributes,
            "Metrics": self.clf.metrics,
        }

        return json.dumps(attr, indent=4)


class SimpleClassifierList:
    def __init__(self, scoring="auto"):
        self.ranked_list = []
        metric_map = {
            "auto": "Training Accuracy",
            "jaccard": "Jaccard Score",
            "f1": "F1 Score",
        }
        self.metric = metric_map[scoring]

    def __str__(self):
        r = 1
        res = []
        for clf in self.ranked_list:
            obj = SimpleClassifierListObject(clf, r)
            res.append(str(obj))
            r += 1
        return "\n".join(res)

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
                clf = SimpleClassifier()
                clf.metrics["Training Accuracy"] = grid_clf.best_score_
                pred_y = grid_clf.predict(train_x)
                clf.metrics["Jaccard Score"] = jaccard_score(
                    train_y, pred_y, average="macro"
                )
                clf.metrics["F1 Score"] = f1_score(train_y, pred_y, average="macro")
                clf.sk_model = grid_clf.best_estimator_
                clf.name = name
                clf.attributes = grid_clf.best_params_
                clf.train_duration = grid_clf.refit_time_
                clf.gridsearch_duration = end - start
                self.ranked_list.append(clf)
        metrik = lambda clf: clf.metrics[self.metric]
        self.ranked_list.sort(reverse=True, key=metrik)
