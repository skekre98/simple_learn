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
import logging
import time

import numpy as np
from sklearn.metrics import f1_score, jaccard_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import all_estimators

from simple_learn.classifiers import SimpleClassifier
from simple_learn.classifiers.param_grid import model_param_map
from simple_learn.encoders import simple_model_encoder


class SimpleClassifierListObject:
    """
    A class used to keep track SimpleClassifiers and
    corresponding rank for terminal display

    ...

    Attributes
    ----------
    clf : simple_learn.classifiers.SimpleClassifier
        the SimpleClassifier object
    rank : int
        the associated rank for classifier
    """

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
            "Index": self.rank - 1,
        }

        str_out = json.dumps(attr, cls=simple_model_encoder.npEncoder, indent=4)
        return str_out

    def __repr__(self):

        attr = {
            "Type": self.clf.name,
            "Rank": self.rank,
            "Training Duration": "{}s".format(self.clf.train_duration),
            "GridSearch Duration": "{}s".format(self.clf.gridsearch_duration),
            "Parameters": self.clf.attributes,
            "Metrics": self.clf.metrics,
            "Index": self.rank - 1,
        }

        repr_out = json.dumps(attr, cls=simple_model_encoder.npEncoder, indent=4)
        return repr_out


class SimpleClassifierList:
    """
    A class used to maintain ranked list of
    SimpleClassifiers

    ...

    Attributes
    ----------
    ranked_list : list
        the ranked list of SimpleClassifiers
    metric : str {auto, jaccard, f1}
        the scoring metric for ranking models
    logger : logging.Logger
        logger for notifying user of warnings

    Methods
    -------
    fit(train_x, train_y, folds=3)
        Fits a given dataset onto SimpleClassifier and
        creates a ranked list based on scores
    pop(index=0)
        Removes a SimpleClassifier at a specific index
        for usage
    """

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
        return "\n".join(res) if len(res) > 1 else "The List is Empty!"

    def __repr__(self):
        r = 1
        res = []
        for clf in self.ranked_list:
            obj = SimpleClassifierListObject(clf, r)
            res.append(str(obj))
            r += 1
        return "\n".join(res) if len(res) > 1 else "The List is Empty!"

    def fit(self, train_x, train_y, folds=3):
        """Trains all classification models from
        parameter grid by running model algorithm search.

        Creates a ranked list of models based on selected
        scoring metric.

        Parameters
        ----------
        train_x : numpy.ndarray
            The features for training classification model
        train_y : numpy.ndarray
            The corresponding label for feature array
        folds : int, optional
            The number of folds for cross validation
        """

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
                try:
                    grid_clf.fit(train_x, train_y)
                except BaseException as error:
                    self.logger.warning(f"{name} failed due to, Error : {error}.")
                    continue
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

    def pop(self, index=0):
        """Removes SimpleClassifier from a specific
        index in ranked list.

        Parameters
        ----------
        index : int
            The index corresponding to the SimpleClassifier
            being removed from ranked list
        """

        return self.ranked_list.pop(index)
