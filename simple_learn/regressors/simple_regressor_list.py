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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import all_estimators

from simple_learn.encoders import simple_model_encoder
from simple_learn.regressors import SimpleRegressor
from simple_learn.regressors.param_grid import model_param_map


class SimpleRegressorListObject:
    """
    A class used to keep track of SimpleRegressors and corresponding rank for
    terminal display

    '''

    Attributes
    ----------
    rgr : simple_learn.regressors.SimpleRegressor
        the SimpleRegressor object
    rank : int
        The associated rank for regressor
    """

    def __init__(self, rgr, rank):
        self.rgr = rgr
        self.rank = rank

    def __str__(self):

        for k in self.rgr.attributes:
            if type(self.rgr.attributes[k]) == np.int64:
                self.rgr.attributes[k] = int(self.rgr.attributes[k])

        attr = {
            "Type": self.rgr.name,
            "Rank": self.rank,
            "Training Duration": "{}s".format(self.rgr.train_duration),
            "GridSearch Duration": "{}s".format(self.rgr.gridsearch_duration),
            "Parameters": self.rgr.attributes,
            "Metrics": self.rgr.metrics,
            "Index": self.rank - 1,
        }

        str_out = json.dumps(attr, cls=simple_model_encoder.npEncoder, indent=4)
        return str_out

    def __repr__(self):
        attr = {
            "Type": self.rgr.name,
            "Rank": self.rank,
            "Training Duration": "{}s".format(self.rgr.train_duration),
            "GridSearch Duration": "{}s".format(self.rgr.gridsearch_duration),
            "Parameters": self.rgr.attributes,
            "Metrics": self.rgr.metrics,
            "Index": self.rank - 1,
        }

        repr_out = json.dumps(attr, cls=simple_model_encoder.npEncoder, indent=4)
        return repr_out


class SimpleRegressorList:
    """
    A class used to maintain ranked list of SimpleRegressors

    '''

    Attributes
    ----------
    ranked_list : list
        the ranked list of SimpleRegressors
    metric : str {auto, mae, mse, r2}
        the scoring metric for ranking models
    logger : logging.Logger
        logger for notifying user of warnings

    Methods
    -------
    fit(train_x, train_y, folds=3)
        Fits a given dataset onto SimpleRegressor and creates a ranked list based
        on scores
    pop(index = 0)
        Removes a SimpleRegressor at a specific index for usage
    """

    def __init__(self, scoring="auto"):
        self.ranked_list = []
        metric_map = {
            "auto": "Training Score",
            "mae": "Mean Absolute Error",
            "mse": "Mean Squared Error",
            "r2": "R-Squared",
        }
        self.metric = metric_map[scoring]

    def __str__(self):
        r = 1
        res = []
        for rgr in self.ranked_list:
            obj = SimpleRegressorListObject(rgr, r)
            res.append(str(obj))
            r += 1
        return "\n".join(res) if len(res) > 1 else "The List is Empty!"

    def fit(self, train_x, train_y, folds=3):
        """
        Trains all regressors from parameter grid by running model algorithm search.

        Creates a ranked list of models based on selected scoring metric.

        Parameters
        ----------
        train_x : numpy.ndarray
            The features for training regression model
        train_y : numpy.ndarray
            The corresponding label for feature array
        folds : int, optional
            The number of folds for cross validation
        """

        estimators = all_estimators(type_filter="regressor")
        for name, RegressionClass in estimators:
            if name in model_param_map:
                param_grid = model_param_map[name]
                grid_rgr = GridSearchCV(
                    RegressionClass(),
                    param_grid,
                    cv=folds,
                    scoring="neg_root_mean_squared_error",
                    verbose=0,
                    n_jobs=-1,
                    error_score="raise",
                )
                start = time.time()
                try:
                    grid_rgr.fit(train_x, train_y)
                except BaseException as error:
                    self.logger.warning(f"{name} failed due to, Error : {error}.")
                    continue
                end = time.time()
                rgr = SimpleRegressor()
                rgr.metrics["Training Score"] = -grid_rgr.best_score_
                pred_y = grid_rgr.predict(train_x)
                rgr.metrics["Mean Absolute Error"] = mean_absolute_error(
                    train_y, pred_y
                )
                rgr.metrics["Mean Squared Error"] = mean_squared_error(train_y, pred_y)
                rgr.metrics["R-Squared"] = r2_score(train_y, pred_y)
                rgr.sk_model = grid_rgr.best_estimator_
                rgr.name = name
                rgr.attributes = grid_rgr.best_params_
                rgr.train_duration = grid_rgr.refit_time_
                rgr.gridsearch_duration = end - start
                self.ranked_list.append(rgr)
            metrik = lambda rgr: rgr.metrics[self.metric]
            self.ranked_list.sort(reverse=False, key=metrik)

    def pop(self, index=0):
        """Removes SimpleRegressor from a specific index in ranked list.

        Parameters
        ----------
        index : int
            The index corresponding to the SimpleRegressor
            being removed from ranked list
        """

        return self.ranked_list.pop(index)
