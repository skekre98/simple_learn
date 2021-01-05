import json
import time

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import all_estimators
from simple_learn.regressors.param_grid import model_param_map


class SimpleRegressor:
    def __init__(self):
        self.name = "Empty Model"
        self.sk_model = None
        self.attributes = dict()
        self.metrics = dict()
        self.gridsearch_duration = None
        self.train_duration = None
        self.failed_models = []

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
        estimators = all_estimators(type_filter="regressor")
        for name, RegressionClass in estimators:
            if name in model_param_map:
                param_grid = model_param_map[name]
                grid_clf = GridSearchCV(
                    RegressionClass(),
                    param_grid,
                    cv=folds,
                    scoring='neg_root_mean_squared_error',
                    verbose=0,
                    n_jobs=-1,
                    error_score='raise'
                )
                start = time.time()
                try:
                    grid_clf.fit(train_x, train_y)
                except ValueError as e:
                    self.failed_models.append(name)
                    print("Model: {}, Error : {} ,".format(name,e))
                    continue
                end = time.time()
                if self.metrics.get("Training Score") is None or -grid_clf.best_score_ < self.metrics.get("Training Score"):
                    self.metrics["Training Score"] = -grid_clf.best_score_
                    pred_y = grid_clf.predict(train_x)
                    self.metrics["mae"] = mean_absolute_error(
                        train_y, pred_y
                    )
                    self.metrics["rmse"] = mean_squared_error(
                        train_y, pred_y,squared=False
                    )
                    self.metrics["r2"] = r2_score(
                        train_y, pred_y
                    )
                    self.sk_model = grid_clf.best_estimator_
                    self.name = name
                    self.attributes = grid_clf.best_params_
                    self.train_duration = grid_clf.refit_time_
                    self.gridsearch_duration = end - start

    def predict(self, pred_x):
        return self.sk_model.predict(pred_x)