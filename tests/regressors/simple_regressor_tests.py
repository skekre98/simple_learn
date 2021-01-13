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

import unittest
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import levene
from sklearn import datasets
from sklearn.datasets import make_regression
from simple_learn.regressors import SimpleRegressor


class TestSimpleRegressor(unittest.TestCase):
    def test_init(self):
        clf = SimpleRegressor()
        self.assertEqual(clf.name, "Empty Model")

    def test_fail_models(self):
        """
        testing warnings of failed models with negative data value in y
        """
        true_x , true_y = [np.random.standard_normal(10),np.random.standard_normal(10),np.random.standard_normal(10)] , np.arange(-3,0)
        clf = SimpleRegressor()
        with self.assertLogs(clf.logger, level='WARNING') as cm:
            clf.fit(true_x, true_y)
            self.assertEqual(len(cm.output), len(clf.failed_models))

    def test_regression(self):
        true_x, true_y = make_regression(
            n_features=4, n_informative=2, random_state=10, shuffle=False
        )

        clf = SimpleRegressor()
        clf.fit(true_x, true_y)
        self.assertIsNotNone(clf.sk_model)
        self.assertTrue(clf.metrics["Training Score"] > 0.0)
        pred_y = clf.predict(true_x)

        stat, p = levene(true_y, pred_y)
        self.assertTrue(p > 0.05)

    def test_boston(self):
        boston = datasets.load_boston()
        true_x = boston.data
        true_y = boston.target

        clf = SimpleRegressor()
        clf.fit(true_x, true_y)
        self.assertIsNotNone(clf.sk_model)
        self.assertTrue(clf.metrics["Training Score"] > 0.0)
        pred_y = clf.predict(true_x)
        stat, p = levene(true_y, pred_y)
        self.assertTrue(p > 0.05)

if __name__ == "__main__":
    unittest.main()
