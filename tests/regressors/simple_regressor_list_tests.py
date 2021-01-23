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

from simple_learn.regressors import SimpleRegressorList


class TestSimpleRegressorList(unittest.TestCase):
    def test_regression(self):
        true_x, true_y = make_regression(
            n_features=4, n_informative=2, random_state=10, shuffle=False
        )

        rgr_list = SimpleRegressorList()
        rgr_list.fit(true_x, true_y)
        self.assertTrue(len(rgr_list.ranked_list) > 0)

        rgr1 = rgr_list.pop(1)
        rgr0 = rgr_list.pop()

        pred0_y = rgr0.predict(true_x)
        pred1_y = rgr1.predict(true_x)

        stat0, p0 = levene(true_y, pred0_y)
        stat1, p1 = levene(true_y, pred1_y)

        self.assertTrue(p0 > 0.05)
        self.assertTrue(p1 < p0)

    def test_boston(self):
        boston = datasets.load_boston()
        true_x = boston.data
        true_y = boston.target

        rgr_list = SimpleRegressorList()
        rgr_list.fit(true_x, true_y)
        self.assertTrue(len(rgr_list.ranked_list) > 0)

        rgr1 = rgr_list.pop(1)
        rgr0 = rgr_list.pop()

        pred0_y = rgr0.predict(true_x)
        pred1_y = rgr1.predict(true_x)

        stat0, p0 = levene(true_y, pred0_y)
        stat1, p1 = levene(true_y, pred1_y)

        self.assertTrue(p0 > 0.05)
        self.assertTrue(p1 < p0)


if __name__ == "__main__":
    unittest.main()
