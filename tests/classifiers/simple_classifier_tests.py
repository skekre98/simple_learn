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

from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from simple_learn.classifiers import SimpleClassifier


class TestSimpleClassifier(unittest.TestCase):
    def __init__(self):
        wine = datasets.load_wine()
        self.x = wine.data
        self.y = wine.target
        self.clf = SimpleClassifier()

    def test_init(self):
        self.assertEqual(clf.name, "Empty Model")
        self.assertEqual(clf.training_accuracy, 0.0)

    def test_fit(self):
        clf.fit(self.x, self.y)
        self.assertIsNotNone(clf.sk_model)
        self.assertTrue(clf.metrics["Training Accuracy"] > 0.0)

    def test_predict(self):
        pred_y = clf.predict(self.x)
        self.assertTrue(accuracy_score(self.y, pred_y) > 0.97)


if __name__ == "__main__":
    unittest.main()
