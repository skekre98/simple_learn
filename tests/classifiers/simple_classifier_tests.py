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

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from simple_learn.classifiers import SimpleClassifier


class TestSimpleClassifier(unittest.TestCase):
    def test_init(self):
        clf = SimpleClassifier()
        self.assertEqual(clf.name, "Empty Model")

    def test_wine(self):
        wine = datasets.load_wine()
        true_x = wine.data
        true_y = wine.target

        clf = SimpleClassifier()
        clf.fit(true_x, true_y)
        self.assertIsNotNone(clf.sk_model)
        self.assertTrue(clf.metrics["Training Accuracy"] > 0.0)

        pred_y = clf.predict(true_x)
        self.assertTrue(accuracy_score(true_y, pred_y) > 0.95)

    def test_iris(self):
        iris = datasets.load_iris()
        true_x = iris.data
        true_y = iris.target

        clf = SimpleClassifier()
        clf.fit(true_x, true_y)
        self.assertIsNotNone(clf.sk_model)
        self.assertTrue(clf.metrics["Training Accuracy"] > 0.0)

        pred_y = clf.predict(true_x)
        self.assertTrue(accuracy_score(true_y, pred_y) > 0.95)

    def test_digits(self):
        digits = datasets.load_digits()
        true_x = digits.data
        true_y = digits.target

        clf = SimpleClassifier()
        clf.fit(true_x, true_y)
        self.assertIsNotNone(clf.sk_model)
        self.assertTrue(clf.metrics["Training Accuracy"] > 0.0)

        pred_y = clf.predict(true_x)
        self.assertTrue(accuracy_score(true_y, pred_y) > 0.95)

    def test_breast_cancer(self):
        breast_cancer_data = datasets.load_breast_cancer()
        true_x = breast_cancer_data.data
        true_y = breast_cancer_data.target

        clf = SimpleClassifier()
        clf.fit(true_x, true_y)
        self.assertIsNotNone(clf.sk_model)
        self.assertTrue(clf.metrics["Training Accuracy"] > 0.0)

        pred_y = clf.predict(true_x)
        self.assertTrue(accuracy_score(true_y, pred_y) > 0.95)


if __name__ == "__main__":
    unittest.main()
