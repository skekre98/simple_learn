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
from sklearn.model_selection import train_test_split

from simple_learn.classifiers import SimpleClassifier


class TestSimpleClassifier(unittest.TestCase):
    def test_init(self):
        clf = SimpleClassifier()
        self.assertEqual(clf.name, "Empty Model")
        self.assertEqual(clf.training_accuracy, 0.0)

    def test_fit(self):
        dataset = datasets.load_wine()
        X = dataset.data
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        clf = SimpleClassifier()
        clf.fit(X_train, y_train)
        self.assertIsNotNone(clf.sk_model)
        self.assertTrue(clf.training_accuracy > 0.0)


if __name__ == "__main__":
    unittest.main()
