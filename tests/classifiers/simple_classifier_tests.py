import unittest

from sklearn import datasets
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
        print(clf.sk_model)
