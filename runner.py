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

# ML package
from sklearn.datasets import load_iris, make_regression

# pip package
from simple_learn.classifiers import SimpleClassifier, SimpleClassifierList
from simple_learn.regressors import SimpleRegressor


class SimpleRunner:
    def __init__(self):
        iris = load_iris()
        self.x = iris.data
        self.y = iris.target
        self.reg_x, self.reg_y = make_regression(
            n_features=4, n_informative=2, random_state=10, shuffle=False
        )

    def run(self):
        # print SimpleClassifier created by iris dataset
        print("\nCreating classification model...")
        self.simple_classifier()

        print("\nCreating classification rankings...")
        self.simple_classifier_list()

        print("\nCreating regression model...")
        self.simple_regressor()

    def simple_classifier(self):
        clf = SimpleClassifier()
        clf.fit(self.x, self.y)
        print(clf)
        clf.save()

    def simple_classifier_list(self):
        clf_list = SimpleClassifierList()
        clf_list.fit(self.x, self.y)
        print(clf_list)

    def simple_regressor(self):
        rgr = SimpleRegressor()
        rgr.fit(self.reg_x, self.reg_y)
        print(rgr)
        rgr.save()


def main():
    SR = SimpleRunner()
    SR.run()


if __name__ == "__main__":
    main()
