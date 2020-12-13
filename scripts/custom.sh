#!/bin/bash

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

function install_dependencies {
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install pre-commit
    python -m pip install tqdm
    python -m pip install twine
    python -m pip install sklearn
}

function setup_env {
    install_dependencies
}

function build_package {
    python setup.py sdist bdist_wheel
}

function trial_script {
    pip install -e ."[devel]"
    python runner.py
    pip uninstall -y simple_learn
}

function github_release {
    version=$(grep version simple_learn/version.py | awk -F'"' '{print $2}')
    git add .
    git commit -m "New release"
    git checkout -b release-${version}
    rm -rf scripts/
    rm -rf build/
    rm -rf simple_learn.egg-info
    rm -rf simple_learn/
    rm .gitignore
    rm .pre-commit-config.yaml
    rm CONTRIBUTING.md
    rm LICENSE
    rm README.md
    rm setup.py
    mv dist/* .
    rm -rf dist/
    git add .
    git commit -a -m "Release version ${version}"
    git tag -a ${version} -m "Release version ${version}" release-${version}
    git push origin ${version}
    git checkout main
    git branch -D release-${version}
}

function release_teardown {
    rm -rf build/
    rm -rf dist/
    rm -rf simple_learn.egg-info
}

function build_and_release {
    build_package
    twine check dist/*
    twine upload dist/*
    github_release
    release_teardown
}

function run_tests {
    python -W -m unittest tests/classifiers/simple_classifier_tests.py -v
}
