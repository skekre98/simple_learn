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
}

function setup_env {
    install_dependencies
}

function build_package {
    python setup.py sdist bdist_wheel
}

function github_release {
    version=$(grep version simple_learn/version.py | awk -F'"' '{print $2}')
    git tag -a ${version} -m "Release version ${version}" main
    git push --tags
}

function release_teardown {
    rm -rf build/
    rm -rf dist/
    rm -rf simple_learn.egg-info
}

function build_and_release {
    github_release
    build_package
    twine check dist/*
    twine upload dist/*
    release_teardown
}
