#!/bin/bash

function install_dependencies {
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install tqdm
    python -m pip install twine
}

function setup_env {
    install_dependencies
}