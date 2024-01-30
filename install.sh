#!/bin/bash

pip install -U pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cu118
# pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install pyyaml ipywidgets
