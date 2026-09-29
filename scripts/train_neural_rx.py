#!/usr/bin/python3

"""Run ``ext/neural_rx/scripts/train_neural_rx.py`` as ``__main__`` (arguments are passed through)."""

import os
import runpy


target = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ext", "neural_rx", "scripts", "train_neural_rx.py")
)

runpy.run_path(target, run_name="__main__")
