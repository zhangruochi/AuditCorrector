import torch
import sys
import os

orig_cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


from pathlib import Path
from omegaconf import OmegaConf
import json

def test_train():
    with open(os.path.join(orig_cwd, "train.json"), "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert isinstance(data[0], dict)
    assert len(data) > 0

    print("total train data: {}".format(len(data)))

    for _ in data:
        print(_)
        break

def test_dev():
    with open(os.path.join(orig_cwd, "dev.json"), "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert isinstance(data[0], dict)
    assert len(data) > 0


    print("total dev data: {}".format(len(data)))

    for _ in data:
        print(_)
        break


def test_test():
    with open(os.path.join(orig_cwd, "test.json"), "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert isinstance(data[0], dict)
    assert len(data) > 0

    print("total test data: {}".format(len(data)))

    for _ in data:
        print(_)
        break
