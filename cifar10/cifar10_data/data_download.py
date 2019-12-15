from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


def download_dataset():
    train_dataset = datasets.CIFAR10('./data', train=True, download=True)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True)


if __name__ == '__main__':
    download_dataset()