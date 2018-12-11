# AI Programming with Python Project

Final Project of the Udacity Artificial Inteligence with Python Nanodegree

Author: Alejandro Notario

Date: August 26, 2018

<hr>

<br>

## Overview

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

In this project, it is trained an image classifier to recognize different species of flowers using a dataset with 102 categories

## Parts

__Part 1 - Development Notebook__

- Preporcessing and preparing datasets wth torchvision's ImageFolder 
- Preparing train loaders, valid loaders, and test loaders with torchvision's DataLoader
- Pre-train network densenet121
- Feedforward network classifier
- Training and validation
- Save training model
- Predicting

__Part 2 - Command Line Application__

- Training new network
- Include three different architectures
- Set hyperparameters for learning rate, number of hidden units, and training epochs
- Choosing training the model on a GPU
- Predicting


## Links:

Flowers dataset

http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

PyTorch package

https://pytorch.org/

Certificate

https://confirm.udacity.com/D9YMFCKJ

## Prerequisites

Python 3

Libraries:

```python

import numpy as np
import time
import torch
from PIL import Image
import matplotlib.gridspec as gridspec
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch.utils.data import DataLoader
import sys
import json
from contextlib import contextmanager
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
```

## Certificate

![alt text](https://github.com/alejandronotario/Image-Classifier/blob/master/certificateAN.jpg)
