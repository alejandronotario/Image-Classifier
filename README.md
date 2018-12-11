# AI Programming with Python Project

Final Project of the Udacity Artificial Inteligence with Python Nanodegree

Author: Alejandro Notario

Date: August 26, 2018

<hr>

<br>

## Overview

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

In this project, it is trained an image classifier to recognize different species of flowers using a dataset with 102 categories

## Flowers dataset

http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html


## Prerequisites

Python 3

Libraries

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
```

## Certificate

![alt text](https://github.com/alejandronotario/Image-Classifier/certificateAN.jpg)
