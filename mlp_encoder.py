#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""

Convolutional siamese network to learn to search.

Author Olof Mogren, olof@mogren.one

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import argparse, datetime, os, random, math, sys
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import scipy.misc


class MLPEncoderModel(nn.Module):
  '''
     Dimensions are given by channels, imgdim1, imgdim2 and pooling_factor.
     Pooling divides spatial dimensions by pooling_factor.
     Decoder reconstructs the same dimensions.
     Convolutional layers use padding and have the same output dimension as input dimension.
  '''
  def __init__(self, model_spec, use_cuda):
    super(MLPEncoderModel, self).__init__()
    if 'fc1_size' in model_spec:
      self.fc1_size = model_spec['fc1_size']
      self.fc2_size = model_spec['encoding_size']
    else:
      self.fc1_size = model_spec.get('encoding_size', 100)
    self.input_size = model_spec.get('input_size', 32)

    if model_spec['activation_fc'] == 'linear':
      self.activation_fc     = None
    else:
      self.activation_fc     = getattr(F, model_spec.get('activation_fc', 'tanh'))

    self.spec = model_spec

    self.fc1 = nn.Linear(self.input_size, self.fc1_size)
    self.layers = [self.fc1]
    if 'fc1_size' in model_spec:
      self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
      self.layers.append(self.fc2)
    for layer in self.layers:
      if model_spec['init'] == 'xavier':
        torch.nn.init.xavier_uniform_(layer.weight)
      elif model_spec['init'] == 'uniform':
        torch.nn.init.uniform_(layer.weight)

    if use_cuda:
      self.fc1.cuda()

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
      if self.activation_fc is not None:
        x = self.activation_fc(x)
    return x

  def get_spec(self):
    return self.spec

