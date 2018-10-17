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


class EncoderModel(nn.Module):
  '''
     Dimensions are given by channels, imgdim1, imgdim2 and pooling_factor.
     Pooling divides spatial dimensions by pooling_factor.
     Decoder reconstructs the same dimensions.
     Convolutional layers use padding and have the same output dimension as input dimension.
  '''
  def __init__(self, model_spec, use_cuda):
    super(EncoderModel, self).__init__()
    self.pooling_factor = model_spec.get('pooling_factor', 2)
    self.filter_size_1 = model_spec.get('filter_size_1', 3)
    self.padding_1 = int((self.filter_size_1-1)/2),
    self.num_filters_1 = model_spec.get('num_filters_1', 16)
    self.fc1_size = model_spec.get('encoding_size', 100)
    self.imgdim1 = model_spec.get('imgdim1', 32)
    self.imgdim2 = model_spec.get('imgdim2', 32)
    self.channels = model_spec.get('channels', 3)

    if model_spec['activation_conv'] == 'linear':
      self.activation_conv   = None
    else:
      self.activation_conv   = getattr(F, model_spec.get('activation_conv', 'leaky_relu'))
    if model_spec['activation_fc'] == 'linear':
      self.activation_fc     = None
    else:
      self.activation_fc     = getattr(F, model_spec.get('activation_fc', 'tanh'))

    self.spec = model_spec

    self.conv1 = nn.Conv2d(self.channels, self.num_filters_1, self.filter_size_1, padding=self.padding_1)
    if model_spec['init'] == 'xavier':
      torch.nn.init.xavier_uniform_(self.conv1.weight)
    elif model_spec['init'] == 'uniform':
      torch.nn.init.uniform_(self.conv1.weight)
    #self.encoder_weights.append((self.conv1.weight,self.conv1.bias))
    # padding. size constant.
    self.pool = nn.MaxPool2d(self.pooling_factor, return_indices=True)
    #fc1_input_size = int(self.imgdim1/self.pooling_factor)*int(self.imgdim2/self.pooling_factor)*self.num_filters_1
    fc1_input_size = self.imgdim1*self.imgdim2*self.num_filters_1
    print("fc1_input_size: {}, x: {}, y: {}".format(fc1_input_size, self.imgdim1, self.imgdim2))
    self.fc1 = nn.Linear(fc1_input_size, self.fc1_size)
    if model_spec['init'] == 'xavier':
      torch.nn.init.xavier_uniform_(self.fc1.weight)
    elif model_spec['init'] == 'uniform':
      torch.nn.init.uniform_(self.fc1.weight)

    if use_cuda:
      self.conv1.cuda()
      self.pool.cuda()
      self.fc1.cuda()

  def forward(self, x):
    #print('before conv layer: ({},{}). x: {}'.format(self.imgdim1, self.imgdim2, x.size()))
    x = self.conv1(x)
    if self.activation_conv is not None:
      x = self.activation_conv(x)
    #print('after conv layer: x: {}'.format(x.size()))
    #x, indices_pool1 = self.pool(x)
    #print('before fc layer: x: {}'.format(x.size()))
    x = x.view(x.size()[0], -1)
    #print('viewed before fc layer: x: {}'.format(x.size()))
    #print('before conv layer: ({},{}). x: {}'.format(self.imgdim1, self.imgdim2, x.size()))
    x = self.fc1(x)
    if self.activation_fc is not None:
      x = self.activation_fc(x)
    return x

  def get_spec(self):
    return self.spec

