#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""

Convolutional auto-encoder.

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
import torch.nn as nn
import torch.nn.functional as F


class DeepEncoderModel(nn.Module):
  '''
     Dimensions are given by channels, imgdim1, imgdim2 and pooling_factor.
     Pooling divides spatial dimensions by pooling_factor.
     Convolutional layers use padding and have the same output dimension as input dimension.
  '''
  def __init__(self, model_spec, use_cuda):
    super(DeepEncoderModel, self).__init__()
    
    self.pooling_factor = model_spec.get('pooling_factor', 2)
    # filter sizes are assumed to be odd numbers, for even padding.
    self.filter_size_1 = model_spec.get('filter_size_1', 3)
    self.padding_1 = int((self.filter_size_1-1)/2)
    self.num_filters_1 = model_spec.get('num_filters_1', 16) #9#6
    self.filter_size_2 = model_spec.get('filter_size_2', 5)
    self.padding_2 = int((self.filter_size_2-1)/2)
    self.num_filters_2 = model_spec.get('num_filters_2', 32) #16
    self.fc1_size = model_spec.get('fc_size', 140) #120#*2
    self.fc2_size = model_spec.get('encoding_size', 100) #84*2
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
    # padding. size constant.
    self.pool = nn.MaxPool2d(self.pooling_factor, return_indices=True)
    self.conv2 = nn.Conv2d(self.num_filters_1, self.num_filters_2, self.filter_size_2, padding=self.padding_2)
    fc1_input_size = int(self.imgdim1/self.pooling_factor**2)*int(self.imgdim2/self.pooling_factor**2)*self.num_filters_2
    self.fc1 = nn.Linear(fc1_input_size, self.fc1_size)
    self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)

    if use_cuda:
      self.conv1.cuda()
      self.pool.cuda()
      self.conv2.cuda()
      self.fc1.cuda()
      self.fc2.cuda()

  def forward(self, x):
    #print('before first conv layer: x: {}'.format(x.size()))
    x = self.conv1(x)
    if self.activation_conv is not None:
      x = self.activation_conv(x)
    #print('after first conv layer: {}'.format(x.size()))
    x, indices_pool1 = self.pool(x)
    #print('before second conv layer: {}'.format(x.size()))
    x = self.conv2(x)
    if self.activation_conv is not None:
      x = self.activation_conv(x)
    #print('after second conv layer: {}'.format(x.size()))
    x, indices_pool2 = self.pool(x)
    #print('before coding layer: {}'.format(x.size()))
    x = x.view(x.size()[0], -1)
    #print('before fc1: {}'.format(x.size()))
    x = self.fc1(x)
    if self.activation_fc is not None:
      x = self.activation_fc(x)
    x = self.fc2(x)
    if self.activation_fc is not None:
      x = self.activation_fc(x)
    
    return x
  def get_spec(self):
    return self.spec

