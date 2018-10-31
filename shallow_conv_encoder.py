#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""

Shallow convolutional network.

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


class ConvEncoderModel(nn.Module):
  '''
     Dimensions are given by channels, imgdim1, imgdim2 and pooling_factor.
     Pooling divides spatial dimensions by pooling_factor.
     Convolutional layers use padding and have the same output dimension as input dimension.
  '''
  def __init__(self, model_spec, use_cuda):
    super(ConvEncoderModel, self).__init__()
    default_activation_conv = 'leaky_relu'
    default_activation_fc   = 'tanh'
    
    self.pooling_factor = model_spec.get('pooling_factor', 2)
    self.filter_size_1 = model_spec.get('filter_size_1', 9)
    self.padding_1 = int((self.filter_size_1-1)/2),
    self.num_filters_1 = model_spec.get('num_filters_1', 128)
    self.fc1_size = model_spec.get('encoding_size', 100)
    self.imgdim1 = model_spec.get('imgdim1', 32)
    self.imgdim2 = model_spec.get('imgdim2', 32)
    self.channels = model_spec.get('channels', 3)
        
    self.activation_conv   = self.get_activation(model_spec.get('activation_conv', default_activation_conv))
    self.activation_out     = self.get_activation(model_spec.get('activation_out', default_activation_fc))

    self.spec = model_spec

    self.conv1 = nn.Conv2d(self.channels, self.num_filters_1, self.filter_size_1, padding=self.padding_1)
    #self.encoder_weights.append((self.conv1.weight,self.conv1.bias))
    # padding. size constant.
    self.pool = nn.MaxPool2d(self.pooling_factor, return_indices=True)
    #fc1_input_size = int(self.imgdim1/self.pooling_factor)*int(self.imgdim2/self.pooling_factor)*self.num_filters_1
    fc1_input_size = self.imgdim1*self.imgdim2*self.num_filters_1
    print("fc1_input_size: {}, x: {}, y: {}".format(fc1_input_size, self.imgdim1, self.imgdim2))
    self.fc1 = nn.Linear(fc1_input_size, self.fc1_size)

    for w in self.parameters():
      if model_spec['init'] == 'xavier':
        torch.nn.init.xavier_uniform_(w)
      elif model_spec['init'] == 'uniform':
        torch.nn.init.uniform_(w)

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
    reps = x
    x = self.fc1(x)
    if self.activation_out is not None:
      x = self.activation_out(x)
    return x, reps

  def get_spec(self):
    return self.spec

  def get_activation(self, label):
    if label == 'linear':
      return None
    elif label == 'logsoftmax':
      return nn.LogSoftmax(dim=-1)
    else:
     return getattr(torch, label)
