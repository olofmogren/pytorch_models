#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""

Deep convolutional network.

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


class DeepConvEncoderModel(nn.Module):
  '''
     Dimensions are given by channels, imgdim1, imgdim2 and pooling_factor.
     Pooling divides spatial dimensions by pooling_factor.
     Convolutional layers use padding and have the same output dimension as input dimension.
  '''
  def __init__(self, model_spec, use_cuda):
    super(DeepConvEncoderModel, self).__init__()
    default_activation_conv = 'leaky_relu'
    default_activation_fc   = 'tanh'
    
    self.spec = model_spec
    self.use_cuda = use_cuda

    self.training = True

    self.spec.setdefault('pooling_factor', 2)
    # filter sizes are assumed to be odd numbers, for even padding.
    self.spec.setdefault('filter_sizes', [5])
    self.paddings = [int((fs-1)/2) for fs in self.spec['filter_sizes']]
    self.spec.setdefault('num_filters', [256]) #9#6
    self.spec.setdefault('fc_sizes', [256, 100]) #120#*2
    self.spec.setdefault('imgdim1', 32)
    self.spec.setdefault('imgdim2', 32)
    self.spec.setdefault('channels', 3)
    self.spec.setdefault('resnet', True)
    self.spec.setdefault('dropout', None)
    self.spec.setdefault('batchnorm', True)

    self.spec.setdefault('activation_conv', 'leaky_relu')
    self.spec.setdefault('activation_fc',   'tanh')
    self.spec.setdefault('activation_out',  'tanh')
        
    self.activation_conv = self.get_activation(self.spec['activation_conv'])
    self.activation_fc   = self.get_activation(self.spec['activation_fc'])
    self.activation_out  = self.get_activation(self.spec['activation_out'])

    self.convs = nn.ModuleList()
    self.batchnorms = nn.ModuleList()

    channels = self.spec['channels']
    for i in range(len(self.spec['filter_sizes'])):
      while len(self.spec['num_filters']) <= i:
        self.spec['num_filters'].append(self.spec['num_filters'][-1])
      while len(self.paddings) <= i:
        self.paddings.append(self.paddings[-1])
      self.convs.append(nn.Conv2d(channels, self.spec['num_filters'][i], self.spec['filter_sizes'][i], padding=self.paddings[i]))
      print('nn.Conv2d('+str(channels)+', '+str(self.spec['num_filters'][i])+', '+str(self.spec['filter_sizes'][i])+', padding='+str(self.paddings[i])+'))')
      self.batchnorms.append(nn.BatchNorm2d(num_features=self.spec['num_filters'][i]))
      channels = self.spec['num_filters'][i]
    #self.encoder_weights.append((self.conv1.weight,self.conv1.bias))
    # padding. size constant.
    self.pool = nn.MaxPool2d(self.spec['pooling_factor'], return_indices=True)
    self.num_poolings = 2

    if self.spec['dropout'] is not None:
      self.dropout = nn.Dropout(p=self.spec['dropout'], inplace=False)

    fc1_input_size = int(self.spec['imgdim1']/self.spec['pooling_factor']**self.num_poolings)*int(self.spec['imgdim2']/self.spec['pooling_factor']**self.num_poolings)*self.spec['num_filters'][-1]

    self.fcs = nn.ModuleList()
    sizes = [fc1_input_size]+self.spec['fc_sizes']
    print(sizes)
    for i in range(len(sizes)-1):
      self.fcs.append(nn.Linear(sizes[i], sizes[i+1]))

    for w in self.parameters():
      #print('parameter {}'.format(w.size()))
      if self.spec['init'] == 'xavier':
        torch.nn.init.xavier_uniform_(w)
      elif self.spec['init'] == 'uniform':
        torch.nn.init.uniform_(w)

    if self.use_cuda:
      for i in range(len(self.convs)):
        self.convs[i].cuda()
        self.batchnorms[i].cuda()
      self.pool.cuda()
      for i in range(len(self.fcs)):
        self.fcs[i].cuda()
    print('Model: {} conv layers, {} fc layers. Output dimension {}.'.format(len(self.convs), len(self.fcs), self.spec['fc_sizes'][-1]))

  def forward(self, x):
    #print('before first conv layer: x: {}'.format(x.size()))
    for i in range(len(self.convs)):
      if i % 2 == 0:
        prev = x
      x = self.convs[i](x)
      if i % 2 == 0 and self.spec['resnet']:
        x = x+prev
      if self.spec['batchnorm']:
        x = self.batchnorms[i](x)
      if self.activation_conv is not None:
        x = self.activation_conv(x)
      if self.spec['dropout'] is not None and self.training:
        #keep = torch.empty(x.size()).uniform_(0, 1) > self.spec['dropout']
        #keep = keep.cuda() if self.use_cuda else keep
        #z = torch.zeros(x.size())
        #z = z.cuda() if self.use_cuda else z
        #x = torch.where(keep, x, z)
        #x = x*(1/(1-self.spec['dropout']))
        x = self.dropout(x)
      #print('after first conv layer: {}'.format(x.size()))
      if not self.spec['pooling_factor'] == 1 and i < self.num_poolings:
        x, indices_pool1 = self.pool(x)
    #print('before coding layer: {}'.format(x.size()))
    x = x.view(x.size()[0], -1)
    #print('before fc1: {}'.format(x.size()))
    for i in range(len(self.fcs)):
      x = self.fcs[i](x)
      if i == len(self.fcs)-1:
        if self.activation_out is not None:
          x = self.activation_out(x)
      else:
        if self.activation_fc is not None:
          x = self.activation_fc(x)
        reps = x
    return x, reps

  def cuda(self):
    super(DeepConvEncoderModel, self).cuda()
    self.use_cuda = True

  def eval(self):
    super(DeepConvEncoderModel, self).eval()
    self.trainng = False

  def train(self, choice):
    super(DeepConvEncoderModel, self).train(choice)
    self.training = choice

  def get_spec(self):
    return self.spec

  def get_activation(self, label):
    if label == 'linear':
      return None
    elif label == 'logsoftmax':
      return nn.LogSoftmax(dim=-1)
    else:
     return getattr(torch, label)
