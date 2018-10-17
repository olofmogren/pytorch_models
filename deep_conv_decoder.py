#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""

Convolutional auto-encoder.

Author Olof Mogren, olof@mogren.one

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


class DeepDecoderModel(nn.Module):
  '''
     Dimensions are given by channels, imgdim1, imgdim2 and pooling_factor.
     Pooling divides spatial dimensions by pooling_factor.
     Decoder reconstructs the same dimensions.
     Convolutional layers use padding and have the same output dimension as input dimension.
  '''
  def __init__(self, model_spec):
    super(AutoencoderModel, self).__init__()

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
    self.channels = channels
    self.imgdim1 = imgdim1
    self.imgdim2 = imgdim2

    if model_spec['activation_conv'] == 'linear':
      self.activation_dconv   = None
    else:
      self.activation_dconv   = getattr(F, model_spec.get('activation_conv', 'leaky_relu'))
    if model_spec['activation_fc'] == 'linear':
      self.activation_dfc     = None
    else
      self.activation_dfc     = getattr(F, model_spec.get('activation_fc', 'tanh'))

    self.spec = model_spec


    self.d_fc2 = nn.Linear(self.fc2_size, self.fc1_size)
    self.d_fc1 = nn.Linear(self.fc1_size, fc1_input_size)
    self.d_conv2 = nn.ConvTranspose2d(self.num_filters_2, self.num_filters_1, self.filter_size_2, padding=self.padding_2)
    self.d_pool = nn.MaxUnpool2d(self.pooling_factor, self.pooling_factor)
    self.d_conv1 = nn.ConvTranspose2d(self.num_filters_1, self.channels, self.filter_size_1, padding=self.padding_1)

    if use_cuda:
      self.d_fc2.cuda()
      self.d_fc1.cuda()
      self.d_conv2.cuda()
      self.d_pool.cuda()
      self.d_conv1.cuda()

  def forward(self, x, classifier_pretraining=False):
    x = self.d_fc2(x)
    if self.activation_dfc is not None:
      x = self.activation_dfc(x)
    x = self.d_fc1(x)
    if self.activation_dfc is not None:
      x = self.activation_dfc(x)
    #print('after two fully connected decoder layers: {}'.format(x.size()))
    x = x.view(-1, self.num_filters_2, int(self.imgdim1/self.pooling_factor**2), int(self.imgdim2/self.pooling_factor**2))
    x = self.d_pool(x, output_size=torch.Size([args.batch_size, self.channels, int(self.imgdim1/self.pooling_factor), int(self.imgdim2/self.pooling_factor)]), indices=indices_pool2)
    #x = self.d_pool2(x)
    #print('before first convtranspose: ({},{})'.format(dim1, dim2))
    x = self.d_conv2(x)
    if self.activation_dconv is not None:
      x = self.activation_dconv(x)
    x = self.d_pool(x, output_size=torch.Size([args.batch_size, self.channels, self.imgdim1, self.imgdim2]), indices=indices_pool1)
    #x = self.d_pool1(x)
    #print('before last convtranspose: ({},{}), x: {}'.format(dim1, dim2, x.size()))
    x = self.d_conv1(x) # b, 3, 32, 32
    x = self.activation_output(x) # b, 3, 32, 32
    #print('returning x: {}'.format(x.size()))
    return x
