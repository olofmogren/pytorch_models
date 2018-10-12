#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""

Generic RNN encoder.

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


class RnnEncoderModel(nn.Module):
  '''
     Dimensions are given by channels, imgdim1, imgdim2 and pooling_factor.
     Pooling divides spatial dimensions by pooling_factor.
     Convolutional layers use padding and have the same output dimension as input dimension.
  '''
  def __init__(self, model_spec, use_cuda):
    super(DeepEncoderModel, self).__init__()
    
    self.vocab_size      = model_spec.get('vocab_size', 128)
    self.embedding_size  = model_spec.get('embedding_size', 128)
    self.rnn_depth       = model_spec.get('rnn_depth', 2)
    self.rnn_state_size  = model_spec.get('rnn_state_size', 128)
    self.bidirectional   = model_spec.get('bidirectional', False) #9#6
    self.num_directions  = 2 if self.bidirectional else 1
    self.fc_out          = model_spec.get('fc_out', False) #9#6
    self.fc_size         = model_spec.get('fc_size', False) #9#6
        
    if model_spec['activation_fc'] == 'linear':
      self.activation_fc = None
    else:
      self.activation_fc = getattr(F, model_spec.get('activation_fc', 'tanh'))

    self.spec = model_spec   

    self.embedding       = nn.Embedding(self.vocab_size, self.embedding_size)
    self.rnn             = nn.GRU(input_size=self.embedding_size,
                                  hidden_size=self.rnn_state_size,
                                  num_layers=rnn_depth, bidirectional=self.bidirectional)

    self.fc = None
    if self.fc_out:
      self.fc = nn.Linear(self.rnn_state_size*(2 if self.bidirectional else 1)*self.rnn_depth, self.fc_size)

    self.use_cuda = use_cuda
    if use_cuda:
      self.rnn.cuda()
      self.fc.cuda()

  def forward(self, x):
    '''
      arguments: x: shape: (seq, batch)
      returns x,out, where x is the last output, or the first and last concatenated if bidirectional. 
                outs are all outputs. shape: ()
    '''
    seq_len = x.size(0)
    batch_size = x.size(1)

    h = torch.zeros(self.rnn_depth * self.num_directions, x.size(1), self.rnn_state_size)
    h = h.cuda() if self.use_cuda else h

    x = self.embedding(x)
    outs,h = self.rnn(x, h)
    if self.bidirectional_encoders:
      x = x.view(seq_len, batch_size, self.num_directions, self.rnn_state_size)
      x = torch.cat((outs[-1,:,0,:], outs[0,:,1,:]), dim=1)
    else:
      x = outs[-1]
    if self.fc is not None:
      x = self.fc(x)
      if self.activation_fc is not None:
        x = self.activation_fc(x)
    return x,outs

  def get_spec(self):
    return self.spec

