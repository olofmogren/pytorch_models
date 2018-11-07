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
    super(RnnEncoderModel, self).__init__()
    
    self.spec = model_spec
    self.use_cuda = use_cuda

    self.training = True

    self.spec.setdefault('enable_embedding', True)
    self.spec.setdefault('vocab_size', 128)
    self.spec.setdefault('embedding_size', 256)
    self.spec.setdefault('rnn_depth', 2)
    self.spec.setdefault('rnn_state_size', 256)
    self.spec.setdefault('bidirectional', False)
    self.num_directions   = 2 if self.spec['bidirectional'] else 1
    self.spec.setdefault('enable_fc_out', False)
    self.spec.setdefault('output_scale', 1)
    self.spec.setdefault('fc_size', [10])

    self.spec.setdefault('activation_fc',   'tanh')
        
    self.activation_fcs   = [self.get_activation(x) for x in self.spec['activation_fc']]

    self.embedding = None
    if self.spec['enable_embedding']:
      self.embedding      = nn.Embedding(self.spec['vocab_size'], self.spec['embedding_size'])
    self.rnn              = nn.GRU(input_size=self.spec['embedding_size'],
                                  hidden_size=self.spec['rnn_state_size'],
                                  num_layers=self.spec['rnn_depth'],
                                  bidirectional=self.spec['bidirectional'])

    self.fc = None
    self.parameters_fc_out = []

    if self.spec['enable_fc_out']:
      #self.fc = nn.Linear(self.spec['rnn_state_size']*(2 if self.spec['bidirectional'] else 1)*self.spec['rnn_depth'], self.fc_size)
      self.fcs = nn.ModuleList()
      for size in self.spec['fc_size']:
        self.fcs.append(nn.Linear(self.spec['rnn_state_size']*(2 if self.spec['bidirectional'] else 1), size))
        self.parameters_fc_out.append(self.fcs[-1].parameters())
        print('fc_size: {}'.format(size))

    #print('fc_size: {}'.format(self.fc_size))
    self.use_cuda = use_cuda
    if use_cuda:
      if self.spec['enable_embedding']:
        self.embedding.cuda()
      self.rnn.cuda()
      if self.spec['enable_fc_out']:
        for fc in self.fcs:
          fc.cuda()

  def forward(self, x):
    '''
      arguments: x: shape: (seq, batch)
      returns x,out, where x is the last output, or the first and last concatenated if bidirectional. 
                outs are all outputs. shape: ()
    '''
    seq_len = x.size(0)
    batch_size = x.size(1)

    h = torch.zeros(self.spec['rnn_depth'] * self.num_directions, batch_size, self.spec['rnn_state_size'])
    h = h.cuda() if self.use_cuda else h

    if self.spec['enable_embedding']:
      x = self.embedding(x)
    rnn_outs,h = self.rnn(x, h)
    if self.spec['bidirectional']:
      x = x.view(seq_len, batch_size, self.num_directions, self.spec['rnn_state_size'])
      x = torch.cat((rnn_outs[-1,:,0,:], rnn_outs[0,:,1,:]), dim=1)
    else:
      x = rnn_outs[-1]
    fc_outs = []
    if self.spec['enable_fc_out']:
      for fc,activation in zip(self.fcs, self.activation_fcs):
        fc_out = []
        for i in range(seq_len):
          o = fc(rnn_outs[i])
          if activation is not None:
            o = activation(o)
          #print('o: {}'.format(o.size()))
          fc_out.append(o)
        fc_out = torch.stack(fc_out, dim=0)
        fc_outs.append(fc_out)
        x = fc_outs
    else:
      x = [x]
    x = [xi*self.spec['output_scale'] for xi in x]
    #print(len(fc_outs))
    return x,rnn_outs

  def get_activation(self, label):
    if label == 'linear':
      return None
    elif label == 'logsoftmax':
      return nn.LogSoftmax(dim=-1)
    else:
     try:
       return getattr(torch, label)
     except:
       print('Unrecognized activation: {}. Using None.'.format(label))
       return None

  def parameters_fc_out(self, index):
    if omit_lower_levels:
      return self.parameters_fc_out[index]
    else:
      return self.parameters()

  def get_spec(self):
    return self.spec

