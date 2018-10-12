#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Copyright (C) 2017 Olof Mogren

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
'''

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, random, datetime, time, math, random, os, argparse, sys, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from subprocess import Popen,PIPE



def print_len_stats():
  maxlen = 0
  minlen = 100
  lens = {}
  for p in relations:
    for l in relations:
      for r in relations[l]:
        for (w1,t1),(w2,t2) in r['wordpairs']:
          lens[len(w1)] = lens.get(len(w1), 0)+1
          lens[len(w2)] = lens.get(len(w2), 0)+1
          maxlen = max(maxlen, len(w1))
          maxlen = max(maxlen, len(w2))
          minlen = min(minlen, len(w1))
          minlen = min(minlen, len(w2))
  print('wordlens: min {}, max{}.'.format(minlen, maxlen))
  l = list(lens.keys())
  l.sort()
  for ln in l:
    print('len: {}, num: {}.'.format(ln, lens[ln]))

def initialize_vocab(_vocab=None):
  global vocab, vocab_size, reverse_vocab
  if _vocab is not None:
    vocab = _vocab
    print('Found vocab: \'{}\''.format(''.join(vocab)))
  else:
    special_tokens = []
    special_tokens.append('<PAD>')
    special_tokens.append('<BOS>')
    special_tokens.append('<EOS>')
    special_tokens.append('<UNK>')
    special_tokens.append('<DRP>')
    if args.native_vocab:
      biglist = []
      for l in languages:
        for i in range(len(native_characters[l])):
          biglist.append(native_characters[l][i])
      biglist = sorted(biglist)
      vocab += biglist
    else:
      for i in range(len(all_characters)):
        vocab.append(all_characters[i])
    vocab = special_tokens+sorted(list(set(vocab)))
    print('Constructed vocab: \'{}\''.format(''.join(vocab)))
  for i in range(len(vocab)):
    reverse_vocab[vocab[i]] = i
  vocab_size = len(vocab)
  print('Vocab size: {}'.format(vocab_size))
  
# turn a unicode string to plain ascii

#def unicodeToAscii(s):
#  return ''.join(
#    c for c in unicodedata.normalize('NFD', s)
#    if unicodedata.category(c) != 'Mn'
#    and c in all_characters
#  )

#print(unicodeToAscii('Ślusàrski'))

def line_to_index_tensor(lines, pad_before=True, append_bos_eos=False, reverse=False, drop_char_p=0.0):
  if reverse:
    lines = [l[::-1] for l in lines]
  seqlen = max([len(l) for l in lines])
  seqlen = min(seqlen, max_sequence_len)
  if append_bos_eos:
    seqlen += 2
  tensor = torch.zeros(len(lines), seqlen).long()
  tensor += reverse_vocab['<PAD>']
  for b in range(len(lines)):
    begin_pos = 0
    if pad_before:
      begin_pos = max(0,seqlen-len(lines[b]))
    else:
      begin_pos = 0
    if append_bos_eos:
      begin_pos += 1
      tensor[b][begin_pos-1] = reverse_vocab['<BOS>']
    for li, letter in enumerate(lines[b]):
      idx = li+begin_pos
      if idx >= seqlen:
        break
      if drop_char_p>0.0 and random.random() < drop_char_p:
        tensor[b][idx] = reverse_vocab['<DRP>']
      tensor[b][idx] = reverse_vocab.get(letter, reverse_vocab['<UNK>'])
    if append_bos_eos:
      eos_pos = min(seqlen-1,begin_pos+len(lines[b]))
      tensor[b][eos_pos] = reverse_vocab['<EOS>']
  if use_cuda:
    tensor = tensor.cuda()
  return tensor

def levenshtein(seq1, seq2):
  '''
    Trusting https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    on this one. Just python3-ified it a bit.
  '''
  oneago = None
  thisrow = list(range(1, len(seq2) + 1)) + [0]
  for x in range(len(seq1)):
    twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
    for y in range(len(seq2)):
      delcost = oneago[y] + 1
      addcost = thisrow[y - 1] + 1
      subcost = oneago[y - 1] + (seq1[x] != seq2[y])
      thisrow[y] = min(delcost, addcost, subcost)
  return thisrow[len(seq2) - 1]

def compute_edit_trees(relations):
  p = 'train'
  for l in relations[p]:
    for r in relations[p][l]:
      r['edittrees'] = set()
      r['edittrees_r'] = set()
      for (w1,t1),(w2,t2) in r['wordpairs']:
        r['edittrees'].add(get_edit_tree(w1,w2))
        r['edittrees_r'].add(get_edit_tree(w2,w1))

def apply_edit_tree(source, tree):
  if tree is None:
    return source
  if tree[0] == 'replace':
    if source == tree[1]:
      return tree[2]
    else:
      return None
  if tree[0] == 'edit':
    prefix = apply_edit_tree(source[:tree[1]], tree[2])
    suffix = apply_edit_tree(source[len(source)-tree[3]:], tree[4])
    if prefix is None or suffix is None:
      return None
    return prefix+source[tree[1]:len(source)-tree[3]]+suffix

def get_edit_tree(source, target):
  if len(source) == 0 or len(target) == 0:
    return ('replace', source, target)
  lcs = longest_common_substring(source, target)
  if len(lcs):
    begin_s   = source.find(lcs)
    neg_end_s = len(source)-begin_s-len(lcs)
    begin_t   = target.find(lcs)
    neg_end_t = len(target)-begin_t-len(lcs)
    left_tree = None
    right_tree = None
    if begin_s > 0 or begin_t > 0:
      left_tree = get_edit_tree(source[:begin_s], target[:begin_t])
    if neg_end_s > 0 or neg_end_t > 0:
      right_tree = get_edit_tree(source[len(source)-neg_end_s:], target[len(target)-neg_end_t:])
  else:
    return ('replace', source, target)
    
  return ('edit', begin_s, left_tree, neg_end_s, right_tree)

def longest_common_substring(s1, s2):
   m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
   longest, x_longest = 0, 0
   for x in range(1, 1 + len(s1)):
     for y in range(1, 1 + len(s2)):
       if s1[x - 1] == s2[y - 1]:
         m[x][y] = m[x - 1][y - 1] + 1
         if m[x][y] > longest:
           longest = m[x][y]
           x_longest = x
       else:
         m[x][y] = 0
   return s1[x_longest - longest: x_longest]

def randomChoice(l):
  return l[random.randint(0, len(l) - 1)]


def topi(x):
  topv, topi = x.data.topk(1)
  return topi

def to_scalar(var):
  # returns a python float
  return var.view(-1)[0]
  #return var.view(-1).data.tolist()[0]

def word_tensor_to_string(t, handle_special_tokens=False):
  word = ''
  for o in range(t.size()[0]):
    index = to_scalar(t[o].data)
    if index == reverse_vocab['<PAD>']:
      continue
    if handle_special_tokens:
      if index == reverse_vocab['<EOS>']:
        break
      elif index == reverse_vocab['<BOS>']:
        continue
    word += vocab[index]
  return word

def prediction_tensor_to_string(t):
  word = ''
  for o in t:
    index = to_scalar(topi(o))
    if index == reverse_vocab['<EOS>']:
      break
    elif index == reverse_vocab['<PAD>']:
      continue
    elif index == reverse_vocab['<BOS>']:
      continue
    word += vocab[index]
  return word

