# -*- coding: UTF-8 -*-

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from ._defaults import *

import torch
from torch.utils.data import DataLoader

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

AUGMENTATIONS = (
  'flip_lr',
	'random_crop'
)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def construct_dl( train_ds, valid_ds, bs ):
  return (
    DataLoader( train_ds, batch_size = bs, shuffle = True ),
    DataLoader( valid_ds, batch_size = bs ),
  )

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# 1D CNN data augmentation
# TODO: make this a separate module and make `data` a sub-package.

def apply_augs_1d( tensor, **augs ):

  assert all( aug in AUGMENTATIONS for aug in augs )

  if 'flip_lr' in augs:
    flip_bool = torch.rand(1).item() < augs[ 'flip_lr' ]
    if flip_bool: tensor = torch.flip( input = tensor, dims = (-1,) )
  if 'random_crop' in augs:
    crop_idx = torch.randint(
                 low = 0,
                 high = tensor.size(-1) - augs[ 'random_crop' ] + 1,
                 size = (1,)
               ).item()
    tensor = tensor[ ..., crop_idx : crop_idx + augs[ 'random_crop' ] ]

  return tensor

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
