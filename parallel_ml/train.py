# -*- coding: UTF-8 -*-

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from ._defaults import *
from .metrics import *

import torch

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def backprop_batch( loss, opt = None ):

  if opt is not None:
    loss.backward()
    opt.step()
    opt.zero_grad()

  return loss.item()

def fit( epochs, model, loss_func:torch.nn.functional, opt, train_dl, valid_dl, metrics = [] ):
  for epoch in range( epochs ):
    model.train()
    losses = torch.empty( len( train_dl ), \
                          device = DEFAULT_DEVICE, dtype = torch.float32 )
    nums = torch.empty( len( train_dl ), \
                          device = DEFAULT_DEVICE, dtype = torch.float32 )
    for n, train_batch in enumerate( train_dl ):
      xb, yb = train_batch
      xb, yb = xb.to( device = DEFAULT_DEVICE ), yb.to( device = DEFAULT_DEVICE )
      loss_train_batch = loss_func( model( xb ), yb )
      losses[n] = backprop_batch( loss_train_batch, opt )
      nums[n] = len( xb )
    loss_train = torch.sum( torch.mul( losses, nums ) ) / torch.sum( nums )

    model.eval()
    losses = torch.empty( len( valid_dl ), \
                          device = DEFAULT_DEVICE, dtype = torch.float32 )
    nums = torch.empty( len( valid_dl ), \
                          device = DEFAULT_DEVICE, dtype = torch.float32 )
    metric_vals_batches = []
    with torch.no_grad():
      for n, valid_batch in enumerate( valid_dl ):
        xb, yb = valid_batch
        xb, yb = xb.to( device = DEFAULT_DEVICE ), yb.to( device = DEFAULT_DEVICE )
        outb = model( xb )
        metric_vals_batches[n:] = \
          [ implement_metrics_batch( metrics, outb, yb ) ]
        loss_valid_batch = loss_func( outb, yb )
        losses[n] = loss_valid_batch.item()
        nums[n] = len( xb )
    loss_valid = torch.sum( torch.mul( losses, nums ) ) / torch.sum( nums )
    metrics_vals = \
      [ ( torch.sum( \
        torch.FloatTensor( [ metric_vals_batch[i].item()*nums[n] for \
          n, metric_vals_batch in \
          enumerate( metric_vals_batches ) ]
        ) ) / torch.sum( nums ) ).item() for i in range( len( metrics ) )
      ]

    print( epoch, loss_train.item(), loss_valid.item(), *metrics_vals )

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

@torch.no_grad()
def implement_metrics_batch( metrics, outb, yb ):

  metrics_vals = []
  for metric in metrics:
    # TODO: Add assertion to check if all metrics supplied in `metrics` exist
    #       in `.metrics.py`
    metrics_vals.append( METRICS_DICT[ metric ]( outb, yb ) )

  return metrics_vals
