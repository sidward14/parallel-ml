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

def fit( epochs, model, loss_func:torch.nn.functional, opt, train_dl, valid_dl = None, *metrics ):
  _num_metrics = len( metrics )
  _metrics_idx_lst = _num_metrics * np.arange( len( valid_dl ) )
  for epoch in range( epochs ):
    model.train()
    losses = torch.empty( len( train_dl ), \
                          device = DEFAULT_DEVICE, dtype = torch.float32 )
    _nums = torch.empty( len( train_dl ), \
                          device = DEFAULT_DEVICE, dtype = torch.float32 )
    for n, train_batch in enumerate( train_dl ):
      xb, yb = train_batch
      xb, yb = xb.to( device = DEFAULT_DEVICE ), yb.to( device = DEFAULT_DEVICE )
      yb.requires_grad_( False )
      loss_train_batch = loss_func( model( xb ), yb )
      losses[n] = backprop_batch( loss_train_batch, opt )
      _nums[n] = len( xb )
    loss_train = torch.sum( torch.mul( losses, _nums ) ) / _nums.sum()

    # Compute metrics:
    metrics_vals = []
    loss_valid = []
    if valid_dl:
      model.eval()
      losses = torch.empty( len( valid_dl ), \
                            device = DEFAULT_DEVICE, dtype = torch.float32 )
      _nums = torch.empty( len( valid_dl ), \
                            device = DEFAULT_DEVICE, dtype = torch.float32 )
      metric_vals_batches = []
      with torch.no_grad():
        for n, valid_batch in enumerate( valid_dl ):
          xb, yb = valid_batch
          xb, yb = xb.to( device = DEFAULT_DEVICE ), yb.to( device = DEFAULT_DEVICE )
          outb = model( xb )
          if metrics:  # i.e. metrics other than validation loss
            metric_vals_batches[ _metrics_idx_lst[n]: ] = \
              [ implement_metrics_batch( metrics, outb, yb ) ]
          loss_valid_batch = loss_func( outb, yb )
          losses[n] = loss_valid_batch.item()
          _nums[n] = len( xb )
        loss_valid = [ ( torch.sum( torch.mul( losses, _nums ) ) / _nums.sum() ).item() ]
        metrics_vals = \
          [ ( torch.sum( \
            torch.FloatTensor( [ metric_vals_batches[n+i].item()*_nums[nn] for \
              nn, n in enumerate( _metrics_idx_lst ) ]
            ) ) / torch.sum( _nums ) ).item() for i in range( len( metrics ) )
          ]

    print( epoch, loss_train.item(), *loss_valid, *metrics_vals )

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

@torch.no_grad()
def implement_metrics_batch( metrics, outb, yb ):

  metrics_vals = []
  for metric in metrics:
    # TODO: Add assertion to check if all metrics supplied in `metrics` exist
    #       in `.metrics.py`
    metrics_vals.append( METRICS_DICT[ metric.lower() ]( outb, yb ) )

  return metrics_vals

@torch.no_grad()
def implement_metrics_batch_gan( metrics, xgenb, xb = None ):

    metrics_vals = []
    for metric in metrics:
      # TODO: Add assertion to check if all metrics supplied in `metrics` exist
      #       in `.metrics.py`
      metrics_vals.append( METRICS_GAN_DICT[ metric.lower() ]( xgenb, xb ) )

    return metrics_vals
