# -*- coding: UTF-8 -*-

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from ._defaults import *
# from .metrics import *

import torch

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# tuple of valid prediction type
PRED_TYPES = ( 'classification', 'regression' )

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# TODO: Implement `histogram_of_confidence()` function (see end of my .ipynb
#       histopathologic cancer deep learning model)

@torch.no_grad()
def predict( model:torch.nn, final_activ:torch.nn.functional, dl, out_type, with_preds = False ):
  model.eval()
  for n, batch in enumerate( dl ):
    batch = [ nb.to( device = DEFAULT_DEVICE ) for nb in batch ]
    preds_batch = model( batch[0] )
    outputs_batch = apply_final_activ( \
      input = preds_batch, out_type = out_type, final_activ = final_activ, with_preds = with_preds
    )
    if with_preds: outputs_batch, preds_batch = outputs_batch
    if not n:
      if with_preds:
        preds = torch.empty( len( dl.dataset ), preds_batch.shape[1], \
                device = DEFAULT_DEVICE, dtype = preds_batch.dtype )
      outputs = torch.empty( len( dl.dataset ), \
                device = DEFAULT_DEVICE, dtype = outputs_batch.dtype )
    if with_preds:
      preds[ n*dl.batch_size : (n+1)*dl.batch_size, : ] = preds_batch
    outputs[ n*dl.batch_size : (n+1)*dl.batch_size ] = outputs_batch

  if with_preds: _re = ( outputs, preds )
  else: _re = output

  return _re

@torch.no_grad()
def apply_final_activ( input, out_type, final_activ, with_preds = False ):
  assert ( out_type in PRED_TYPES )

  preds = final_activ( input, dim = 1 )
  if out_type == 'classification': output = preds.argmax( dim = 1 )
  elif out_type == 'regression': output = preds.max( dim = 1 )

  if with_preds: _re = ( output.view( preds.shape[0], -1 ).squeeze( ), preds )
  else: _re = output.view( preds.shape[0], -1 ).squeeze( )

  return _re

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
