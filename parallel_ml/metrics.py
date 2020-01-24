# -*- coding: UTF-8 -*-

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from ._defaults import *
from .multiprocessing import parallel_map_func_re

from functools import partial
import warnings

import numpy as np
# import skimage.io
# import matplotlib.pyplot as plt
# import skimage.segmentation
# from scipy.sparse import coo_matrix, csc_matrix

import torch

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

@torch.no_grad()
def accuracy( input:torch.cuda.FloatTensor, targs:torch.cuda.LongTensor ):
  _n = targs.shape[0]
  input = input.argmax( dim = 1 ).view( _n, -1 )
  targs = targs.view( _n, -1 )
  return ( input == targs ).float().mean()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# TODO
@torch.no_grad()
def acc_segmentation( input:torch.cuda.FloatTensor, targs:torch.cuda.IntTensor, rm_bknd = True ):
  """The intent of this metric is to have a metric where it's easy to understand the implications of the value it outputs since the output value of mAP IoU is more difficult to understand intuitively."""
  _n = targs.shape[0]
  input = input.argmax( dim = 1 ).view( _n, -1 )
  targs = targs.view( _n, -1 )
  return ( input == targs ).float().mean()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

@torch.no_grad()
def mAP_IoU( preds:torch.cuda.FloatTensor, targs:torch.cuda.IntTensor,
  thresholds:list, num_tot_classes:int, bknd_idx:int = None, max_workers:int = 1,
  rm_bknd = True, fast_hist2d:bool = False, sparsify:bool = False ):
  """Segmentation: calculates the IoU-based precision at a specified IoU threshold (t)
  { TP(t) / ( TP(t) + FP(t) + FN(t) ) } for a single image."""

  #----------------------------------------------------------------------------#

  _fmin = 1e-9

  if rm_bknd:
    if not isinstance( bknd_idx, int ):
      raise Exception( 'If removing background is desired, please specify index of backround in preds.' )

  # The following catches whether there will be division by 0 further below:
  if not ( min( thresholds ) > _fmin and max( thresholds ) < 1 - _fmin ):
    raise Exception( 'Minimum specified threshold must be > 1e-9 and maximum specified threshold must be < (1 - 1e-9).' )

  if bknd_idx != 0:
    warnings.warn( 'Warning: To maintain consistency across segmentation models, please make `bknd_idx` be equal ' + \
      'to index number 0.' )
    first_idx = False
  else:
    first_idx = True

  #----------------------------------------------------------------------------#

  # TODO: Implement fast np.bincount/np.unique/etc. to get num classes for targs and preds

  num_classes_IU = _ if fast_hist2d else ( num_tot_classes, num_tot_classes, )

  #----------------------------------------------------------------------------#

  _preds_flat_np = preds.view( -1, preds.size()[-2]*preds.size()[-1] ).cpu().numpy().astype( np.int32 )
  _targs_flat_np = targs.view( -1, targs.size()[-2]*targs.size()[-1] ).cpu().numpy().astype( np.int32 )

  #----------------------------------------------------------------------------#

  # TODO: Implement CUDA (GPU Parallel Computing) version of histogram2d
  histogram2d_async = partial( np.histogram2d, bins = num_classes_IU, \
    range = np.array( [[-0.5, num_classes_IU[0] - .5], [-0.5, num_classes_IU[1] - .5]] ) )
  intersections = parallel_map_func_re(
    histogram2d_async,
    np.stack( ( _targs_flat_np, _preds_flat_np, ), axis = 0 ),
    max_workers = max_workers
  )

  intersections = torch.stack( tuple( \
    torch.from_numpy( intersection[0] ).to( device = DEFAULT_DEVICE, dtype = torch.float64 ) for \
    intersection in intersections ) )

  #----------------------------------------------------------------------------#

  # TODO: Implement CUDA (GPU Parallel Computing) version of histogram
  # Calculate Union. First calculate areas (needed for finding the union between all objects).
  # Shape : (true_objects, pred_objects)
  bincount1d_preds_async = partial( np.bincount, minlength = num_classes_IU[0] )
  area_preds = parallel_map_func_re(
    bincount1d_preds_async,
    [ _preds_flat_np ],
    max_workers = max_workers
  )
  area_preds = torch.stack( tuple( \
    torch.from_numpy( area_pred ).to( device = DEFAULT_DEVICE, dtype = torch.float64 ) for \
    area_pred in area_preds ) )

  bincount1d_trues_async = partial( np.bincount, minlength = num_classes_IU[0] )
  area_trues = parallel_map_func_re(
    bincount1d_trues_async,
    [ _targs_flat_np ],
    max_workers = max_workers
  )
  area_trues = torch.stack( tuple( \
    torch.from_numpy( area_true ).to( device = DEFAULT_DEVICE, dtype = torch.float64 ) for \
    area_true in area_trues ) )

  area_preds.unsqueeze_( -1 ); area_trues.unsqueeze_( -2 )

  unions = area_trues + area_preds - intersections  # subtract intersection to remove double-countings

  #----------------------------------------------------------------------------#

  ## Exclude background from the analysis if rm_bknd == True
  if rm_bknd:
    if first_idx: intersections = intersections[:, 1:, 1:]; unions = unions[:, 1:, 1:]
    else:
      intersections = torch.cat( ( intersections[:, :, :bknd_idx], intersections[:, :, bknd_idx+1:], ) )
      intersections = torch.cat( ( intersections[:, :bknd_idx, :], intersections[:, bknd_idx+1:, :], ) )
      unions = torch.cat( ( unions[:, :, :bknd_idx], unions[:, :, bknd_idx+1:], ) )
      unions = torch.cat( ( unions[:, :bknd_idx, :], unions[:, bknd_idx+1:, :], ) )
    unions[ unions == 0 ] = _fmin

  # Calculate the Intersection over Union (IoU) for all preds labels-targs labels pairs
  iou = torch.div( intersections, unions )
  # print( iou.shape )
  # iou_sparse = coo_matrix( iou.ravel('C'), \
  #   (true_obj_coords.ravel('C'), pred_obj_coords.ravel('C')), shape = (num_tot_classes, num_tot_classes) ).tocsc()
  if sparsify:
    iou = torch.sparse.FloatTensor( iou, device = DEFAULT_DEVICE )

  # Calculate the inverse Identity Matrix (needed to compute FP and FN). Shape : (num_tot_classes, num_tot_classes)
  _inv_eye = torch.abs( \
    torch.eye( iou.shape[-2], iou.shape[-1], dtype = torch.int8 ) - 1 ) \
    .to( device = DEFAULT_DEVICE, dtype = torch.uint8 )
  _inv_eye.unsqueeze_( 0 )

  # Loop over IoU thresholds
  prec = torch.empty( targs.shape[0], len( thresholds ), dtype = torch.float64, device = DEFAULT_DEVICE )
  for n, t in enumerate( thresholds ):

    tp, fp, fn = precision_at( iou, t, _inv_eye )

    _prec_n = tp / ( tp + fp + fn )
    _prec_n[ _prec_n != _prec_n ] = 0.
    prec[ :, n ] = _prec_n

  return prec.mean()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# def sparsify( ... )

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# IoU Precision helper function
@torch.no_grad()
def precision_at( iou:torch.cuda.FloatTensor, threshold:float, inv_eye:torch.cuda.FloatTensor ):
  """Segmentation: calculates IoU-based precision for TP, FP, and FN for single input threshold."""

  matches = iou > threshold
  # true_positives = np.sum( matches, axis=1 ) == 1   # Correct objects
  true_positives = torch.diagonal( matches, dim1 = -2, dim2 = -1 ) == 1
  # false_positives = np.sum(matches, axis=0) == 0  # Missed objects

  matches = iou > 1 - threshold
  false_positives = torch.sum( matches * inv_eye , dim = -2, dtype = torch.float64 ) > 0
  # false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
  false_negatives = torch.sum( matches * inv_eye , dim = -1, dtype = torch.float64 ) > 0

  tp, fp, fn = \
    torch.sum( true_positives, dim = -1, dtype = torch.float64 ), \
    torch.sum( false_positives , dim = -1, dtype = torch.float64 ), \
    torch.sum( false_negatives, dim = -1, dtype = torch.float64 )

  return tp, fp, fn

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# CPU-parallelizable versions of functions:

@torch.no_grad()
def bincountdd_parallel_map( x:torch.cuda.IntTensor ):
  return torch.bincount( x )

@torch.no_grad()
def histogram2d_parallel_map( vec0:torch.cuda.FloatTensor, vec1:torch.cuda.IntTensor, bins:tuple ):
    """Compute numpy's histogram2d in parallel on the CPU. This is mainly because a GPU-parallel version is nowhere to
    be found."""

    return np.histogram2d( vec0, vec1, bins )[0]

@torch.no_grad()
def unique_duo_parallel( pred, targ, bknd_idx, first_idx, i ):
  # These are necessary to calculate the Intersection as well as the Sparse IoU Matrix further below.
  _true_uniques = torch.unique( targ )
  _pred_uniques = torch.unique( pred )

  _true_obj_lst = _true_uniques[1:] if first_idx else torch.cat( ( _true_uniques[:bknd_idx], _true_uniques[bknd_idx+1:] ) )
  _pred_obj_lst = _pred_uniques[1:] if first_idx else torch.cat( ( _pred_uniques[:bknd_idx], _pred_uniques[bknd_idx+1:] ) )

  return ( _true_obj_lst, _pred_obj_lst, )

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# Use this to map string input args of metric names to actual fxns of module:
# TODO: Implement the commented out metrics in this module and then uncomment

METRICS_DICT = {
  'accuracy' : accuracy,
#  'precision' : precision,
#  'recall' : recall,
#  'f1_score' : f1_score,
  'acc_segmentation' : acc_segmentation,
  'mAP_IoU' : mAP_IoU
}

METRICS_GAN_DICT = {
#  'inception' : inception,
#  'fid' : fid,
#  'ms-ssim' : diversity,
#  'discriminator loss' : disc_loss,
#  'discriminator realness' : realness
}

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
