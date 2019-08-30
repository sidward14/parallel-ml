# -*- coding: UTF-8 -*-

"""General functions and methods for CPU-based parallel computing
"""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import concurrent.futures

# import torch
import numpy as np
import pdb

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def parallel_map_func_re( func, arrs:np.ndarray, max_workers:int = 2 ):
  "Call `func` on `arr` for each index in `dim` in parallel using `max_workers`."
  #imax_workers = ifnone(max_workers, defaults.cpus)
  if max_workers < 2: pout = map( func, *arrs )
  else:
  #print(arr)
    with concurrent.futures.ProcessPoolExecutor( max_workers = max_workers ) as ex:
      #futures = [ex.submit(func,o,i) for i,o in enumerate(arr)]
      pout = ex.map( func, *arrs )
      #results = []
  return pout
  #if any([o is not None for o in results]): return results

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# if __name__ == '__main__':
  # pass

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
