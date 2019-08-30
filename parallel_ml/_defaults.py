# -*- coding: UTF-8 -*-

"""Package-wide default variables based on user's hardware capabilities.
"""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import torch

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

DEFAULT_DEVICE = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

# torch.autograd.set_grad_enabled( mode = False )
#
# torch.set_default_tensor_type( t = torch.cuda.FloatTensor )
#
# #------------------------------------------------------------------------------#
#
# def return_to_pytorch_defaults():
#   torch.autograd.set_grad_enabled( mode = True )
#   torch.set_default_tensor_type( t = torch.FloatTensor )
#
# import atexit
# atexit.register( return_to_pytorch_defaults )

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
