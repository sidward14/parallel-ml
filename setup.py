# -*- coding: UTF-8 -*-

import os.path
from setuptools import setup

def readme( ):

  with open( os.path.abspath(
    os.path.join(
      os.path.dirname( __file__ ),
      'README.rst' ) ) ) as f:

    return f.read( )

setup(
  name = 'parallel-ml',
  version = '0.0.2',
  author = 'Sidhartha Parhi',
  author_email = 'sidhartha.parhi@gmail.com',
  description = "Parallel ML utilities (GPU or CPU) that I can't find anywhere else",
  long_description = readme( ),
  url = "https://github.com/sidward14/parallel-ml",
  packages = [
    'parallel_ml',
  ],
  dependency_links = [ ],
  install_requires = [
    'numpy >= 1.16',
    #'torch >= 1.1.0',
    'pandas >= 0.24.2',
    'matplotlib >= 3.1.0', ],
  include_package_data = True,
  zip_safe = False
)
