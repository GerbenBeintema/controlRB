#!/usr/bin/env python3
"""Run 'pip install -e .' to install this packege"""

from setuptools import setup, find_namespace_packages

packages = [a for a in find_namespace_packages(where='.') if a[:9]=='controlRB']

setup(name='controlRB',
      version='0.2.1',
      description = 'Controlable thermal Convection model',
      author = 'Gerben Beintema',
      author_email = 'g.i.beintema@tue.nl',
      license = 'BSD 3-Clause License',
      python_requires = '>=3.6',
      packages=packages,
      install_requires=['Pillow','matplotlib', 'tqdm', 'numpy','setGPU','scikit-image']  # And any other dependencies foo needs
     )
