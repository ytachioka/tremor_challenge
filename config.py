#!/usr/bin/env python

import os

realpath = r'data/TrainingDataPD25/real'
virtpath = r'data/TrainingDataPD25/virtual'
rootdir = r'.'  # replace with your project path
real_directory = os.path.join(rootdir,realpath)
real_directory_test = 'data/TestData/real'
virt_directory = os.path.join(rootdir,virtpath)

## 変更される
splits = [0.9, 0.1]
#seed_value = 2025


