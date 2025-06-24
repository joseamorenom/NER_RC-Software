# -*- coding: utf-8 -*-
"""
Created on Thu May  4 18:47:46 2023

@author: sanmo
"""
import os
default_path = os.path.dirname(os.path.abspath(__file__))
default_path = default_path.replace('\\', '/')

from functionsrc import training_model_rc, usage_cuda_rc, use_model_rc

path_data = default_path + '/../../data/RC/test.txt'
rel2id_data = default_path + '/../../data/RC/rel2id.json'
print(usage_cuda_rc(True))
training_model_rc('p', path_data, rel2id_data, 2)

# output_dir = default_path + '/../../out_RC.json'

# print(use_model_rc('new', path_data, output_dir))
