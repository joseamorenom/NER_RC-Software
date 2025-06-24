# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:56:09 2022

@author: Santiago Moreno
"""
import os 
import argparse
from functions import json_to_txt, training_model, characterize_data, upsampling_data, str2bool, usage_cuda,copy_data
default_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(default_path)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, usage='Train a new model with given data (GPU optional)')
    parser.add_argument('-f','--fast', type=str2bool, nargs='?',const=True, default=False, help='Training fast option (Only for functioning test)', choices=(True, False), required=False)
    parser.add_argument('-m','--model', type=str, nargs='?', help='New model name', required=True)
    parser.add_argument('-s','--standard', type=str2bool, nargs='?',const=True, default=False, help='Standard CONLL input or not', choices=(True, False), required=False)
    parser.add_argument('-id','--input_dir', type=str, nargs='?', help='Absolute path input directory', required=True)
    parser.add_argument('-u','--up_sample_flag', type=str2bool, nargs='?',const=True, default=False , help='Boolean value to upsampling the data = True or not upsampling = False', required=False, choices=(True, False))
    parser.add_argument('-cu','--cuda', type=str2bool, nargs='?', const=True, default=False, help='Boolean value for using cuda to Train the model (True). By defaul False.', choices=(True, False), required=False)

    args = parser.parse_args()
    

    if args.fast: epochs = 1
    else: epochs = 20
    
    if args.standard:
        copy_data(args.input_dir)
        not_error=True
    else:
        Error = json_to_txt(args.input_dir)
        if type(Error)==int:
            print('Error processing the input documents, code error {}'.format(Error))
            not_error=False
        else:
            not_error=True

    if not_error:
        if args.up_sample_flag:
            entities_dict=characterize_data()
            entities = list(entities_dict.keys())
            entities_to_upsample = [entities[i] for i,value in enumerate(entities_dict.values()) if value < 200]
            upsampling_data(entities_to_upsample, 0.8,  entities)
            
        if args.cuda: cuda_info = usage_cuda(True)
        else: cuda_info = usage_cuda(False)
        
        print(cuda_info)
        
        Error = training_model(args.model,epochs)
        if type(Error)==int:
            print('Error training the model, code error {}'.format(Error))
        else: 
            print('Training complete')
