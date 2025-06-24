# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 00:32:02 2022

@author: Santiago Moreno
"""
import os 
import argparse
from functions import use_model, str2bool, usage_cuda

default_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(default_path)
output_dir = "../../data/tagged/document_tagged.json"


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, usage='Tag a document with a pre-trained model (GPU optional)')
    parser.add_argument('-m','--model', default='CCC', type=str, nargs='?', help='New model name', required=True)
    parser.add_argument('-id','--input_data', type=str, nargs='?', help='Absolute path input file', required=True)
    parser.add_argument('-od','--output_data', const=output_dir, default=output_dir, type=str, nargs='?', help='Absolute path output file', required=False)
    parser.add_argument('-cu','--cuda', type=str2bool, nargs='?', const=True, default=False, help='Boolean value for using cuda to Train the model (True). By defaul False.', choices=(True, False), required=False)
    args = parser.parse_args()
    
    #print(args.model, args.input_data, args.output_data)
    if args.cuda: cuda_info = usage_cuda(True)
    else: cuda_info = usage_cuda(False)
    print(cuda_info)
    Error = use_model(args.model, args.input_data, args.output_data)
    if type(Error)==int:
        print('Tagged not complete, error code {}'.format(Error))
    else:
        print('Tagged complete')

# path_data = "C:/Users/gita/OneDrive - Universidad de Antioquia/GITA/Maestría/Programas/Datasets/camara_comercio_NER/gt/3cb4fa20-89cb-11e8-a485-d149999fe64b-0.json "
# output_dir = "C:/Users/gita/OneDrive - Universidad de Antioquia/GITA/Maestría/Programas/Software NER/document_tagged.json"
# sentence = use_model('CCC', path_data, output_dir)
