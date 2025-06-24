# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:15:20 2022

@author: gita
"""
import os 
import sys
default_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(default_path)
sys.path.insert(0, default_path+'/src/graph')

from src.graph.GUI import execute_GUI

if __name__ == '__main__':
    execute_GUI() 