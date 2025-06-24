# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:06:56 2022

@author: gita
"""

from distutils.core import setup
import py2exe

setup(
    options={"py2exe": {"bundle_files": 1}},
    console=[{
        "script": "execute_GUI.py"
    }]
)