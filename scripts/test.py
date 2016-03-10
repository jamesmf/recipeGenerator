# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 18:52:41 2016

@author: jmf
"""

from __future__ import print_function
import numpy as np
import sys
from keras.models import model_from_json

def loadThatModel(folder):
    with open(folder+".json",'rb') as f:
        json_string     = f.read()
    model = model_from_json(json_string)
    model.load_weights(folder+".h5")
    return model
    
    
model   = loadThatModel("../models/recipeRNNcharState")