# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:25:15 2017

@author: jn107154
"""

import pickle
import pandas as pd
import numpy as np
from Selective import SelectiveNB

with open("L:/APD/BayesNets/traindf-devdata.pickle", "rb") as myfile:
    traindf = pickle.load(myfile)

n = traindf.shape[0]
ind = np.random.rand(n) < 0.75
trainpima = traindf.loc[ind]
testpima = traindf.loc[~ind]

class_col_name = "ACSMedicalExposure"
#mytest = SelectiveNB(trainpima, class_col_name, xclass = 1, init_col='Age')
SNB =  SelectiveNB(trainpima, class_col_name, xclass = 'Y', init_col="EXPOSURE_NUMBER")
nbmodel = SNB.Build(traindf)


