#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:24:58 2018

@author: hendrawahyu
"""
import pandas as pd
import numpy as np

### ANALYSIS ###
def check_numerical_skew(data, debug = False):
    from scipy.stats import skew
    
    # take all numerical values index
    numeric_feat = data.dtypes[data.dtypes != 'object'].index
    
    # check skew
    skewed_feat = data[numeric_feat].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feat})
    if(debug == True):
        print('\nSkew in Numerical Features:\n')
        print(skewness.head(20))
    return skewness

def apply_boxcox(data, skew, lam = 0.15, debug = False):
    from scipy.special import boxcox1p
    
    skew = skew[abs(skew) > 0.75]
    if(debug == True):
        print("There are {} skewed numerical features to Box Cox transform".format(skew.shape[0]))
    skewed_features = skew.index
    for feat in skewed_features:
        data[feat] = boxcox1p(data[feat], lam)
    