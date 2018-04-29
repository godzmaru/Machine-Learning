#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 19:50:13 2018

@author: hendrawahyu
"""

### BINARY PARAMETERS ###
# Random Forest parameters
rf_params = {'n_jobs': -1, 'n_estimators': 500,'warm_start': True,'max_depth': 6, 'min_samples_leaf': 2,'max_features' : 'sqrt','verbose': 0 }

# Extra Trees Parameters
et_params = { 'n_jobs': -1, 'n_estimators':500, 'max_depth': 8, 'min_samples_leaf': 2, 'verbose': 0}

# AdaBoost parameters
ada_params = { 'n_estimators': 500,'learning_rate' : 0.75 }

# Gradient Boosting parameters
gb_params = { 'n_estimators': 500,'max_depth': 5,'min_samples_leaf': 2,'verbose': 0 }

# Support Vector Classifier parameters 
svc_params = {'kernel' : 'linear', 'C' : 0.025}


### REGRESSOR PARAMETERS ###
ls_params = {'alpha':0.0005}
en_params = {'alpha':0.0005, 'l1_ratio':0.9}
kr_params = {'alpha':0.6, 'kernel':'polynomial', 'degree':2, 'coef0':2.5}
gbr_params = {'n_estimators':3000, 'learning_rate':0.05, 'max_depth':4, 
              'max_features':'sqrt', 'min_samples_leaf':15, 'min_samples_split':10, 
              'loss':'huber'}
xgr_params = {'colsample_bytree':0.4603, 'gamma':0.0468, 'learning_rate':0.05, 
              'max_depth':3, 'min_child_weight':1.7817, 'n_estimators':2200,
              'reg_alpha':0.4640, 'reg_lambda':0.8571,'subsample':0.5213, 'silent':1,
              'nthread': -1}
lgbr_params = {'objective':'regression','num_leaves':5,'learning_rate':0.05, 
               'n_estimators':720, 'max_bin': 55, 'bagging_fraction': 0.8,
               'bagging_freq': 5, 'feature_fraction': 0.2319, 'feature_fraction_seed':9, 
               'bagging_seed':9, 'min_data_in_leaf':6, 'min_sum_hessian_in_leaf':11}