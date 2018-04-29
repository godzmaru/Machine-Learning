#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 10:23:13 2018

@author: hendrawahyu
"""

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split

import numpy as np

### Classifier Models ###
class Classifier(object):
    def __init__(self, models, seed = 0, params = None):
        params['random_state'] = seed
        self.models = models(**params)
        
    def train(self, X, y):
        self.models.fit(X, y)
        
    def predict(self, X):
        return self.models.predict(X)
    
    def fit(self,X,y):
        return self.models.fit(X,y)
    
    def feature_importances(self,X,y):
        print(self.models.fit(X,y).feature_importances_)


### Stacking Regressor models ###
class StackingAveragedRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


class StackingRetrainedRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_models, n_folds = 5, use_feat_secondary = False):
        self.base_models = base_models
        self.meta_models = meta_models
        self.n_folds = n_folds
        self.use_feat_secondary = use_feat_secondary
    
    def fit(self, X, y):
        self.base_models_ = [clone(x) for x in self.base_models]
        self.meta_models_ = clone(self.meta_models)
        kfold = KFold(n_splits = self.n_folds, shuffle = True)
        
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models_):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(model)
                instance.fit(X[train_idx], y[train_idx])
                out_of_fold_predictions[holdout_idx, i] = instance.predict(X[holdout_idx])
        
        # train meta-model
        if self.use_feat_secondary:
            self.meta_models_.fit(np.hstack((X, out_of_fold_predictions)), y)
        else:
            self.meta_models_.fit(out_of_fold_predictions, y)
            
        # retrain base models on all data
        for regr in self.base_models_:
            regr.fit(X, y)
        
        return self
    
    def predict(self, X):
        meta_features = np.column_stack([ regr.predict(X) for regr in self.base_models_])
        if self.use_feat_secondary:
            return self.meta_models_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_models_.predict(meta_features)
  

        
        