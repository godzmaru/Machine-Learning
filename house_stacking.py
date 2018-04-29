#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 11:55:14 2018

@author: hendrawahyu
"""

import pandas as pd
import numpy as np
from plot import scatter, normplot, heatcorr
from analysis import check_numerical_skew, apply_boxcox

# machine learning
from classifier import Classifier, StackingAveragedRegressor, StackingRetrainedRegressor
import params as pr
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


#Limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) 


### WRANGLING ###
def data_wrangling():
    global df
    from sklearn.preprocessing import LabelEncoder
    
    # fill all categorical features / missing data with None
    cols = ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", 
        "GarageFinish", "GarageQual", "GarageCond", 'BsmtQual', 'BsmtCond', 
        "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType", "MSSubClass")
    for c in cols:
        df[c] = df[c].fillna('None')
     
    # fill all missing numerical value with 0
    cols = ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 
         'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea')
    for col in cols:
        df[col] = df[col].fillna(0)
        
    # fill missing value by median
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median())) 
    
    # Adding total sqfootage feature 
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # fill missing value with the most frequent values
    cols = ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd',
         'SaleType', 'Functional')
    for col in cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # since we treat some index with string None, we apply string to the rest
    cols = ('OverallCond', 'YrSold', 'MoSold', 'MSSubClass')
    for c in cols:
        df[c] = df[c].astype(str)
    
    ### drop ###
    df = df.drop(['Utilities'], axis=1)
    
    ### Label Encoder ###
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
    for c in cols:
        lbl = LabelEncoder() 
        lbl.fit(list(df[c].values)) 
        df[c] = lbl.transform(list(df[c].values))

### SCORING ###
def rmsle_cv(model, n_folds = 5):
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    
    kf = KFold(n_splits = n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))   
    

### MAIN ###
if __name__ == '__main__':
    train_df = pd.read_csv('../Machine Learning/house/train_house.csv')
    test_df = pd.read_csv('../Machine Learning/house/test_house.csv')
    n_train = train_df.shape[0]
    y_train = train_df.SalePrice.values
    
    # According to Ames Housing Doc, there are outliers in training data
    #scatter(train_df, 'GrLivArea', 'SalePrice')
    train_df = train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)].index)
    
    # merge 2 dataframe together and drop Id features
    df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)
    df.drop('Id', axis =1, inplace = True)
    
    # By analysing normally distributed data, we can see its skew
    #normplot(train_df, 'SalePrice')    # uncomment to check normalized plot
    
    # since it is right skew (indicated by black solid line)
    train_df.SalePrice = np.log1p(train_df.SalePrice)
    #normplot(train_df, 'SalePrice')    # uncomment to check result
    
    # to check data correlation
    corr = train_df.corr()
    #heatcorr(corr)                     # uncomment to check the correlation
    
    # wrangling #
    data_wrangling()
    
    # check numerical skewness
    skew = check_numerical_skew(df)
    apply_boxcox(df, skew)
    
    # separate all categorical features and separate df
    df = pd.get_dummies(df)
    train = df[:n_train]
    test = df[n_train:]
    
    ### MODEL ###
    lasso = make_pipeline(RobustScaler(), Classifier(Lasso, seed = 1, params = pr.ls_params))
    ENet = make_pipeline(RobustScaler(), Classifier(ElasticNet, seed = 3, params = pr.en_params))
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    GBoost = Classifier(GradientBoostingRegressor, seed = 5, params = pr.gbr_params)
    model_xgb = Classifier(xgb.XGBRegressor, seed = 7, params = pr.xgr_params)
    model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
    
    ### OUT-OF-FOLD PREDICTIONS ###    
    # stacked average
    stacked_averaged_models = StackingAveragedRegressor(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)
    
    # stacked retrained
    stacked_retrained_models = StackingRetrainedRegressor(base_models = (ENet, GBoost, KRR),
                                                 meta_models = lasso, use_feat_secondary = True)
    
    
    ### ENSEMBLING ###
    stacked_averaged_models.fit(train.values, y_train)
    stacked_train_pred = stacked_averaged_models.predict(train.values)
    stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
    print(rmsle(y_train, stacked_train_pred))
    
    ### RETRAINED ###
    stacked_retrained_models.fit(train.values, y_train)
    stacked_re_train_pred = stacked_retrained_models.predict(train.values)
    stacked_re_pred = np.expm1(stacked_retrained_models.predict(test.values))
    print(rmsle(y_train, stacked_re_train_pred))
    
    model_xgb.fit(train, y_train)
    xgb_train_pred = model_xgb.predict(train)
    xgb_pred = np.expm1(model_xgb.predict(test))
    print(rmsle(y_train, xgb_train_pred))
    
    model_lgb.fit(train, y_train)
    lgb_train_pred = model_lgb.predict(train)
    lgb_pred = np.expm1(model_lgb.predict(test.values))
    print(rmsle(y_train, lgb_train_pred))
    
    print('RMSLE score on train data:')
    print(rmsle(y_train,stacked_train_pred*0.70 +
                   xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
    
    ensemble_avg = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
    ensemble_re = stacked_re_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
    
    ### SUBMISSION ###
    submission_avg = pd.DataFrame({'Id': test_df.Id, 'Survived': ensemble_avg })
    submission_re = pd.DataFrame({'Id': test_df.Id, 'Survived': ensemble_re })
    submission_avg.to_csv('../Machine Learning/submission/output_house_avg.csv', index=False)
    submission_re.to_csv('../Machine Learning/submission/output_house_re.csv', index=False)