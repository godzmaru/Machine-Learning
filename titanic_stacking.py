#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 13:24:00 2018

@author: hendrawahyu
"""

# data analysis and wrangling
import numpy as np
import pandas as pd
import re

# machine learning
from classifier import Classifier
import params as pr
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,\
                            GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
import xgboost as xgb

### WRANGLING ###
def data_wrangling(debug = False):
    global df
    from sklearn.preprocessing import LabelEncoder
    
    ### Title ###
    # extract title from the name 
    df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
    #df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand = False)
    
    # replace specific titles
    df.loc[df['Title'].isin(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']), 'Title'] = 'Rare'
    df.loc[df['Title'].isin(['Mlle', 'Ms']), 'Title'] = 'Miss'
    df.loc[df['Title'] == 'Mme', 'Title'] = 'Mrs'
    
    ### Port Embarkation ###
    freq_port = df.Embarked.dropna().mode()[0]
    df.Embarked = df.Embarked.fillna(freq_port)
    
    ### Fare ###
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FareBand'] = pd.qcut(df['Fare'], 4)
        
    ### Age ###
    age_avg = df['Age'].mean()
    age_std = df['Age'].std()
    age_null = df['Age'].isnull().sum()
    age_random = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null)
    df.loc[np.isnan(df['Age']),'Age'] = age_random
    df['AgeBand'] = pd.cut(df['Age'], 5)
    
    ### Family Size ###
    # combine SibSp and Parch Features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # create new feature isAlone, instantiate to 0 and fillin with 1 if no family
    df['isAlone'] = 0
    df.loc[df['FamilySize'] == 1,'isAlone'] = 1
    
    ### Cabin ###
    df['Cabin'] = df['Cabin'].fillna('X')
    df['CabinBand'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").findall(x)[0])
    
    ### Dropping data ###
    cols = ['PassengerId','Name', 'Fare', 'Age', 'FamilySize', 'SibSp', 'Parch', 'Cabin', 'Ticket']
    for c in cols:
        df.drop(c, axis=1, inplace=True)

    ### Label Encoder ###
    cols1 = ['Title', 'Sex', 'Embarked', 'AgeBand', 'FareBand', 'CabinBand']
    for c in cols1:
        lbl = LabelEncoder() 
        lbl.fit(list(df[c].values)) 
        df[c] = lbl.transform(list(df[c].values))
    
    if(debug == True):
        df.info()
        print('-'* 40)
        print(df.head(10))


### OUT-OF-FOLD PREDICTIONS ###
def get_oof(model, X, y, x_test, nfolds = 5, seed = 0):
    from sklearn.cross_validation import KFold
    
    oof_train = np.zeros((X.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((nfolds, x_test.shape[0]))
    
    kf = KFold(X.shape[0], n_folds= nfolds, shuffle = False, random_state=seed)
    for i, (train_index, test_index) in enumerate(kf):
        model.train(X[train_index], y[train_index])
        oof_train[test_index] = model.predict(X[test_index])
        oof_test_skf[i, :] = model.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


### MAIN ###
if __name__ == '__main__':
    # opening and reading from csvfile
    train_df = pd.read_csv('../Machine Learning/titanic/train.csv')
    test_df = pd.read_csv('../Machine Learning/titanic/test.csv')
    
    n_train = train_df.shape[0]
    n_test = test_df.shape[0]
    
    # merge dataframes
    df = pd.concat([train_df, test_df]).reset_index(drop = True)
    
    ### DATA WRANGLING ###
    data_wrangling()

    ### MODEL ###
    train = df[:n_train]
    train.Survived = train.Survived.astype(int)
    test = df[n_train:]
    y_train = train.Survived.values
    
    train.drop('Survived', axis=1, inplace=True)
    test.drop('Survived', axis=1, inplace=True)
    x_train = train.values
    x_test = test.values
    SEED = 0
    
    # Create 5 objects that represent our 4 models
    rf = Classifier(RandomForestClassifier, seed=SEED, params=pr.rf_params)
    et = Classifier(ExtraTreesClassifier, seed=SEED, params= pr.et_params)
    ada = Classifier(AdaBoostClassifier, seed=SEED, params= pr.ada_params)
    gb = Classifier(GradientBoostingClassifier, seed=SEED, params= pr.gb_params)
    svc = Classifier(SVC, seed=SEED, params= pr.svc_params)
    
    # Create our OOF train and test predictions. These base results will be used as new features
    et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)        # Extra Trees
    rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test)         # Random Forest
    ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)     # AdaBoost 
    gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test)         # Gradient Boost
    svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test)      # Support Vector Classifier
    
    ### SECOND LEVEL PREDICTIONS ###
    x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
    x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

    gbm = xgb.XGBClassifier( n_estimators= 2000, max_depth= 4, min_child_weight= 2,
                            gamma=0.9, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',
                            nthread= -1, scale_pos_weight=1).fit(x_train, y_train)
    predictions = gbm.predict(x_test)
    
    ### SUBMISSION ###
    submission = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions })
    submission.to_csv('../Machine Learning/submission/output_stack.csv', index=False)