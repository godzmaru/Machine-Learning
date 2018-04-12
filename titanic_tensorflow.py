#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 22:01:36 2018

tensorflow - basic

@author: hendrawahyu
"""

import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split

def process_title(debug = False):
    global df
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    
    # extract title from the name 
    df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

    # replace specific titles
    df.loc[df['Title'].isin(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']), 'Title'] = 'Rare'
    df.loc[df['Title'].isin(['Mlle', 'Ms']), 'Title'] = 'Miss'
    df.loc[df['Title'] == 'Mme', 'Title'] = 'Mrs'
    
    # map to dict of title_mapping
    df['Title'] = df['Title'].map(title_mapping)
    # fill all N/A values with 0
    df['Title'] = df.Title.fillna(0)
    df.Title = df.Title.astype(int)
    
    #dropping name
    df.drop(['Name'], axis = 1, inplace = True)
    
    if(debug == True):
        df.info()
    

def process_sex():
    global df
    df.Sex = df.Sex.map({'male':1, 'female': 0}).astype(int)
    #df.Sex = np.where(df['Sex'] == 'male', 1, 0)
   
    
def process_embarkation():
    global df
    # check frequent port from 'Embarked'
    freq_port = df.Embarked.dropna().mode()[0]
    df.Embarked = df.Embarked.fillna(freq_port)
    df.Embarked = df.Embarked.map({'S': 0, 'C': 1, 'Q':2}).astype(int)


def process_fare():
    global df
    # fill 'Fare' single missing values in test_df with median 
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # convert Fare into numeric fare band
    df.loc[ df['Fare'] <= 7.91, 'Fare']= 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare']= 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']= 2
    df.loc[ df['Fare'] > 31, 'Fare']= 3
    df['Fare'] = df['Fare'].astype(int)


def process_family(debug = False):
    global df
    
    # 1 signifies 1 person (no sibling spouse + no parent children)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['isAlone'] = 0
    
    # if nth-row of 'Family Size' == 1, put 1 to 'isAlone'
    df.loc[df['FamilySize'] == 1,'isAlone'] = 1
    df.drop(['FamilySize'], axis=1, inplace = True)

   
def process_age(choice = 1):
    '''
    choice:
        1. random numbers between (mean - std) and (mean + std)
        2. random forest regressor 
        3. random number median
    '''
    global df
    from sklearn.ensemble import RandomForestRegressor

    if(choice == 1):
        age_avg = df['Age'].mean()
        age_std = df['Age'].std()
        age_null = df['Age'].isnull().sum()
        age_random = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null)
        
        df['Age'][np.isnan(df['Age'])] = age_random
    
    elif(choice ==2):
        # rev 0.0
        age_df = df[['Age', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']]
        # instantiate X and y for training model to predict Age
        X = age_df.loc[(df.Age.notnull())].values[:, 1::]
        y = age_df.loc[(df.Age.notnull())].values[:,0]
    
        # A random forest is a meta estimator that fits a number of classifying 
        # decision trees on various sub-samples of the dataset and use averaging 
        # to improve the predictive accuracy and control over-fitting.
        rf = RandomForestRegressor(n_estimators = 2000, n_jobs = -1)
        rf.fit(X, y)
        predict_age = rf.predict(age_df.loc[ df.Age.isnull()].values[:,1::])
        df.loc[(df.Age.notnull()), 'Age'] = predict_age
    
    elif(choice == 3):
        guess_ages = np.zeros((2,3))
        for i in range(0, 2):      
            for j in range(0, 3):  
                guess_df = df.loc[(df['Sex'] == i) & (df['Pclass'] == j+1), 'Age'].dropna()    
                age_guess = guess_df.median()
                # Convert random age float to nearest .5 age
                guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
                df.loc[ (df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1), 'Age'] = guess_ages[i,j]
    
    # Mapping Age
    df.loc[ df['Age'] <= 16, 'Age']= 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age']= 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age']= 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age']= 3
    df.loc[ df['Age'] > 64, 'Age']= 4
    
    df['Age'] = df['Age'].astype(int)


def process_cabin(debug = False):
    global df
    import seaborn as sns
    '''
    passengers with a cabin have generally more chance to survive
    set to True to see this
    '''
    if(debug == True):
        df.loc[ (df.Cabin.isnull()), 'Cabin'] = 'X'
        g = sns.factorplot(y = 'Survived', x = 'Cabin', data = df, kind = 'bar', \
                           order = ['A', 'B', 'C', 'D', 'E', 'F' , 'G', 'T', 'X'])
        g = g.set_ylabels('Survival Probability')
    else:
        df.loc[df.Cabin.isnull(), 'Cabin'] = 'X'
        # create feature for the alphabetical part of the cabin number
        df['Cabin'] = df['Cabin'].map( lambda x : getCabinLetter(x))
        df['Cabin'] = pd.factorize(df['Cabin'])[0]
        df['Cabin'] = df['Cabin'].astype(int) 
        
def getCabinLetter(cabin):
    """
    Find the letter component of the Cabin variable
    """
    match = re.compile("([a-zA-Z]+)").search(cabin)
    if match:
        return match.group()
    else:
        return 'X'


def process_ticket():
    '''
    Ticket with same prefixes could be booked for cabins placed together and may
    have a similar class and survival
    '''
    global df
    
    df['TicketPrefix'] = df['Ticket'].map( lambda x : getTicketPrefix(x.upper()))
    df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('[\.?\/?]', '', x) )
    df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('STON', 'SOTON', x) )
    df['TicketPrefixId'] = pd.factorize(df['TicketPrefix'])[0]
    df.drop(['Ticket', 'TicketPrefix'], axis = 1, inplace = True)
    df['TicketPrefixId'] = df['TicketPrefixId'].astype(int)

def getTicketPrefix(ticket):
    """
    Find the letter component of the ticket variable)
    """
    match = re.compile("([a-zA-Z\.\/]+)").search(ticket)
    if match:
        return match.group()
    else:
        return 'X'  

def drop_correlation():
    global df
    
    # calculate the correlation matrix
    df_corr = df.drop(['Survived', 'PassengerId'],axis=1).corr(method='spearman')
    
    # create a mask to ignore self-
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr
    
    drops = []
    # loop through each variable
    for col in df_corr.columns.values:
        # if we've already determined to drop the current variable, continue
        if np.in1d([col],drops):
            continue
        
        # find all the variables that are highly correlated with the current variable 
        # and add them to the drop list 
        corr = df_corr[abs(df_corr[col]) > 0.98].index
        #print col, "highly correlated with:", corr
        drops = np.union1d(drops, corr)
        
        print('\ndropping highly correlated features:\n')
        df.drop(drops, axis=1, inplace=True)
        

# main
if __name__ == '__main__':
    EPOCHS = 100
    
    # opening and reading from csvfile
    train_df = pd.read_csv('../Machine Learning/titanic/train.csv')
    test_df = pd.read_csv('../Machine Learning/titanic/test.csv')
    
    # add column 'Deceased' (* Tensorflow y)
    train_df['Deceased'] = train_df['Survived'].apply(lambda s: 1 - s)

    # merge dataframes
    df = pd.concat([train_df, test_df])
    # re number the df index
    df.reset_index(inplace = True)
    # drop new column (axis = 1) called 'index'
    df.drop('index', axis = 1, inplace = True)
    # re index the starting column from 1 to 0
    df = df.reindex_axis(train_df.columns, axis = 1)
    
    process_cabin()
    process_ticket()
    process_title()
    process_sex()
    process_embarkation()
    process_fare()
    process_age(choice = 3)
    
    # Move the 'Deceased' column back to the first position
    columns_list = list(df.columns.values)
    columns_list.remove('Deceased')
    new_col_list = list(['Deceased'])
    new_col_list.extend(columns_list)
    df = df.reindex(columns=new_col_list)
    
    # drop 'PassengerId'
    df.drop(['PassengerId'], axis=1, inplace = True)
    
    # separating train and test dataframe and converting into numpy array
    train = df[:train_df.shape[0]]
    train1 = train.values
    
    test = df[train_df.shape[0]:]
    test.drop(['Survived', 'Deceased'], axis = 1, inplace = True)
    #test = test.values
    
    Xd = train1[0::, 2::]
    yd = train1[0::, 0:2]
    
    X_train, X_test, y_train, y_test = train_test_split(Xd, yd,test_size=0.25,random_state=None)

    #TENSORFLOW
    # create symbolic variables
    X = tf.placeholder(tf.float32, shape=[None, 10])
    y = tf.placeholder(tf.float32, shape=[None, 2])
    # weights and bias are the variables to be trained
    W = tf.Variable(tf.random_normal([10, 2]), name='weights')
    b = tf.Variable(tf.zeros([2]), name='bias')
    y_pred = tf.nn.softmax(tf.matmul(X, W) + b)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred + 1e-10), reduction_indices=[1]))
    # use gradient descent optimizer to minimize cost
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #training loop
        for epoch in range(EPOCHS):
            total_loss = 0
            for i in range(len(X_train)):
                feed_dict = {X: [X_train[i]], y: [y_train[i]]}
                _, loss = sess.run([train_step, cross_entropy], feed_dict=feed_dict)
                total_loss += loss
            #print('Epoch: %04d, total loss: %.9f' % (epoch + 1, total_loss))
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_pred,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy', sess.run(accuracy, feed_dict={X: X_test , y: y_test }))
        
        # Predict using the model
        predictions = np.argmax(sess.run(y_pred, feed_dict={X: test}), 1)
        
        submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": predictions})
        submission.to_csv("../Machine Learning/output_tf.csv", index=False)
