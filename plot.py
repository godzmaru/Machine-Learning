#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:28:48 2018

@author: hendrawahyu
"""
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import norm #for some statistics

color = sns.color_palette()
sns.set_style('darkgrid')

### VISUALIZATION NUMERICAL ###
def scatter(data, x_plot, y_plot):
    fig, ax = plt.subplots()
    ax.scatter(x = data[x_plot], y = data[y_plot])
    plt.ylabel(x_plot, fontsize = 13)
    plt.xlabel(y_plot, fontsize = 13)
    plt.show()

def normplot(data, a, QQPlot= True, debug = False):
    sns.distplot(data[a], fit = norm)
    (mu, sigma) = norm.fit(data[a])
    if(debug == True):
        print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
    plt.ylabel('Frequency')
    if(QQPlot == True):
        fig, ax = plt.subplots()
        stats.probplot(data[a], plot = plt)
        plt.show()

def heatcorr(corr_data):
    plt.subplots(figsize = (12, 9))
    sns.heatmap(corr_data, vmax = 0.9, square = True)
    