#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:55:18 2019

@author: MichaelYANG
"""


nasdaq index
Vanguard Total Bond Market Index Fund Investor Shares (VBMFX)
Invesco DB Commodity Index Tracking Fund (DBC)

import pandas as pd
import pandas_datareader.data as web
import numpy as np
import datetime
from scipy.optimize import minimize
TOLERANCE = 1e-10


#Use pands_datareader to download Nasdaq index data
import pandas_datareader.data as web
#set time
start = datetime.datetime(2008,12,31)
end = datetime.datetime(2018,12,31)
#download the data
df = web.get_data_yahoo(['DBC','^IXIC','VBMFX'],start, end,interval='m').loc[:, 'Adj Close']

#Define parameters 
covariances= 12 *df.pct_change().iloc[2:,:].cov().values
 # Initial weights: equally weighted
initial_weights = [1 / df.shape[1]] * df.shape[1]#此处用[]是矩阵乘法，用（）是数值
 # The desired contribution of each asset to the portfolio risk: we want all asset to contribute equally
assets_risk_budget = [1 / df.shape[1]] * df.shape[1]



#Define optimization function
def _risk_budget_objective_error(weights, args):

    # The covariance matrix occupies the first position in the variable
    covariances = args[0]

    # The desired contribution of each asset to the portfolio risk occupies the
    # second position
    assets_risk_budget = args[1]

    # We convert the weights to a matrix
    weights = np.matrix(weights)

    # We calculate the risk of the weights distribution
    portfolio_risk = np.sqrt((weights * covariances * weights.T))

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = np.multiply(weights.T, covariances * weights.T)/portfolio_risk

    # We calculate the desired contribution of each asset to the risk of the
    # weights distribution
    assets_risk_target = np.multiply(portfolio_risk, assets_risk_budget)

    # Error between the desired contribution and the calculated contribution of
    # each asset
    error = sum(np.square(assets_risk_contribution - assets_risk_target.T))

    # It returns the calculated error
    return error


# get the risk parity weight
def get_RP_weights():

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})


    TOLERANCE = 1e-10
    optimize_result = minimize(fun=_risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances, assets_risk_budget],
                               method='SLSQP',
                               constraints=constraints,
                               tol=TOLERANCE,
                               options={'disp': False})
    weights = optimize_result.x
    return weights

# Convert the weights to a pandas Series
RP_weights = pd.Series(get_RP_weights(),index = df.columns, name = 'weight')


##find MV weights

def MV_fun(weights, args):
    covariance = args
    weights = np.matrix(weights)
    portfolio_variance = weights * covariance * weights.T
    return portfolio_variance

def get_MV_weights():
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})
 
    optimize_result = minimize(fun = MV_fun,
                               x0=initial_weights,
                               args = covariances,
                               method='SLSQP',
                               constraints=constraints,
                               tol=TOLERANCE,
                               options={'disp': False})
    weights = optimize_result.x
    return weights

MV_weights =  pd.Series(get_MV_weights(),index = df.columns, name = 'weight')

EQ_weithts =  pd.Series(initial_weights,index = df.columns, name = 'weight')


#plot the return

df.pct_change().plot(grid = True)





===================================================

## Use quandl to get stock price

import quandl
import datetime
 
# We will look at stock prices over the past year, starting at January 1, 2016
start = datetime.datetime(2018,1,1)
end = datetime.datetime(2018,2,1)
 
# Let's get Apple stock data; Apple's ticker symbol is AAPL
# First argument is the series we want, second is the source ("yahoo" for Yahoo! Finance), third is the start date, fourth is the end date
apple = quandl.get("NASDAQOMX/COMP", start_date = start, end_date = end )
 
type(apple)
apple.head()


#Populating the interactive namespace from numpy and matplotlib
import matplotlib.pyplot as plt   # Import matplotlib
# This line is necessary for the plot to appear in a Jupyter notebook
%matplotlib inline
# Control the default size of figures in this Jupyter notebook
%pylab inline
pylab.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots
df["Adj Close"].plot(grid = True) # Plot the adjusted closing price of AAPL
