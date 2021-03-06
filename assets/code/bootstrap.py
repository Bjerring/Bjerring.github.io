# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:28:07 2019

@author: ttb
"""

#%%

import pandas as pd
import numpy as np
from pandas_datareader import data
import random
from scipy.stats import truncnorm
import seaborn as sns
import matplotlib.pyplot as plt

class Bootstrapping: 
    
    def BBpath(self,dat_avail,block_size=5,n_ahead=20):
        """
        Generate block boostrapped path "n_ahead" into the future
        """

        draws = random.sample(range(0, dat_avail.shape[0]-block_size), int(np.ceil(n_ahead/block_size)) )

        path = pd.DataFrame(index=range( block_size*int(np.ceil(n_ahead/block_size)) ),columns=dat_avail.columns)
        for i in range(len(draws)):
            path.iloc[(i*block_size): ((i+1)*block_size)] = dat_avail.iloc[draws[i]:(draws[i]+block_size) ].values

        # take the n_ahead vector and save it
        scen_out = (1+path).cumprod(axis=0).iloc[-1]-1

        return scen_out.values

    def BBscenario(self,dat_avail,no_scen=1000,block_size=5,n_ahead=20,para_window=(3*253)):
        """
        Block boostrapping using a normal for-loop
        """

        # reduce data set to only include specific window of data
        dat_avail = dat_avail.iloc[(-para_window):-1 ]

        scen = pd.DataFrame(0,columns=dat_avail.columns,index=range(no_scen))

        for j in range(no_scen):
            scen.iloc[j] = self.BBpath(dat_avail)

        return(scen)


    def get_truncated_normal(self,mean=0, sd=1, low=0, upp=10):
        """
        Generate truncated normal distribution function
        """
        return(truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd))


    def BBpath_normal(self,dat_avail,block_size=5,n_ahead=20):
        """
        Generate block boostrapped path "n_ahead" into the future
        """

        mean = dat_avail.shape[0]-block_size
        std = np.std(range(mean)) #mean*sd_fraction
        random.seed(2019)

        normal = self.get_truncated_normal(mean=mean, sd=std, low=0, upp=mean)
        draws = normal.rvs( int(np.ceil(n_ahead/block_size)) ).astype(int)

        path = pd.DataFrame(index=range( block_size*int(np.ceil(n_ahead/block_size)) ),columns=dat_avail.columns,data=0.0)

        for i in range(len(draws)):
            path.iloc[(i*block_size): ((i+1)*block_size)] = dat_avail.iloc[draws[i]:(draws[i]+block_size) ].values

        scen_out = (1+path).cumprod()-1

        return(scen_out.iloc[n_ahead-1])


    def BBscenario_sequential(self,dat_avail,no_scen=1000,block_size=5,n_ahead=20,para_window=(3*253)):
        """
        Block boostrapping using a normal for-loop
        """

        # reduce data set to only include specific window of data
        dat_avail = dat_avail.iloc[(-para_window):-1 ]

        scen = pd.DataFrame(columns=dat_avail.columns)

        for j in range(no_scen):
            scen = scen.append(self.BBpath_normal(dat_avail), ignore_index=True)

        return(scen)
        
#%%
# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2015-01-01'
end_date = '2018-12-31'

# tickers
tickers = ["SPY","EEM"]

# User pandas_reader.data.DataReader to load the desired data.
panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)
df_close = panel_data["Adj Close"]

# Daily returns, and remove the first row as it is NA
df_ret = df_close.pct_change().iloc[1:]

# ini boot
Boot = Bootstrapping()

# Block bootstrapping
BBscen = Boot.BBscenario(dat_avail=df_ret)

# block bootstrapping using truncated normal
BBtail = Boot.BBscenario_sequential(dat_avail=df_ret)


# plot truncated normal distribution
n_ahead = 20
block_size = 5
mean = df_ret.shape[0]-block_size
sd = np.std(range(mean))
low = 0
upp = mean
random.seed(2019)

# plot truncated normal distribution
tcn = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
t_bars = tcn.rvs( 10000 ).astype(int)
plt.hist(t_bars,bins=30)


## plot historical monthly returns
sns.jointplot(x="EEM", y="SPY", data=df_close.resample('M').last().pct_change().iloc[1:] )

## plot block bootstrapped scenarios
sns.jointplot(x="EEM", y="SPY", data=BBscen)

## plot truncated block bootstrapped scenarios
sns.jointplot(x="EEM", y="SPY", data=BBtail)



#%%
        
        
        
        
    