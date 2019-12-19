# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:12:03 2019

@author: ttb
"""

#%%
import pandas as pd
import numpy as np
from pandas_datareader import data
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis


#%% functions

class MomentMatch:
    
    def MomentMatchingFunc(self,x,Para_match):
        """ Moment maching function. Create a discrete approximitation of a finite set of statistical moments using a defined number of scenarios
        
        Parameters
        ----------
        x : dict
            Scenarios (variables)
        Para_match : numpy array
            statistical moments to be matched    
        
        Returns
        -------
        float
            returns the error of the approximation
       
        References
        ----------
            .. [1] Hoyland, Kjetil, Michal Kaut, and Stein W. Wallace. A heuristic for moment-matching scenario generation. Computational optimization and applications 24.2-3 (2003): 169-185.
        """
        # true moments
        exp_mu = Para_match[0]
        exp_sds = Para_match[1]
        exp_skew = Para_match[2]
        exp_kur = Para_match[3]
        exp_cov_m = Para_match[4][0]

        # get correlation matrix from covariance matrix
        std_ = np.sqrt(np.diag(exp_cov_m))
        exp_cor_m = exp_cov_m / np.outer(std_, std_)

        # create scenarios
        xx = np.reshape(x, (-1, len(exp_mu)))

        # moments of the scenarios
        m1 = np.mean(xx,axis=0)
        m2 = np.std(xx,axis=0)
        m3 = [skew(xx[:,i]) for i in range(len(exp_mu)) ]
        m4 = [kurtosis(xx[:,i]) for i in range(len(exp_mu)) ]

        # sum of sum of squared residuals
        epsilon = sum( m1 - exp_mu )**2 + sum( m2 - exp_sds)**2 + sum( m3 - exp_skew)**2 + sum( m4 - exp_kur)**2 + sum(sum( (np.corrcoef(xx,rowvar=False)-(exp_cor_m))**2))

        return(epsilon)

    def MomentMatching(self,Para_match,S_N=30,silent=False):
        """ Moment Matching method, which minimize the error term between an approximation and the defined moments. The method starts by initializing the parameters using a normal distribution
        
        """
        # get covariance matrix
        covar = Para_match[4][0]
        # get starting parameters using a random normal distribution
        x_ini = np.random.multivariate_normal(Para_match[0],covar,S_N).ravel() # initial guess is a multivariate normal distribution

        # minimize the residuals
        res = minimize(self.MomentMatchingFunc, x_ini, method='CG',args=(Para_match)) # method="CG",options={'maxiter':60, 'disp': silent} 

        # get results and format them as a matrix
        scen = np.reshape(res.x, (-1, len(Para_match[0])))

        return(scen)

#%%

# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2010-01-01'
end_date = '2018-12-31'

# tickers
tickers = ["SPY","AGG"]

# User pandas_reader.data.DataReader to load the desired data.
panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)
df_close = panel_data["Adj Close"]

# Daily returns, and remove the first row as it is NA
df_ret = df_close.resample('M').last().pct_change().iloc[1:]

# define "True" moments
mu_ = df_ret.mean().values
std_ = df_ret.std().values
skew_ = df_ret.skew().values
kur_ = df_ret.kurtosis().values
cov_ = df_ret.cov().values

# ini moment matching algorithm
MM = MomentMatch()
# input parameters
Para_match = np.array([mu_,std_,skew_,kur_,[cov_] ])
# find a reduced number of scenarios with approx same statistical moments
MM_scen = MM.MomentMatching(Para_match=Para_match)
MM_df = pd.DataFrame(MM_scen,columns = df_ret.columns)

mu_
MM_df.mean()
(mu_-MM_df.mean())/MM_df.mean()

std_
MM_df.std()
(std_-MM_df.std())/MM_df.std()

skew_
MM_df.skew()
(skew_-MM_df.skew())/MM_df.skew()

kur_
MM_df.kurtosis()
(kur_-MM_df.kurtosis())/MM_df.kurtosis()

cov_
MM_df.cov()





df1 = df_ret.copy()
df2 = MM_df.copy()
df1['kind'] = 'dist1'
df2['kind'] = 'dist2'
df=pd.concat([df1,df2])

def multivariateGrid(col_x, col_y, col_k, df, k_is_color=False, scatter_alpha=.5):
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=name
        g.plot_joint(
            colored_scatter(df_group[col_x],df_group[col_y],color),
        )
        sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            color=color,
        )
        sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            color=color,            
            vertical=True
        )
    # Do also global Hist:
    sns.distplot(
        df[col_x].values,
        ax=g.ax_marg_x,
        color='grey'
    )
    sns.distplot(
        df[col_y].values.ravel(),
        ax=g.ax_marg_y,
        color='grey',
        vertical=True
    )
    plt.legend(legends)
     


multivariateGrid('AGG', 'SPY', 'kind', df=df)

plt.show()


