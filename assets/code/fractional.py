# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:09:18 2019

@author: ttb
"""

#%%

import pulp
import pandas as pd
import numpy as np
from pandas_datareader import data

#%% functions

def MeanCVaR(mu,mu_b,scen,scen_b,max_weight=1,min_weight=None,cvar_alpha=0.05):
        
    """ This function finds the optimal enhanced index portfolio according to some benchmark. The portfolio corresponds to the tangency portfolio where risk is evaluated according to the CVaR of the tracking error. The model is formulated using fractional programming.
    
    Parameters
    ----------
    mu : pandas.Series with float values
        asset point forecast
    mu_b : pandas.Series with float values
        Benchmark point forecast
    scen : pandas.DataFrame with float values
        Asset scenarios
    scen_b : pandas.Series with float values
        Benchmark scenarios
    max_weight : float
        Maximum allowed weight    
    cvar_alpha : float
        Alpha value used to evaluate Value-at-Risk one
    min_val : float    
        Values less than this are set to zero. This is used to eliminate floating points    
    
    
    Returns
    -------
    float
        Asset weights in an optimal portfolio
        
    Notes
    -----
    The tangency mean-CVaR portfolio is effectively optimized as:
        .. math:: STARR(w) = frac{w^T*mu - mu_b}{CVaR(w^T*scen - scen_b)}
    This procedure is described in [1]_
       
    References
    ----------
        .. [1] Stoyanov, Stoyan V., Svetlozar T. Rachev, and Frank J. Fabozzi. "Optimal financial portfolios." Applied Mathematical Finance 14.5 (2007): 401-436.
       
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> mean = (0.08, 0.09)
    >>> cov = [[0.19, 0.05], [0.05, 0.18]]
    >>> scen = pd.dataframe(np.random.multivariate_normal(mean, cov, 100))
    >>> portfolio = IndexTracker(pd.Series(mean), pd.Series(0), pd.dataframe(scen),pd.Series(0,index=scen.index) )
    """
    
    
    # define index
    i_idx = mu.index
    j_idx = scen.index
    
    # number of scenarios
    N = scen.shape[0]    
    M = 100000 # large number
    
    # define variables
    x = pulp.LpVariable.dicts("x", ( (i) for i in i_idx ),
                                         lowBound=0,
                                         cat='Continuous') 
    
    # loss deviation
    VarDev = pulp.LpVariable.dicts("VarDev", ( (t) for t in j_idx ),
                                         lowBound=0,
                                         cat='Continuous')
    
    # scaling variable used by fractional programming formulation
    Tau = pulp.LpVariable("Tau",   lowBound=0,
                                         cat='Continuous')
    
    # value at risk
    VaR = pulp.LpVariable("VaR",   lowBound=0,
                                         cat='Continuous')
            
    b_z = pulp.LpVariable.dicts("b_z",   ( (i) for i in i_idx ),
                                         cat='Binary')
    
    # auxiliary to make a non-linear formulation linear (see Linear Programming Models based on Omega Ratio for the Enhanced Index Tracking Problem) 
    Tau_i = pulp.LpVariable.dicts("Tau_i", ( (i) for i in i_idx ),
                                         lowBound=0,
                                         cat='Continuous')
    
    
    #####################################
    ## define model
    model = pulp.LpProblem("Enhanced Index Tracker (STAR ratio)", pulp.LpMaximize)
     
    #####################################
    ## Objective Function
             
    model += pulp.lpSum([mu[i] * x[i] for i in i_idx] ) - Tau*mu_b
                      
    #####################################
    # constraint
                      
    # calculate CVaR
    for t in j_idx:
        model += -pulp.lpSum([ scen.loc[t,i]  * x[i] for i in i_idx] ) + scen_b[t]*Tau - VaR <= VarDev[t]
    
    model += VaR + 1/(N*cvar_alpha)*pulp.lpSum([ VarDev[t] for t in j_idx]) <= 1    
    
    ### price*number of products cannot exceed budget
    model += pulp.lpSum([ x[i] for i in i_idx]) == Tau    
                                                
    ### Concentration limits
    # set max limits so it cannot not be larger than a fixed value  
    ###
    for i in i_idx:
        model += x[i] <= max_weight*Tau
    
    ### Add minimum weight constraint, either zero or atleast minimum weight
    if min_weight is not None:
    
        for i in i_idx:                           
            model += x[i] >= min_weight*Tau_i[i]
            model += x[i] <= max_weight*Tau_i[i]
            model += Tau_i[i] <= M*b_z[i]
            model += Tau_i[i] <= Tau
            model += Tau - Tau_i[i] + M*b_z[i] <= M
    
    
    # solve model
    model.solve(pulp.PULP_CBC_CMD(maxSeconds=60, msg=1, fracGap=0))
    
    # print an error if the model is not optimal
    if pulp.LpStatus[model.status] != 'Optimal':
        print("Whoops! There is an error! The model as error status:" + pulp.LpStatus[model.status] )
        
    
    #Get positions    
    if pulp.LpStatus[model.status] == 'Optimal':
     
        # print variables
        var_model = dict()
        for variable in model.variables():
            var_model[variable.name] = variable.varValue
         
        # solution with variable names   
        var_model = pd.Series(var_model,index=var_model.keys())
         
         
        long_pos = [i for i in var_model.keys() if i.startswith("x") ]
                     
        # total portfolio with negative values as short positions
        port_total = pd.Series(var_model[long_pos].values ,index=[t[2:] for t in var_model[long_pos].index])
        
        # value of tau
        tau_val = var_model[var_model.keys() == 'Tau']
    
        opt_port = port_total/tau_val[0]
        opt_port[opt_port < 0.00001] = 0 # get rid of floating numbers
            
    return opt_port
    

#%%

# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2010-01-01'
end_date = '2018-12-31'

# tickers
tickers = ["SPY","IJS","EFA","EEM","AGG"]

# User pandas_reader.data.DataReader to load the desired data.
panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)
df_close = panel_data["Adj Close"]

# monthly returns from daily prices, and remove the first row as it is NA
df_ret = df_close.resample('M').last().pct_change().iloc[1:]

#%% compute the optimal portfolio outperforming zero percentage return

mu = df_ret.mean()
mu_b = 0
scen = df_ret
scen_b = pd.Series(0,index=df_ret.index)
min_weight = 0
cvar_alpha=0.05
OptPort_zero = MeanCVaR(mu,mu_b,scen,scen_b,max_weight=1,min_weight=None,cvar_alpha=cvar_alpha)

# historic portfolio return
port_ret_zero = df_ret.dot(OptPort_zero)

# cumulative return of the portfolio
(1+port_ret_zero).cumprod().plot()

# average monthly return
mu.dot(OptPort_zero)

# average monthly CVaR
port_ret_zero.loc[port_ret_zero <= port_ret_zero.quantile(0.05)].mean()

# sharpe ratio
port_ret_zero.mean()/port_ret_zero.std()*np.sqrt(12)

#%% compute the optimal portfolio outperforming zero percentage return

mu = df_ret.mean()
mu_b = df_ret.mean().mean()
scen = df_ret
scen_b = df_ret.mean(axis=1)
min_weight = 0
cvar_alpha=0.05
OptPort_1N = MeanCVaR(mu,mu_b,scen,scen_b,max_weight=1,min_weight=None,cvar_alpha=cvar_alpha)

# historic portfolio return
port_ret_1N = df_ret.dot(OptPort_1N)

# cumulative return of the portfolio
(1+port_ret_1N).cumprod().plot()

# average monthly return
mu.dot(OptPort_1N)

# average monthly CVaR
port_ret_1N.loc[port_ret_1N <= port_ret_1N.quantile(0.05)].mean()

# sharpe ratio
port_ret_1N.mean()/port_ret_1N.std()*np.sqrt(12)


#%% Which are best? what if we can apply leverage

# leverage
risk_ratio = port_ret_1N.std()/port_ret_zero.std()

# assuming zero financing rate
financing_rate = 0

# horse race
(1+risk_ratio*port_ret_zero - financing_rate).cumprod().plot()
(1+port_ret_1N).cumprod().plot()

# break-even financing rate 
financing_rate = (risk_ratio*port_ret_zero).mean() - port_ret_1N.mean()
# horse race
(1+risk_ratio*port_ret_zero - financing_rate).cumprod().plot()
(1+port_ret_1N).cumprod().plot()




