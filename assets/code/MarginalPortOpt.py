# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 09:30:23 2020

@author: ttb
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:33:47 2019

@author: ttb
"""

#%% packages

import pulp
import pandas as pd
import numpy as np
from pandas_datareader import data


#%% functions

def PortfolioRiskTarget(mu,scen,CVaR_target=1,lamb=1,max_weight=1,cvar_alpha=0.05,max_conc_risk=0.5):
    
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
    
    Returns
    -------
    float
        Asset weights in an optimal portfolio
        
    """
     
    # define index
    i_idx = mu.index
    j_idx = scen.index
    
    # number of scenarios
    N = scen.shape[0]    
    
    # define variables
    x = pulp.LpVariable.dicts("x", ( (i) for i in i_idx ),
                                         lowBound=0.00001,
                                         cat='Continuous') 
    
    # loss deviation
    VarDev = pulp.LpVariable.dicts("VarDev", ( (t) for t in j_idx ),
                                         lowBound=0,
                                         cat='Continuous')
        
    # value at risk
    VaR = pulp.LpVariable("VaR",   lowBound=0,
                                         cat='Continuous')
    CVaR = pulp.LpVariable("CVaR",   lowBound=0,
                                         cat='Continuous')
    
    # binary variable connected to th marginal risk contribution constraint
    b_z = pulp.LpVariable.dicts("b_z",   ( (t) for t in j_idx ),
                                         cat='Binary')
        
    #####################################
    ## define model
    model = pulp.LpProblem("Mean-CVaR Optimization", pulp.LpMaximize)
     
    #####################################
    ## Objective Function
             
    model += lamb*(pulp.lpSum([mu[i] * x[i] for i in i_idx] )) - (1-lamb)*CVaR
                      
    #####################################
    # constraint
                      
    # calculate CVaR
    for t in j_idx:
        model += -pulp.lpSum([ scen.loc[t,i]  * x[i] for i in i_idx] ) - VaR <= VarDev[t]
    
    model += VaR + 1/(N*cvar_alpha)*pulp.lpSum([ VarDev[t] for t in j_idx]) == CVaR    
    
    model += CVaR <= CVaR_target        
    
    ### price*number of products cannot exceed budget
    model += pulp.lpSum([ x[i] for i in i_idx]) == 1    
                                                
    ### Concentration limits
    # set max limits so it cannot not be larger than a fixed value  
    ###
    #for i in i_idx:
    #    model += x[i] <= max_weight
    
    # marginal risk constribution constrains
    for t in j_idx:
        model += b_z[t] <= 1000000*VarDev[t]

    for i in i_idx:
#        model += 1/(N*cvar_alpha) * pulp.lpSum([ scen.loc[t,i] * b_z[t] for t in j_idx]) <= max_conc_risk * CVaR_target / x[i]
        model += pulp.lpSum([ scen.loc[t,i] * b_z[t] for t in j_idx]) <= max_conc_risk * CVaR_target * N * cvar_alpha / x[i]
        
       
    # solve model
    model.solve()
    
    # print an error if the model is not optimal
    if pulp.LpStatus[model.status] != 'Optimal':
        print("Whoops! There is an error! The model has error status:" + pulp.LpStatus[model.status] )
        
    
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
    
        opt_port = port_total
    
    # set flooting data points to zero and normalize
    opt_port[opt_port < 0.000001] = 0
    opt_port = opt_port/sum(opt_port)
    
    # return portfolio, CVaR, and alpha
    return opt_port, var_model["CVaR"], sum(mu * port_total)


def PortfolioLambda(mu,scen,max_weight=1,cvar_alpha=0.05,ft_points=15,max_conc_risk=0.5):

    # asset names
    assets = mu.index

    # column names
    col_names = mu.index.values.tolist() 
    col_names.extend(["Mu","CVaR","STAR"])
    # number of frontier points 

    
    # store portfolios
    portfolio_ft = pd.DataFrame(columns=col_names,index=list(range(ft_points)))
    
    # maximum risk portfolio    
    lamb=0.99999
    max_risk_port, max_risk_CVaR, max_risk_mu = PortfolioRiskTarget(mu=mu,scen=scen,CVaR_target=100,lamb=lamb,max_weight=max_weight,cvar_alpha=cvar_alpha,max_conc_risk=max_conc_risk)
    portfolio_ft.loc[ft_points-1,assets] = max_risk_port
    portfolio_ft.loc[ft_points-1,"Mu"] = max_risk_mu
    portfolio_ft.loc[ft_points-1,"CVaR"] = max_risk_CVaR
    portfolio_ft.loc[ft_points-1,"STAR"] = max_risk_mu/max_risk_CVaR
    
    # minimum risk portfolio
    lamb=0.00001
    min_risk_port, min_risk_CVaR, min_risk_mu= PortfolioRiskTarget(mu=mu,scen=scen,CVaR_target=100,lamb=lamb,max_weight=max_weight,cvar_alpha=cvar_alpha,max_conc_risk=max_conc_risk)
    portfolio_ft.loc[0,assets] = min_risk_port
    portfolio_ft.loc[0,"Mu"] = min_risk_mu
    portfolio_ft.loc[0,"CVaR"] = min_risk_CVaR
    portfolio_ft.loc[0,"STAR"] = min_risk_mu/min_risk_CVaR
    
    # CVaR step size
    step_size = (max_risk_CVaR-min_risk_CVaR)/ft_points # CVaR step size
    
    # calculate all frontier portfolios
    for i in range(1,ft_points-1):
        CVaR_target = min_risk_CVaR + step_size*i
        i_risk_port, i_risk_CVaR, i_risk_mu= PortfolioRiskTarget(mu=mu,scen=scen,CVaR_target=CVaR_target,lamb=1,max_weight=max_weight,cvar_alpha=cvar_alpha,max_conc_risk=max_conc_risk)
        portfolio_ft.loc[i,assets] = i_risk_port
        portfolio_ft.loc[i,"Mu"] = i_risk_mu
        portfolio_ft.loc[i,"CVaR"] = i_risk_CVaR
        portfolio_ft.loc[i,"STAR"] = i_risk_mu/i_risk_CVaR
        
    return portfolio_ft
     
    
#%%

# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2010-01-01'
end_date = '2018-12-31'

# tickers
tickers = ["SPY","IJR","EFA","EEM","AGG"]

# User pandas_reader.data.DataReader to load the desired data.
panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)
df_close = panel_data["Adj Close"]

# monthly returns from daily prices, and remove the first row as it is NA
df_ret = df_close.resample('M').last().pct_change().iloc[1:]


#%%

mu = df_ret.mean()
scen = df_ret
min_weight = 0
cvar_alpha=0.05
max_conc_risk = 0.5
_port, _ , _ = PortfolioRiskTarget(mu,scen,CVaR_target=100,lamb=1,max_weight=1,cvar_alpha=0.05,max_conc_risk=max_conc_risk)

print(_port)



#%% compute the optimal portfolio outperforming zero percentage return

mu = df_ret.mean()
scen = df_ret
min_weight = 0
cvar_alpha=0.05
Frontier_port = PortfolioLambda(mu,scen,max_weight=1,min_weight=None,cvar_alpha=cvar_alpha)


Frontier_port[["CVaR","Mu"]].plot(x="CVaR",y="Mu",label="Efficient Frontier",title="Risk-Reward",figsize=(10,5))
plt.show()

Frontier_port[tickers].plot.bar(stacked=True,title="Allocation vs Efficient Points",figsize=(10,5))
plt.show()

