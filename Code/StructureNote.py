# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:15:28 2020

@author: renli
"""

#%%
# =============================================================================
# portfolio constructing
# =============================================================================
def EqualShareValue(df,buy_date,initial_balance):
    import pandas as pd
    buy_date=pd.Timestamp(buy_date)
    shares=df.loc[buy_date,:]
    shares=initial_balance/shares.shape[0]/shares
    return shares

def EqualPortfolioValue(df,start_date,end_date,shares):
    import pandas as pd
    start_date,end_date=pd.Timestamp(start_date),pd.Timestamp(end_date)
    # shares=EqualShareValue(df,start_date,initial_balance)
    value=(df.loc[start_date:end_date,:]*shares).sum(axis=1)
    value.name='Value'
    return value

def Value2Return(portfolio_value):
    import numpy as np
    portfolio_daily_logreturn=(np.log(portfolio_value).diff().dropna()/
                               portfolio_value.index.to_series().diff().dropna().map(lambda x:np.sqrt(x.days)))
    return portfolio_daily_logreturn
#%%
# =============================================================================
# volatility model
# =============================================================================
def realized_volatility(portfolio_value,end_date,window=None):
    import numpy as np
    import pandas as pd
    end_date=np.datetime64(end_date,'D')
    start_date=np.busday_offset(end_date,-window-1) if window!=None else portfolio_value.index[0]
    start_date,end_date=pd.Timestamp(start_date),pd.Timestamp(end_date)
    portfolio_value=portfolio_value[start_date:end_date]
    portfolio_daily_logreturn=Value2Return(portfolio_value)
    return portfolio_daily_logreturn.std()*np.sqrt(365)

def garch_volatility(portfolio_value,end_date,window=None):
    import arch
    end_index=portfolio_value.index.to_list().index(end_date)
    if window==None: window=end_index
    portfolio_value=portfolio_value.iloc[end_index-window:end_index+1]
    portfolio_logreturn=Value2Return(portfolio_value)
    model=arch.arch_model(y=portfolio_logreturn,mean='Constant',rescale=True,
                          lags=0,vol='GARCH',p=1,o=0,q=1,power=2)
    return model.fit()
    
#%%
# =============================================================================
# option model
# =============================================================================
def d1(S,K,r,tau,sigma):
    import numpy as np
    return (np.log(S/K)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))

def d2(S,K,r,tau,sigma):
    import numpy as np
    return (np.log(S/K)+(r-0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))

def vanilla_call(S,K,r,tau,sigma):
    from scipy.stats import norm
    import numpy as np
    if tau==0: return np.max([S-K,0])
    return (S * norm.cdf(d1(S, K, r, tau, sigma))-
            K * np.exp(-r * tau)* norm.cdf(d2(S, K, r, tau, sigma)))

def vanilla_put(S,K,r,tau,sigma):
    from scipy.stats import norm
    import numpy as np
    if tau==0: return np.max([K-S,0])
    return (K * np.exp(-r * tau)* norm.cdf(-d2(S, K, r, tau, sigma))-
            S * norm.cdf(-d1(S, K, r, tau, sigma)))

def binary_call(S,K,r,tau,sigma):
    from scipy.stats import norm
    import numpy as np
    return np.exp(-r * tau) * norm.cdf(d2(S, K, r, tau, sigma))

def binary_put(S,K,r,tau,sigma):
    from scipy.stats import norm
    import numpy as np
    return np.exp(-r * tau) * norm.cdf(-d2(S, K, r, tau, sigma))

def risk_reversal(S,K1,K2,r,tau,sigma):
    # import numpy as np
    # if K1<=K2: return np.nan
    return vanilla_call(S,K1,r,tau,sigma)-vanilla_put(S,K2,r,tau,sigma)

def straddle(S,K,r,tau,sigma):
    return vanilla_call(S,K,r,tau,sigma)+vanilla_put(S,K,r,tau,sigma)

def strangle(S,K1,K2,r,tau,sigma):
    # import numpy as np
    # if K1<=S or K2>=S: return np.nan
    return vanilla_call(S,K1,r,tau,sigma)+vanilla_put(S,K2,r,tau,sigma)

def butterfly(S,K,K1,K2,r,tau,sigma):
    return strangle(S,K1,K2,r,tau,sigma)-straddle(S,K,r,tau,sigma)

def butterflyD(S,K,K1,K2,r,tau,sigma):
    return (-2) * vanilla_call(S,K,r,tau,sigma) + vanilla_call(S,K1,r,tau,sigma) \
            + vanilla_call(S,K2,r,tau,sigma)

def chooser(S,K,r,tau,sigma):
    import numpy as np
    return vanilla_call(S,K,r,tau,sigma)+max(0,K * np.exp(-r * tau) - S)

def bull_spread(S,K1,K2,r,tau,sigma):
    return vanilla_call(S,K1,r,tau,sigma)-vanilla_call(S,K2,r,tau,sigma)

def bear_spread(S,K1,K2,r,tau,sigma):
    return vanilla_put(S,K1,r,tau,sigma)-vanilla_put(S,K2,r,tau,sigma)

def sigma(sigma_1, sigma_2, rho):
    import numpy as np
    return np.sqrt(sigma_1**2 - 2 * rho * sigma_1 * sigma_2 + sigma_2**2)

def spread_call(S1, S2, K, r, tau, sigma_1, sigma_2, rho):
    from scipy.stats import norm
    import numpy as np
    sigma_2*=S2/(S2+K*np.exp(-r*tau))
    return S1 * norm.cdf(d1(S1, S2 + K * np.exp(-r * tau), 0, tau, sigma(
        sigma_1, sigma_2, rho))) - (S2 + K * np.exp(-r * tau))* norm.cdf(
            d2(S1, S2 + K * np.exp(-r * tau), 0, tau, sigma(sigma_1, sigma_2, rho)))
            
def spread_put(S1, S2, K, r, tau, sigma_1, sigma_2, rho):
    from scipy.stats import norm
    import numpy as np
    sigma_2*=S2/(S2+K*np.exp(-r*tau))
    return (S2 + K * np.exp(-r * tau)) * norm.cdf(
        -d2(S1, S2 + K * np.exp(-r * tau), 0, tau, sigma(sigma_1, sigma_2, rho))) - S1 * norm.cdf(
            -d1(S1, S2 + K * np.exp(-r * tau), 0, tau, sigma(sigma_1, sigma_2, rho)))
#%%
# =============================================================================
# greek model
# =============================================================================
def vanilla_call_delta(S,K,r,tau,sigma):
    from scipy.stats import norm
    return norm.cdf(d1(S, K, r, tau, sigma))

def vanilla_put_delta(S,K,r,tau,sigma):
    from scipy.stats import norm
    return -norm.cdf(-d1(S, K, r, tau, sigma))

def vanilla_vega(S,K,r,tau,sigma):
    from scipy.stats import norm
    import numpy as np
    return S * norm.pdf(d1(S, K, r, tau, sigma)) * np.sqrt(tau)

def approximate_greek(price_fcn,partial_var,epsilon,**kwargs):
    upper_kwargs=kwargs.copy()
    lower_kwargs=kwargs.copy()
    upper_kwargs[partial_var]+=epsilon
    lower_kwargs[partial_var]-=epsilon
    return (price_fcn(**upper_kwargs)-price_fcn(**lower_kwargs))/(2*epsilon)
#%%
# =============================================================================
# auxilliary tools
# =============================================================================
def kdeplot(x,cumulative=False,dpi=200):
    import numpy as np
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(0,dpi=dpi)
    sns.kdeplot(x, shade = True, cumulative = cumulative,kernel='gau')
    sample_x=np.linspace(x.min(),x.max(), 100)
    sns.lineplot(x=sample_x,y=norm.pdf(sample_x,x.mean(),x.std()))
    
def fplot(f,xmin,xmax,num=100,**kwargs):
    import numpy as np
    import seaborn as sns
    x=np.linspace(xmin,xmax,num=num)
    f=np.vectorize(f)
    y=f(x)
    sns.lineplot(x=x,y=y,**kwargs)
    return np.concatenate((x.reshape((num,1)),y.reshape((num,1))),axis=1)
#%%
# =============================================================================
# structure note solver
# =============================================================================
def option_payoff(ST,option_fcn,**option_kwargs):
    option_kwargs['S']=ST
    option_kwargs['tau']=0
    return option_fcn(**option_kwargs)
    
def structure_note_payoff(ST,par,bond_tau,bond_yield,participate_rate,option_fcn,**option_kwargs):
    import numpy as np
    bond_value=par/np.exp(bond_tau*bond_yield)
    option_price=option_fcn(**option_kwargs)
    extra_coupon=par-bond_value-participate_rate*option_price
    option_value=participate_rate*option_payoff(ST,option_fcn,**option_kwargs)
    return (bond_value+extra_coupon)*np.exp(bond_tau*bond_yield)+option_value

#%%
# =============================================================================
# experiment tools
# =============================================================================
def option_payoff_fplot(option_fcn,Smin,Smax,option_kwargs):
    """
    Parameters
    ----------
    option_fcn : function
        above option pricing fcn.
    Smin : double
        minimum ST.
    Smax : double
        maximum ST.
    option_kwargs : dict
        parameters for option_fcn. keys must be parameter's name, e.g. S,K,r...,
        values are the corresponding numbers.

    Returns
    -------
    (x,y) : ndarray (n*2)
        first col is x, second col is y. Also, a figure of (x,y) is drawn.
    """
    import matplotlib.pyplot as plt
    plt.figure(dpi=200)
    f=lambda ST:option_payoff(ST,option_fcn,**option_kwargs)
    return fplot(f,Smin,Smax,500)
    
def structure_note_payoff_fplot(option_fcn,Smin,Smax,structure_note_kwargs):
    """
    Parameters
    ----------
    option_fcn : function
        above option pricing fcn.
    Smin : double
        minimum ST.
    Smax : double
        maximum ST.
    structure_note_kwargs : dict
        parameters for option_fcn. keys must be parameter's name, e.g. bond_tau,bond_yield,K,r...,
        values are the corresponding numbers.

    Returns
    -------
    (x,y) : ndarray (n*2)
        first col is x, second col is y. Also, a figure of (x,y) is drawn.
    """
    import matplotlib.pyplot as plt
    plt.figure(dpi=200)
    f=lambda ST:structure_note_payoff(ST,option_fcn=option_fcn,**structure_note_kwargs)
    return fplot(f,Smin,Smax,500)

def option_payoff_fplot_list(option_fcn,Smin,Smax,option_kwargs_list):
    import numpy as np
    import matplotlib.pyplot as plt
    res=[]
    plt.figure(dpi=200)
    for option_kwargs in option_kwargs_list:
        f=lambda ST:option_payoff(ST,option_fcn,**option_kwargs)
        x,y=fplot(f,Smin,Smax,500).T
        if not res: res.append(x)
        res.append(y)
    return np.array(res).T

def structure_note_payoff_fplot_list(option_fcn,Smin,Smax,structure_note_kwargs_list):
    import numpy as np
    import matplotlib.pyplot as plt
    res=[]
    plt.figure(dpi=200)
    for structure_note_kwargs in structure_note_kwargs_list:
        f=lambda ST:structure_note_payoff(ST,option_fcn=option_fcn,**structure_note_kwargs)
        x,y=fplot(f,Smin,Smax,500).T
        if not res: res.append(x)
        res.append(y)
    return np.array(res).T

def option_payoff_fplot_with_dist(option_fcn,Smin,Smax,option_kwargs):
    import numpy as np
    from scipy.stats import lognorm
    import matplotlib.pyplot as plt
    dist=lognorm(s=option_kwargs['sigma']*np.sqrt(option_kwargs['tau']),
                  scale=option_kwargs['S']*np.exp(
                  (option_kwargs['r']-option_kwargs['sigma']**2/2)*
                  option_kwargs['tau']))
    f=lambda ST:option_payoff(ST,option_fcn,**option_kwargs)
    if ('K1' in option_kwargs) and ('K2' in option_kwargs):
        conditional_payoff=dist.expect(f,option_kwargs['K1'],option_kwargs['K2'],conditional=True)
        conditional_prob=dist.cdf(option_kwargs['K2'])-dist.cdf(option_kwargs['K1'])
    else:
        conditional_payoff=0
        conditional_prob=0
    _,ax1=plt.subplots(dpi=200)
    plt.title('conditional payoff = '+str(round(conditional_payoff,2))+'; '+
                                          'conditional prob = '+str(round(conditional_prob,3)))
    plt.xlabel('ST')
    plt.ylabel('PDF')
    fplot(dist.pdf,Smin,Smax,500,ax=ax1,c='C1',label='pdf of ST')
    plt.legend(loc='upper left')
    ax1.set_ylim(top=dist.pdf(dist.mean())*2)
    ax2=ax1.twinx()
    plt.ylabel('payoff')
    res=fplot(f,Smin,Smax,500,ax=ax2,c='C0',label='payoff')
    plt.legend(loc='upper right')
    return res

def structure_note_payoff_fplot_with_dist(option_fcn,Smin,Smax,structure_note_kwargs):
    import numpy as np
    from scipy.stats import lognorm
    import matplotlib.pyplot as plt
    dist=lognorm(s=structure_note_kwargs['sigma']*np.sqrt(structure_note_kwargs['tau']),
                  scale=structure_note_kwargs['S']*np.exp(
                  (structure_note_kwargs['r']-structure_note_kwargs['sigma']**2/2)*
                  structure_note_kwargs['tau']))
    f=lambda ST:structure_note_payoff(ST,option_fcn=option_fcn,**structure_note_kwargs)
    if ('K1' in structure_note_kwargs) and ('K2' in structure_note_kwargs):
        conditional_payoff=dist.expect(f,structure_note_kwargs['K1'],structure_note_kwargs['K2'],conditional=True)
        conditional_prob=dist.cdf(structure_note_kwargs['K2'])-dist.cdf(structure_note_kwargs['K1'])
    else:
        conditional_payoff=0
        conditional_prob=0
    _,ax1=plt.subplots(dpi=200)
    plt.title('conditional payoff = '+str(round(conditional_payoff,2))+'; '+
                                          'conditional prob = '+str(round(conditional_prob,3)))
    plt.xlabel('ST')
    plt.ylabel('PDF')
    fplot(dist.pdf,Smin,Smax,500,ax=ax1,c='C1',label='pdf of ST')
    plt.legend(loc='upper left')
    ax1.set_ylim(top=dist.pdf(dist.mean())*2)
    ax2=ax1.twinx()
    plt.ylabel('payoff')
    res=fplot(f,Smin,Smax,500,ax=ax2,c='C0',label='payoff')
    plt.legend(loc='upper right')
    return res

def butterfly_K_solver(par,bond_tau,bond_yield,option_fcn,**option_kwargs):
    """
    Solve out the best strike price for a butterfly (or whatever option also involve K,K1,K2)
    Return a namedtuple, can be called in a form as OptimizedParam.xxx --'extra_coupon','participate_rate','K1','K2'
    """
    if ('K' not in option_kwargs) or ('K1' not in option_kwargs) or ('K2' not in option_kwargs):
        print("option_kwargs doesn't satisfy.")
        return None
    
    from collections import namedtuple
    import numpy as np
    from scipy.optimize import root_scalar
    participate_rate=option_kwargs.pop('participate_rate',1)#here 1 means set participate rate = 1 by force
    OptimizedParam=namedtuple('OptimizedParam',['extra_coupon','participate_rate','K1','K2'])
    def f(epsilon):
        option_kwargs['K1']=option_kwargs['K']-epsilon
        option_kwargs['K2']=option_kwargs['K']+epsilon
        bond_value=par/np.exp(bond_tau*bond_yield)
        option_price=option_fcn(**option_kwargs)
        remain_money = par - bond_value - participate_rate * option_price 
        return remain_money
    res=root_scalar(f,method='brentq',bracket=(0,option_kwargs['K']-1e-6))
    option_kwargs['K1'],option_kwargs['K2']=option_kwargs['K']-res.root,option_kwargs['K']+res.root
    extra_coupon = par - par/np.exp(bond_tau*bond_yield) - participate_rate * option_fcn(**option_kwargs)
    return OptimizedParam(extra_coupon*np.exp(bond_tau*bond_yield),
                          participate_rate,option_kwargs['K1'],option_kwargs['K2'])

#%%
if __name__ == "__main__":
    import pandas as pd
    df=pd.read_excel(r'D:\Document\Study_File\NYU_Tandon_MFE\IAQF\Dstock_col.xlsx').set_index('Date')
    # df2=pd.read_excel(r'D:\Document\Study_File\NYU_Tandon_MFE\IAQF\Rstock_col.xlsx').set_index('Date')
    portfolio_value=EqualPortfolioValue(df,df.index[0],df.index[-1],
                                        EqualShareValue(df,df.index[0],100))
    portfolio_logreturn=Value2Return(portfolio_value)
    kdeplot(portfolio_logreturn)
    # vol1=realized_volatility(portfolio_value,portfolio_value.index[-1])
    # vol2=garch_volatility(portfolio_value,portfolio_value.index[-1]).forecast()