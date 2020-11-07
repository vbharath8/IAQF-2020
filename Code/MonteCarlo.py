# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 13:49:21 2020

@author: renli
"""
import numba as nb
import numpy as np
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
# payoff fcn
# =============================================================================
@nb.jit(nopython=True,nogil=True)
def vanilla_call_payoff(K,stoh_path):
    """
    Parameters
    ----------
    K : double
        strike price.
    stoh_path : ndarray (n*m)
        n paths; m time steps.
    
    Returns
    -------
    payoff : double
        average payoff for n paths.
    """
    payoff=stoh_path[:,-1]-K
    payoff=(payoff>0)*payoff
    return np.mean(payoff)

@nb.jit(nopython=True,nogil=True)
def vanilla_put_payoff(K,stoh_path):
    payoff=K-stoh_path[:,-1]
    payoff=(payoff>0)*payoff
    return np.mean(payoff)

@nb.jit(nopython=True,nogil=True)
def binary_call_payoff(K,stoh_path):
    payoff=(stoh_path[:,-1]-K)>0
    return np.mean(payoff)

@nb.jit(nopython=True,nogil=True)
def binary_put_payoff(K,stoh_path):
    payoff=(K-stoh_path[:,-1])>0
    return np.mean(payoff)

def risk_reversal_payoff(K1,K2,stoh_path):
    if K1<=K2: return None
    return vanilla_call_payoff(K1,stoh_path)-vanilla_put_payoff(K2,stoh_path)

def straddle(K,stoh_path):
    return vanilla_call_payoff(K,stoh_path)+vanilla_put_payoff(K,stoh_path)

def strangle(S,K1,K2,stoh_path):
    if K1<=S or K2>=S: return None
    return vanilla_call_payoff(K1,stoh_path)+vanilla_put_payoff(K2,stoh_path)

def butterfly(S,K,K1,K2,stoh_path):
    return strangle(S,K1,K2,stoh_path)-straddle(K,stoh_path)

def chooser(S,K,r,tau,stoh_path):
    return vanilla_call_payoff(K,stoh_path)+max(0,K * np.exp(-r * tau) - S)
#%%
# =============================================================================
# simulation
# =============================================================================
def get_kde(portfolio_return,**kwargs):
    from scipy.stats import gaussian_kde
    return gaussian_kde(portfolio_return,**kwargs)

@nb.jit(nopython=True,nogil=True)
def simulate_path_eurostyle(S,r,sigma,random_return):
    S_vec=S*np.ones((random_return.shape[0],))
    for col_idx in range(random_return.shape[1]):
        S_vec*=np.exp((r-(sigma**2)/2)+random_return[:,col_idx])
    return S_vec.reshape((S_vec.shape[0],1))

@nb.jit(nopython=True,nogil=True)
def simulate_path_amestyle(S,r,sigma,random_return):
    random_return[:,0]=S*np.exp((r-sigma**2/2)+random_return[:,0])
    for col_idx in range(1,random_return.shape[1]):
        random_return[:,col_idx]=random_return[:,col_idx-1]*np.exp(
            (r-sigma**2/2)+random_return[:,col_idx])
    return random_return

def monte_carlo_pricing_kde(S,r,sigma,payoff_fcn,sampler,path_num,step_num,kwargs,**kwgs):
    random_return=sampler.resample(size=path_num*step_num).reshape((path_num,step_num))
    stoh_path=simulate_path_eurostyle(S,r,sigma,random_return)
    return payoff_fcn(stoh_path=stoh_path,**kwargs)

def monte_carlo_pricing_dist(S,r,sigma,payoff_fcn,sampler,path_num,step_num,kwargs,**kwgs):
    random_return=sampler.rvs(size=path_num*step_num).reshape((path_num,step_num))
    stoh_path=simulate_path_eurostyle(S,r,sigma,random_return)
    return payoff_fcn(stoh_path=stoh_path,**kwargs)

#%%
# =============================================================================
# multiprocessing
# =============================================================================
def multiprocess_on_fcn(fcn,kwargs_list):
    from multiprocessing import Pool
    res=[]
    with Pool() as pool:
        for i,kwargs in enumerate(kwargs_list):
            res.append((i,pool.apply_async(func=fcn,args=(),kwds=kwargs)))
        pool.close()
        pool.join()
    res.sort(key=lambda x:x[0])
    res=[ele[1].get() for ele in res]
    return res

def fplot(f,xmin,xmax,num=100):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    x=np.linspace(xmin,xmax,num=num)
    f=np.vectorize(f)
    y=f(x)
    plt.figure(dpi=200)
    sns.lineplot(x=x,y=y)
    return y

def multiprocess_fplot(f,xmin,xmax,point_num):
    import numpy as np
    import seaborn as sns
    def _multiprocess_on_fcn(fcn,args_list):
        from multiprocessing import Pool
        res=[]
        with Pool() as pool:
            for i,args in enumerate(args_list):
                res.append((i,pool.apply_async(func=fcn,args=args)))
            pool.close()
            pool.join()
        res.sort(key=lambda x:x[0])
        res=[ele[1].get() for ele in res]
        return res
    x=np.linspace(xmin,xmax,num=point_num)
    args=[(ele,) for ele in x]
    y=_multiprocess_on_fcn(f,args)
    sns.lineplot(x=x,y=y)
    
#%%

    
if __name__=='__main__':
    import pandas as pd
    from scipy.stats import norm
    df=pd.read_excel(r'D:\Document\Study_File\NYU_Tandon_MFE\IAQF\Dstock_col.xlsx').set_index('Date')
    portfolio_value=EqualPortfolioValue(df,df.index[0],df.index[-1],
                                        EqualShareValue(df,df.index[-1],100))
    portfolio_logreturn=Value2Return(portfolio_value)
    r=np.mean(portfolio_logreturn)
    sigma=np.std(portfolio_logreturn)
    portfolio_logreturn-=np.mean(portfolio_logreturn)#centralization
    kde_sampler=get_kde(portfolio_logreturn)#this pdf is for daily logreturn
    norm_sampler=norm(scale=sigma)
    """
    For example, if we want to calculate some option's expected payoff by MC,
    we can copy n times the same parameters and run it by multiprocessing.
    Thus, if we calculate the average value of these n expected payoff, we
    essentially increase the path_num by n times!
    The following is an example of calling "multiprocess_on_fcn" function.
    """
    payoff_kwargs={'K':100}
    kwargs={
        'S':100,
        'r':r,
        'sigma':sigma,
        'payoff_fcn':vanilla_call_payoff,
        'sampler':kde_sampler,
        'path_num':100000,
        'step_num':365,
        'kwargs':payoff_kwargs
        }
    kwargs_list=[kwargs.copy() for _ in range(10)]
    res=multiprocess_on_fcn(monte_carlo_pricing_kde,kwargs_list)