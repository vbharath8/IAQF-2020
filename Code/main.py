# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:15:31 2020

@author: renli
"""

#%%
if __name__=='__main__':
    import StructureNote as SN
    import numpy as np
    # setting parameters
    option_kwargs={
        'S':373.51,
        'K':373.51,
        'K1':366.06,
        'K2':380.96,
        'r':0.0147,
        'tau':1,
        'sigma':0.1429
        }
    structure_note_kwargs={
        'par':100,
        'bond_tau':1,
        'bond_yield':0.0225,
        'participate_rate':1
        }
    structure_note_kwargs.update(option_kwargs)
    """
    example for plotting one parameter setting.
    """
    # res1=SN.option_payoff_fplot(SN.butterflyD, 340, 400, option_kwargs)
    # res2=SN.structure_note_payoff_fplot(SN.butterflyD, 340, 400,structure_note_kwargs)
    
    """
    example for plotting multiple parameter settings.
    """
    # option_kwargs_list=[]
    # structure_note_kwargs_list=[]
    # for _ in range(10):
    #     option_kwargs['K1']-=5
    #     option_kwargs['K2']+=5
    #     temp=option_kwargs.copy()
    #     option_kwargs_list.append(temp)
    # for i in range(10):
    #     temp=structure_note_kwargs.copy()
    #     temp.update(option_kwargs_list[i])
    #     structure_note_kwargs_list.append(temp)

    # res1=SN.option_payoff_fplot_list(SN.butterfly, 0, 200, option_kwargs_list)
    # res2=SN.structure_note_payoff_fplot_list(SN.butterfly, 0, 200, structure_note_kwargs_list)
    """
    example for plotting with theoritical ST distribution
    """
    res1=SN.option_payoff_fplot_with_dist(SN.butterflyD,300,450,option_kwargs)
    res2=SN.structure_note_payoff_fplot_with_dist(SN.butterflyD,300,450,structure_note_kwargs)
    """
    example for searching best K1, K2 for butterfly
    """
    # res=SN.butterfly_K_solver(option_fcn=SN.butterfly,**structure_note_kwargs)