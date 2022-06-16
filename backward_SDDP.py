# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:26:11 2021

@author: rehpe
"""

from param import *
def backward(t,j,tempo_aux,stage,res,cen):

    from dateutil.relativedelta import relativedelta

    PI = {};PIy = {};new_dict = np.array([]);PIs={};vsults = {};custos = {};alfa = {};ysults={};PIys = {};
    for s in cen:
        ysults[s]={};
        prob = np.zeros(abert)
        custo = np.array([]);
        for i in range(abert):
            if bacia==1:
                for b in hidros.BACIA.unique():
                    stage[j]['ctr'][b].RHS = res[b][i+1].loc[t].iloc[0];
                    for q in range(ordem):
                        stage[j]['ctr_y'][b,o].RHS = cen[s][f'ysult{b}_{q}'][j-2]
            else:
                for h in hidros.index:
                    stage[j]['ctr'][h].RHS = res[res['Iteradores']==t][f"{i+1}"][hidros.COMPATIBILIDADE_SPT.loc[h]];
                    for q in range(ordem):
                        if t-relativedelta(months=+q+1)<start_month:
                            stage[j]['yp'][h,q+1].UB = y_hist.loc[hidros.COMPATIBILIDADE_SPT.loc[h], y_hist.columns == t-relativedelta(months=+q+1)][0]
                            stage[j]['yp'][h,q+1].LB = y_hist.loc[hidros.COMPATIBILIDADE_SPT.loc[h], y_hist.columns == t-relativedelta(months=+q+1)][0]
                        else:    
                            stage[j]['yp'][h,q+1].UB = cen[s][f'ysult{h}'][j-2-q]
                            stage[j]['yp'][h,q+1].LB = cen[s][f'ysult{h}'][j-2-q]
            for h in reservatorios:
                    stage[j]['ctrv'][h].RHS = cen[s][f'vsult{h}'][j-2]
                

            stage[j][f'm{j}'].update()
            stage[j][f'm{j}'].optimize()
            # stage[j][f'm{j}'].write(f'SDDP_back_{j}{i}.lp')
            # stage[j][f'm{j}'].write(f'SDDP_back_{j}{i}.sol')
            tempo_aux[j][s][1][i] = {}
            tempo_aux[j][s][1][i] = stage[j][f'm{j}'].runtime
            status = stage[j][f'm{j}'].status
            if status == gb.GRB.INFEASIBLE:
                stage[j][f'm{j}'].computeIIS()
                stage[j][f'm{j}'].write("model.ilp")
            elif status != 2:
                print("não é ótimo")
            
            PI[i] = np.array([stage[j][f'ctrv'][h].pi for h in reservatorios])
            PIy[i] = np.array([stage[j]['yp'][h,q+1].RC for q in range(ordem) for h in hidros.index])
            custo = np.append(custo,np.array(stage[j][f'm{j}'].objVal))
        
        PIs[s]={};custos[s]={};PIys[s]={};
        vsults[s] = np.array([cen[s][f'vsult{h}'][j-2] for h in reservatorios]) 
        ysults[s] = np.array([cen[s][f'ysult{h}'][j-2-q] if (j-2-q)>=0 else y_hist.loc[hidros.COMPATIBILIDADE_SPT.loc[h], y_hist.columns == t-relativedelta(months=+q+1)][0] for q in range(ordem) for h in hidros.index]) 

        for i in range(abert): 
            PIys[s][i] = PIy[i]
            PIs[s][i] = PI[i]
            custos[s][i] = custo[i] - np.inner(vsults[s],PIs[s][i])- sum(np.inner(ysults[s][q*len(hidros.index):(q+1)*len(hidros.index)],PIys[s][i][q*len(hidros.index):(q+1)*len(hidros.index)]) for q in range(ordem))
    # print(j,PIys)
    return PIs,vsults,custos,PIys,ysults,tempo_aux