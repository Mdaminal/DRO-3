# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:26:11 2021

@author: rehpe
"""

from param import *
def backward(t,j,tempo_aux,stage,res,cen):

    from dateutil.relativedelta import relativedelta

    PI = {};PIy = {};new_dict = np.array([]);PIs={};vsults = {};custos = {};alfa = {};gama={};ysults={};PIys = {};
    
    
    for s in cen:
        prob = np.zeros(abert)
        custo = np.array([]);
        for i in range(abert):
            for h in hidros.index:
                stage[j]['ctr'][h].RHS = res[h][i+1].loc[t].iloc[0];
                for q in range(ordem):
                    # stage[j]['ctr_y'][h,o].RHS = cen[s][f'ysult{h}_{q}'][j-2]
                    stage[j]['yp'][h,q].UB = cen[s][f'ysult{h}_{q}'][j-2]
                    stage[j]['yp'][h,q].LB = cen[s][f'ysult{h}_{q}'][j-2]

            for h in reservatorios:
                    stage[j]['ctrv'][h].RHS = cen[s][f'vsult{h}'][j-2]
                

            stage[j][f'm{j}'].update()
            stage[j][f'm{j}'].optimize()
            tempo_aux[j][s][1][i] = {}
            tempo_aux[j][s][1][i] = stage[j][f'm{j}'].runtime
            status = stage[j][f'm{j}'].status
            if status == gb.GRB.INFEASIBLE:
                stage[j][f'm{j}'].computeIIS()
                stage[j][f'm{j}'].write("model.ilp")
            elif status != 2:
                print("não é ótimo")
            
            PI[i] = np.array([stage[j][f'ctrv'][h].pi for h in reservatorios])
            PIy[i] = np.array([stage[j]['yp'][h,0].RC for h in hidros.index])
            # PIy[i] = np.array([[stage[j]['yp'][h,q].RC for h in hidros.index] for q in range(ordem)])
            custo = np.append(custo,np.array(stage[j][f'm{j}'].objVal))
        
        PIs[s]={};custos[s]={};PIys[s]={};
        vsults[s] = np.array([cen[s][f'vsult{h}'][j-2] for h in reservatorios]) 
        # print('back',j, custo)
        # print(cen[s][f'ysult{h}_{q}'][j-2])
        ysults[s] = np.array([cen[s][f'ysult1_0'][j-2]]) 
        # ysults[s] = np.array([[cen[s][f'ysult{h}_{q}'][j-2] for h in hidros.index] for q in range(ordem)]) 
        gama[s] = np.array([cen[s]['gama'][j-2]])
        for i in range(abert): 
            PIs[s][i] = PI[i]
            PIys[s][i] = PIy[i]
            # print(PI,ysults[s][q])
            # print(ysults[s],PIy[s][i])
            custos[s][i] = custo[i] - np.inner(vsults[s],PIs[s][i]) - np.inner(ysults[s],PIys[s][i])

            # custos[s][i] = custo[i] - np.inner(vsults[s],PIs[s][i]) - sum(np.inner(ysults[s][q],PIys[s][i][q]) for q in range(ordem))

    return PIs,vsults,custos,tempo_aux,gama,PIys,ysults