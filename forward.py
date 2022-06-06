# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:51:22 2020

@author: rehpe
"""
from param import *
import time
def forward(stage,res,stages,cen,g,simu):

    from dateutil.relativedelta import relativedelta
    tempo_aux = {}
    for j in range(1,len(g)+1):
        tempo_aux[j] = {}
        for s in cen:
            tempo_aux[j][s] = {}
            tempo_aux[j][s][0]={} ## 0 indica forward

    j=1;
    for h in hidros.index:
        stage[j]['ctr'][h].RHS = res[h][1].loc[start_month].iloc[0]
        if hidros.loc[h]['USINA_FIO_DAGUA']==0:
            stage[j]['ctrv'][h].RHS = v0

    stage[j][f'm{j}'].update()
    # stage[j][f'm{j}'].reset()
    stage[j][f'm{j}'].optimize()
    
    status = stage[j][f'm{j}'].status
    if status == gb.GRB.INFEASIBLE:
        stage[j][f'm{j}'].computeIIS()
        stage[j][f'm{j}'].write("model.ilp")
    elif status != 2:
        print("não é ótimo")

    # stage[j]['m{0}'.format(j)].write('SDDP_11_simu.lp')
    # stage[j]['m{0}'.format(j)].write('SDDP_11_simu.sol')
    # print('Number of constraints: %d' % stage[j][f'm{j}'].NumConstrs);
    # print('Number of continuous variables: %d' % (stage[j][f'm{j}'].NumVars - stage[j][f'm{j}'].NumBinVars));
    # print('Number of binary variables: %d' % stage[j][f'm{j}'].NumBinVars);
    for s in cen:
        tempo_aux[j][s][0] = stage[j][f'm{j}'].runtime
        for h in hidros.index:
            if hidros.loc[h]['USINA_FIO_DAGUA']==0:
                cen[s][f'vsult{h}'] = [stage[1]['v'][h].x]
            for q in range(ordem):
                cen[s][f'ysult{h}_{q}'] = [stage[1]['y'][h].x]
        # cen[s]['gama'] = [stage[j]['gama'].x]
        # for i in range(abert):
        #     cen[s][f'nu{i}'] = [stage[j]['nu'][i].x]
        if simu==1:
            cen[s]['phsult'] = [sum(stage[1]['q'][h].x for h in hidros.index)]
            cen[s]['gtsult'] = [sum(stage[1]['gt'][i].x for i in termos.index)]
            cen[s]['vsult'] = [sum(stage[1]['v'][h].x for h in hidros[(hidros.USINA_FIO_DAGUA==0)].index)]
            cen[s]['CMO'] = [stage[1]['CMO'].pi/horas]
            # cen[s]['dfc'] = [stage[1]['d'].x]
            # print(sum(stage[1]['gt'][i].x*termos['CUSTO'][i] for i in termos.index))
            cen[s]['custo_simu1'] = copy.deepcopy(stage[j][f'm{j}'].objVal) - copy.deepcopy(sum(stage[j]['nu'][i].x for i in range(abert)))/abert
            # print(copy.deepcopy(stage[j][f'm{j}'].objVal) - copy.deepcopy(sum(stage[j]['nu'][i].x for i in range(abert)))/abert)
            cen[s]['custo_simu'] = sum(stage[1]['gt'][i].x*termos['CUSTO'][i] for i in termos.index)
        cen[s]['custo']={}; cen[s]['custo'][1] = [copy.deepcopy(stage[j][f'm{j}'].objVal)]
    
    for s in cen:
        inp=1
        for j in range(2,stages):
            for h in hidros.index:
                for q in range(ordem):
                    stage[j]['yp'][h,q].UB = cen[s][f'ysult{h}_{q}'][j-2]
                    stage[j]['yp'][h,q].LB = cen[s][f'ysult{h}_{q}'][j-2]
                if hidros.loc[h]['USINA_FIO_DAGUA']==0:
                   stage[j]['ctrv'][h].RHS = cen[s][f'vsult{h}'][j-2]

                stage[j]['ctr'][h].RHS = res[h][cen[s]['y'][inp]+1].loc[start_month+relativedelta(months=+j*g.loc[j][0]-g.loc[j][0])].iloc[0]
            
            inp+=1
            stage[j][f'm{j}'].update()
            # stage[j][f'm{j}'].reset()
            stage[j][f'm{j}'].optimize()
            # print('Number of constraints: %d' % stage[j][f'm{j}'].NumConstrs);
            # print('Number of continuous variables: %d' % (stage[j][f'm{j}'].NumVars));
            tempo_aux[j][s][0] = stage[j][f'm{j}'].runtime
            # tempo_aux.append(time.time()-tempo_ini)
            status = stage[j][f'm{j}'].status
            if status == gb.GRB.INFEASIBLE:
                stage[j][f'm{j}'].computeIIS()
                stage[j][f'm{j}'].write("model.ilp")
            # list_st.append(status)
            # stage[j]['m{0}'.format(j)].write(f'SDDP_{j}{s}_simu.lp')
            # stage[j]['m{0}'.format(j)].write(f'SDDP_{j}{s}_simu.sol')

            for h in hidros.index:
                if hidros.loc[h]['USINA_FIO_DAGUA']==0:
                    cen[s][f'vsult{h}'].append(stage[j]['v'][h].x);
                for q in range(ordem):
                    cen[s][f'ysult{h}_{q}'].append(stage[j]['y'][h].x);
            # if j<T:
                # print(j,cen[s]['gama'])
                # cen[s]['gama'].append(stage[j]['gama'].x)
                # for i in range(abert):
                #     cen[s][f'nu{i}'].append(stage[j]['nu'][i].x)
            if simu == 1:
                cen[s]['CMO'].append(stage[j][f'CMO'].pi/horas)
                # cen[s]['dfc'].append(stage[j]['d'].x)           
                cen[s]['phsult'].append(sum(stage[j]['q'][h].x for h in hidros.index));
                cen[s]['vsult'].append(sum(stage[j]['v'][h].x for h in hidros[(hidros.USINA_FIO_DAGUA==0)].index));
                cen[s]['gtsult'].append(sum(stage[j]['gt'][i].x for i in termos.index));
                if j<T:
                    cen[s]['custo_simu1'] += copy.deepcopy(stage[j][f'm{j}'].objVal) - copy.deepcopy(sum(stage[j]['nu'][i].x for i in range(abert)))/abert
                else:cen[s]['custo_simu1'] += copy.deepcopy(stage[j][f'm{j}'].objVal)
                cen[s]['custo_simu'] += sum(stage[j]['gt'][i].x*termos['CUSTO'][i] for i in termos.index)

    return cen, stage,tempo_aux


