# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:51:22 2020

@author: rehpe
"""
from param import *
import time
def forward(stage,res,stages,cen,g,simu,cuts,o):
    from dateutil.relativedelta import relativedelta
    tempo_aux = {}; prob = {};distr = {};#z={}
    for j in range(1,len(g)):
        # if len(cuts[1][0][0])!=0:
        if len(cuts[1][0])!=0:
            prob[j]=np.array([]); distr[j]=np.array([]);#z[j]={}
            # for a in range(abert):
            #     z[j][a]={}
            #     for i in range(abert):
            #         z[j][a][i]={}

    for j in range(1,len(g)+1):
        tempo_aux[j] = {}
        for s in cen:
            tempo_aux[j][s] = {}
            tempo_aux[j][s][0]={} ## 0 indica forward

    j=1;t=start_month
    if bacia==1:
        for b in hidros.BACIA.unique():
            stage[j]['ctr'][b].RHS = res[b][1].loc[start_month].iloc[0]
    else:
        for h in hidros.index:
            stage[j]['ctr'][h].RHS = res[res['Iteradores']==start_month]["1"][hidros.COMPATIBILIDADE_SPT.loc[h]]
            for q in range(ordem):
                stage[j]['yp'][h,q+1].UB = y_hist.loc[hidros.COMPATIBILIDADE_SPT.loc[h], y_hist.columns == t-relativedelta(months=+q+1)][0]
                stage[j]['yp'][h,q+1].LB = y_hist.loc[hidros.COMPATIBILIDADE_SPT.loc[h], y_hist.columns == t-relativedelta(months=+q+1)][0]
    for h in hidros.index:
        if hidros.loc[h]['USINA_FIO_DAGUA']==0:
            stage[j]['ctrv'][h].RHS = hidros.loc[h]['VOLUME_INICIAL']*(hidros.loc[h]['VOLUME_MAXIMO_OPERACIONAL']-hidros.loc[h]['VOLUME_MINIMO_OPERACIONAL'])/100

    stage[j][f'm{j}'].update()
    # stage[j][f'm{j}'].reset()
    stage[j][f'm{j}'].optimize()
    
    for a in range(abert):
        if len(cuts[j][0][a])!=0:
        # for a in range(abert):
            
            # for i in range(abert):
            #     for k in range(len(cuts[j][i][a])):
            #         z[j][a][i][k] = cuts[j][i][a][k].pi
        
            prob[j] = np.append(prob[j],sum(sum(cuts[j][i][a][k].pi for k in range(len(cuts[j][i][a]))) for i in range(abert)))
            distr[j] = np.append(distr[j], beta*prob[j][a]+(1-beta)/abert) 
            
    status = stage[j][f'm{j}'].status
    if status == gb.GRB.INFEASIBLE:
        stage[j][f'm{j}'].computeIIS()
        stage[j][f'm{j}'].write("model.ilp")
    elif status != 2:
        print("n??o ?? ??timo")

    # stage[j]['m{0}'.format(j)].write('forward_DRO.lp')
    # stage[j]['m{0}'.format(j)].write('forward_DRO.sol')
    # print('Number of constraints: %d' % stage[j][f'm{j}'].NumConstrs);
    # print('Number of continuous variables: %d' % (stage[j][f'm{j}'].NumVars - stage[j][f'm{j}'].NumBinVars));
    # print('Number of binary variables: %d' % stage[j][f'm{j}'].NumBinVars);
    for s in cen:
        tempo_aux[j][s][0] = stage[j][f'm{j}'].runtime
        for h in hidros.index:
            if hidros.loc[h]['USINA_FIO_DAGUA']==0:
                cen[s][f'vsult{h}'] = [stage[1]['v'][h].x]
        if bacia==1:
            for b in hidros.BACIA.unique():
                for q in range(ordem):
                    cen[s][f'ysult{b}_{q}'] = [stage[1]['y'][b].x]
        else:
            for h in hidros.index:
                cen[s][f'ysult{h}'] = [stage[1]['yp'][h,0].x]
        cen[s]['gama'] = [stage[j]['gama'].x]
        for i in range(abert):
            cen[s][f'nu{i}'] = [stage[j]['nu'][i].x]
        if simu==1:
            cen[s]['phsult'] = [sum(stage[1]['q'][h].x for h in hidros.index)]
            cen[s]['gtsult'] = [sum(stage[1]['gt'][i].x for i in termos.index)]
            cen[s]['vsult'] = [sum(stage[1]['v'][h].x for h in hidros[(hidros.USINA_FIO_DAGUA==0)].index)]
            cen[s]['CMO'] = [stage[1]['CMO'].pi/horas]
            cen[s]['dfc'] = [stage[1]['d'].x]
            cen[s]['corte'] = stage[1]['d'].x
            cen[s]['custo_simu'] = sum(stage[1]['gt'][i].x*termos['CUSTO'][i] for i in termos.index)
        cen[s]['custo']={}; cen[s]['custo'][1] = [copy.deepcopy(stage[j][f'm{j}'].objVal)]
    
    for s in cen:
        inp=1;t=start_month+relativedelta(months=+1)
        for j in range(2,stages):
            for h in hidros.index:
                if hidros.loc[h]['USINA_FIO_DAGUA']==0:
                   stage[j]['ctrv'][h].RHS = cen[s][f'vsult{h}'][j-2]
            if bacia==1:
                for b in hidros.BACIA.unique(): 
                    for q in range(ordem):
                        stage[j]['yp'][b,q+1].UB = cen[s][f'ysult{b}_{q}'][j-2]
                        stage[j]['yp'][b,q+1].LB = cen[s][f'ysult{b}_{q}'][j-2]
                         # stage[j]['ctr_y'][b,o].RHS = cen[s][f'ysult{b}_{q}'][j-2]
                    
                    if o>2:
                        soret = np.random.choice(np.arange(1,abert+1),p = distr[j-1])
                        stage[j]['ctr'][b].RHS = res[b][soret].loc[start_month+relativedelta(months=+j*g.loc[j][0]-g.loc[j][0])].iloc[0]
                    else: stage[j]['ctr'][b].RHS = res[b][cen[s]['y'][inp]+1].loc[start_month+relativedelta(months=+j*g.loc[j][0]-g.loc[j][0])].iloc[0]
            else:
                for h in hidros.index: 
                    for q in range(ordem):
                        if t-relativedelta(months=+q+1)<start_month:
                            stage[j]['yp'][h,q+1].UB = y_hist.loc[hidros.COMPATIBILIDADE_SPT.loc[h], y_hist.columns == t-relativedelta(months=+q+1)][0]
                            stage[j]['yp'][h,q+1].LB = y_hist.loc[hidros.COMPATIBILIDADE_SPT.loc[h], y_hist.columns == t-relativedelta(months=+q+1)][0]
                        else:    
                            stage[j]['yp'][h,q+1].UB = cen[s][f'ysult{h}'][j-2-q]
                            stage[j]['yp'][h,q+1].LB = cen[s][f'ysult{h}'][j-2-q]
                    if o>20:
                        soret = np.random.choice(np.arange(1,abert+1),p = distr[j-1])
                        stage[j]['ctr'][h].RHS =  res[res['Iteradores']==t][f"{soret}"][hidros.COMPATIBILIDADE_SPT.loc[h]]
                    else: stage[j]['ctr'][h].RHS = res[res['Iteradores']==t][f"{cen[s]['y'][inp]+1}"][hidros.COMPATIBILIDADE_SPT.loc[h]]
            inp+=1
            stage[j][f'm{j}'].update()
            
            # stage[j][f'm{j}'].reset()
            stage[j][f'm{j}'].optimize()
            if s == list(cen.keys())[0]:
                if j<T:
                    for a in range(abert):
                        if len(cuts[j][0][a])!=0:
                        # for a in range(abert):
                            
                            # for i in range(abert):
                            #     for k in range(len(cuts[j][i][a])):
                            #         z[j][a][i][k] = cuts[j][i][a][k].pi
                        
                            prob[j] = np.append(prob[j],sum(sum(cuts[j][i][a][k].pi for k in range(len(cuts[j][i][a]))) for i in range(abert)))
                            distr[j] = np.append(distr[j],beta*prob[j][a]+(1-beta)/abert) 
            # print('Number of constraints: %d' % stage[j][f'm{j}'].NumConstrs);
            # print('Number of continuous variables: %d' % (stage[j][f'm{j}'].NumVars));
            tempo_aux[j][s][0] = stage[j][f'm{j}'].runtime
            # tempo_aux.append(time.time()-tempo_ini)
            status = stage[j][f'm{j}'].status
            if status == gb.GRB.INFEASIBLE:
                stage[j][f'm{j}'].computeIIS()
                stage[j][f'm{j}'].write("model.ilp")
            # list_st.append(status)
            # stage[j]['m{0}'.format(j)].write(f'forward_{j}{s}_DRO.lp')
            # stage[j]['m{0}'.format(j)].write(f'forward_{j}{s}_DRO.sol')

            for h in hidros.index:
                if hidros.loc[h]['USINA_FIO_DAGUA']==0:
                    cen[s][f'vsult{h}'].append(stage[j]['v'][h].x);
            if bacia == 1:
                for b in hidros.BACIA.unique():
                    for q in range(ordem):
                        cen[s][f'ysult{b}_{q}'].append(stage[j]['y'][b].x);                
            else:
                for h in hidros.index:
                    # cen[s][f'ysult{h}'].append(stage[j]['yn'][h].x);
                    cen[s][f'ysult{h}'].append(stage[j]['yp'][h,0].x);

            if j<T:
                cen[s]['gama'].append(stage[j]['gama'].x)
                for i in range(abert):
                    cen[s][f'nu{i}'].append(stage[j]['nu'][i].x)
            if simu == 1:
                cen[s]['CMO'].append(stage[j][f'CMO'].pi/horas)
                cen[s]['phsult'].append(sum(stage[j]['q'][h].x for h in hidros.index));
                cen[s]['vsult'].append(sum(stage[j]['v'][h].x for h in hidros[(hidros.USINA_FIO_DAGUA==0)].index));
                cen[s]['gtsult'].append(sum(stage[j]['gt'][i].x for i in termos.index));
                cen[s]['custo_simu'] += sum(stage[j]['gt'][i].x*termos['CUSTO'][i] for i in termos.index)
                cen[s]['dfc'].append(stage[j]['d'].x)
                cen[s]['corte'] += stage[j]['d'].x
            t+=relativedelta(months=+1)
    return cen, stage,tempo_aux,prob,distr#,z


