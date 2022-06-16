# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:15:43 2020

@author: rehpe
"""
from param import *
 
def modelo(g,t_simu):
    from dateutil.relativedelta import relativedelta
    import datetime as dt
    stages = len(g)
    nos_per=[abert**(i) for i in range(T)]
    nos_per_aux=[];aux = 0
    for i in nos_per:
        aux += i
        nos_per_aux.append(aux) 
    if simu==1: cortes = pd.read_csv(os.path.join(path1,f'cortes_DRO_{stages}_{seed}.csv'),sep = ';',index_col = 0)    
    obj = 0; stage = {};t=start_month
    for j in range(1,len(g)+1):
        stage[j]={}; 
        
        ## Variáveis
        stage[j][f'm{j}'] = gb.Model(f'PDDE_{j}')
        stage[j]['vp'] = stage[j][f'm{j}'].addVars(hidros[(hidros['USINA_FIO_DAGUA']==0)].index,lb=0,vtype=gb.GRB.CONTINUOUS, name='vp')
        if j<T:
            if mc==1:
                stage[j]['alfa'] = stage[j][f'm{j}'].addVars(abert, obj=1/abert,  vtype=gb.GRB.CONTINUOUS, name='alfa')
            else:stage[j]['alfa'] = stage[j][f'm{j}'].addVar(obj=1,  vtype=gb.GRB.CONTINUOUS, name='alfa')
            if simu==1:
                stage[j]['alfa'] = stage[j][f'm{j}'].addVars(abert_ot, obj=1/abert,  vtype=gb.GRB.CONTINUOUS, name='alfa')

        stage[j]['gt'] = stage[j][f'm{j}'].addVars(termos.index,lb=0, ub=termos['POTENCIA_MAXIMA'],obj=termos['CUSTO'], vtype=gb.GRB.CONTINUOUS, name='gt') 
        stage[j]['y'] = stage[j][f'm{j}'].addVars(hidros.index,lb = -gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name='y')  
        stage[j]['yf'] = stage[j][f'm{j}'].addVars(hidros.index,lb=0,obj=5250, vtype=gb.GRB.CONTINUOUS, name='yf')  
        if bacia == 1:
            stage[j]['yn'] = stage[j][f'm{j}'].addVars(hidros.BACIA.unique(),lb=0, vtype=gb.GRB.CONTINUOUS, name='yn') 
            stage[j]['yp'] = stage[j][f'm{j}'].addVars(hidros.BACIA.unique(),ordem,lb=0,vtype=gb.GRB.CONTINUOUS, name='yp')
        else: 
            stage[j]['yp'] = stage[j][f'm{j}'].addVars(hidros.index,ordem+1,lb = -gb.GRB.INFINITY,vtype=gb.GRB.CONTINUOUS, name='yp')
        stage[j]['v'] = stage[j][f'm{j}'].addVars(hidros[(hidros['USINA_FIO_DAGUA']==0)].index,lb=0, ub=hidros[(hidros['USINA_FIO_DAGUA']==0)].VOLUME_MAXIMO_OPERACIONAL-hidros[(hidros['USINA_FIO_DAGUA']==0)].VOLUME_MINIMO_OPERACIONAL, vtype=gb.GRB.CONTINUOUS, name='v')
        stage[j]['s'] = stage[j][f'm{j}'].addVars(hidros.index,lb=0,  vtype=gb.GRB.CONTINUOUS, name='s')
        stage[j]['ph'] = stage[j][f'm{j}'].addVars(hidros.index,lb=0, ub = hidros['POTENCIA_MAXIMA'], vtype=gb.GRB.CONTINUOUS, name='ph')
        stage[j]['q'] = stage[j][f'm{j}'].addVars(hidros.index,lb=0, ub = hidros['TURBINAMENTO_MAXIMO'], vtype=gb.GRB.CONTINUOUS, name='q')      
        stage[j]['d'] = stage[j][f'm{j}'].addVar(lb=0, ub = demanda['1'].loc[t],obj = cd.loc[1][(cd.Iteradores == t)]['1'], vtype=gb.GRB.CONTINUOUS, name='d')      
        
        stage[j]['CMO'] = stage[j][f'm{j}'].addLConstr(gb.quicksum(stage[j][f'gt'][i] for i in termos.index)\
                            + gb.quicksum(stage[j]['ph'][h] for h in hidros.index) + stage[j]['d'] == demanda['1'].loc[t]) 
        
        stage[j][f'm{j}'].addConstrs((stage[j]['ph'][h] - prod['0'][h]*stage[j]['q'][h] == 0 for h in hidros.index))         
        
        stage[j][f'm{j}'].addConstrs((stage[j]['q'][h] + stage[j]['s'][h] - gb.quicksum(stage[j]['q'][jus] + stage[j]['s'][jus]\
                          for jus in hidros.JUSANTE_TURBINAS[hidros.JUSANTE_TURBINAS == h].index.tolist()) - stage[j]['y'][h] - stage[j]['yf'][h] == 0\
                          for h in hidros[hidros['USINA_FIO_DAGUA']==1].index)) 
        
        stage[j]['ctrv'] = stage[j][f'm{j}'].addConstrs((stage[j]['vp'][h] == 0 for h in hidros[hidros['USINA_FIO_DAGUA']==0].index))
       
        stage[j][f'm{j}'].addConstrs((stage[j]['v'][h]/cz + (stage[j]['q'][h] + stage[j]['s'][h] - gb.quicksum(stage[j]['q'][jus]\
                                    + stage[j]['s'][jus] for jus in hidros.JUSANTE_TURBINAS[hidros.JUSANTE_TURBINAS == h].index.tolist()) - stage[j]['y'][h] - stage[j]['yf'][h])\
                                    - stage[j]['vp'][h]/cz  == 0 for h in hidros[hidros['USINA_FIO_DAGUA']==0].index))                        

        if bacia == 1:
            stage[j]['ctr'] = stage[j][f'm{j}'].addConstrs((stage[j]['yn'][b] - gb.quicksum(f[(f.Iteradores==(t-relativedelta(months=+o+1)).month)].loc[h]['{0}'.format(o+1)]*\
                          stage[j]['yp'][b,o] for o in range(ordem))  == 0 for b in hidros.BACIA.unique()))
            stage[j][f'm{j}'].addConstrs((stage[j]['y'][h] - stage[j]['yn'][hidros.loc[h].BACIA]*coef_part[(coef_part['MES']==t.month)][f'{h}'] == 0 for h in hidros.index))
        else:
            stage[j]['ctr'] = stage[j][f'm{j}'].addConstrs((stage[j]['yp'][h,0] - gb.quicksum(f[(f.Iteradores==t)].loc[hidros.COMPATIBILIDADE_SPT.loc[h]][f'{o+1}']*\
                          stage[j]['yp'][h,o+1] for o in range(ordem))  == 0 for h in hidros.index))
            # stage[j]['ctr'] = stage[j][f'm{j}'].addConstrs((stage[j]['yn'][h] - gb.quicksum(f[(f.Iteradores==t)].loc[hidros.COMPATIBILIDADE_SPT.loc[h]][f'{o+1}']*\
            #               stage[j]['yp'][h,o] for o in range(ordem))  == 0 for h in hidros.index))
            # stage[j][f'm{j}'].addConstrs((stage[j]['y'][h] - stage[j]['yn'][h]  ==  -gl.valor.loc[hidros.COMPATIBILIDADE_SPT.loc[h]] for h in hidros.index))
            stage[j][f'm{j}'].addConstrs((stage[j]['y'][h] - stage[j]['yp'][h,0]  ==  -gl.valor.loc[hidros.COMPATIBILIDADE_SPT.loc[h]] for h in hidros.index))

        stage[j][f'm{j}'].update()
        stage[j][f'm{j}'].Params.outputflag=0
        stage[j][f'm{j}'].Params.method=1
        # stage[j][f'm{j}'].write(f'modelo{j}.lp')
        t+=relativedelta(months=+1)
    return stage