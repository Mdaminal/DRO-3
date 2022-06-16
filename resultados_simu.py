# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 15:17:46 2021

@author: rehpe
"""
import pandas as pd
import os
from param import *
from dateutil.relativedelta import relativedelta
# from param import *
from pandas import ExcelWriter
def resultados(cen,path1,stages,seed,siduru,fph_flag,ordem,r):
    
    writer = ExcelWriter(f'simu_var_{stages}_{r}_{seed}_{siduru}.xlsx')
    corte_carga = []
    for s in range(NC_simu):
        if cen[s]['corte'] != 0:
            corte_carga.append(1)
    corte_carga = sum(corte_carga)/NC_simu
    df = {'corte':[]}
    df['corte'].append(corte_carga)
    df = pd.DataFrame(df)
    #df.to_csv(os.path.join(path1,f'corte_carga_{stages}_{seed}_{ordem}_{fph_flag}_{t}_{siduru}.csv'),sep = ';')
    df.to_excel(writer, sheet_name = 'corte')
    
    df = {'s':[],'custo':[]}
    for s in range(NC_simu):
        df['s'].append(s)
        df['custo'].append(cen[s]['custo_simu'])
    mean_custo = sum(df['custo'])/len(df['custo'])        
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(path1,f'custo_simulação_{stages}_{r}_{seed}_{siduru}.csv'),sep = ';')
    
    # df = {'s':[],'custo':[]}
    # for s in range(NC_simu):
    #     df['s'].append(s)
    #     df['custo'].append(cen[s]['custo_simu1'])
    # mean_custo = sum(df['custo'])/len(df['custo'])        
    # df = pd.DataFrame(df)
    # df.to_csv(os.path.join(path1,f'custo_simulação1_{stages}_{r}_{seed}_{siduru}.csv'),sep = ';')
    
    ########### Segundo jeito ####################
    column_names = list(range(1,T+1))
    index_name = index = range(1,NC_simu+1)
    
    df = {'r':[],'scenario':[],'periodo':[],'phsult':[],'vsult':[],'gsult':[],'CMO':[],'deficit':[]}
    t_aux=start_month;
    for t in range(T):
        for s in range(NC_simu):
            df['periodo'].append(t_aux)
            df['scenario'].append(s)
            df['r'].append(r)
            df['phsult'].append(cen[s]['phsult'][t])
            df['gsult'].append(cen[s]['gtsult'][t])
            df['vsult'].append(cen[s]['vsult'][t])
            df['CMO'].append(cen[s]['CMO'][t])
            df['deficit'].append(cen[s]['dfc'][t])
        t_aux+=relativedelta(months=1);
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(path1,f'variaveis_{r}_{siduru}.csv'),sep = ';',index=None)
    
    
    ## Produção hidrelétrica ####################
    phsult = []
    for s in range(NC_simu):
        phsult.append(cen[s]['phsult'])

    df = pd.DataFrame(phsult, columns = column_names, index = index_name) 
    #df.to_csv(os.path.join(path1,f'ph_{stages}_{seed}_{ordem}_{fph_flag}_{t}_{siduru}.csv'),sep = ';')
    df.to_excel(writer, sheet_name = 'phsult')
    ## Geração termelétrica ####################
    gtsult = []
    for s in range(NC_simu):
        gtsult.append(cen[s]['gtsult'])
    df = pd.DataFrame(gtsult, columns = column_names, index = index_name) 
    #df.to_csv(os.path.join(path1,f'gt_{stages}_{seed}_{ordem}_{fph_flag}_{t}_{siduru}.csv'),sep = ';')
    df.to_excel(writer, sheet_name = 'gsult')
    ## volume ####################
    vsult = []
    for s in range(NC_simu):
        vsult.append(cen[s]['vsult'])
    df = pd.DataFrame(vsult, columns = column_names, index = index_name) 
    #df.to_csv(os.path.join(path1,f'v_{stages}_{seed}_{ordem}_{fph_flag}_{t}_{siduru}.csv'),sep = ';')
    df.to_excel(writer, sheet_name = 'vsult')
    ### CMO ###################################################
    
    cmo = []
    for s in range(NC_simu):
        cmo.append(cen[s]['CMO'])
        #print(cen[s]['CMO'])
    cmo = pd.DataFrame(cmo, columns = column_names, index = index_name) 
    #cmo.to_csv(os.path.join(path1,f'CMO_{stages}_{seed}_{ordem}_{fph_flag}_{t}_{siduru}.csv'),sep = ';')
    cmo.to_excel(writer, sheet_name = 'CMO')
    ### déficit ###################################################
    dfc = []
    for s in range(NC_simu):
        dfc.append(cen[s]['dfc'])
    df = pd.DataFrame(dfc, columns = column_names, index = index_name) 
    # #df.to_csv(os.path.join(path1,f'dfc_{stages}_{seed}_{ordem}_{fph_flag}_{t}_{siduru}.csv'),sep = ';')
    df.to_excel(writer, sheet_name = 'dfc')
    writer.save()
    ### volumes por hidros ###################################################
    # writer = ExcelWriter(f'vsult_{stages}_{r}_{seed}_{siduru}.xlsx')
    # df = {'scenario':[],'periodo':[]}
    # for h in reservatorios:
    #     df[f'v{h}']=[]
    # t_aux=start_month;
    # for t in range(T):
    #     for s in range(NC_simu):
    #         df['periodo'].append(t_aux)
    #         df['scenario'].append(s)
    #         for h in hidros[(hidros.USINA_FIO_DAGUA==0)].index:
    #             df[f'v{h}'].append(cen[s][f'vsult{h}'][t])
    #     t_aux+=relativedelta(months=1);
    # df = pd.DataFrame(df)
    # df.to_csv(os.path.join(path1,f'volumes_{r}_{siduru}.csv'),sep = ';',index=None)
    df = {'Id_Estagio':[],'Periodo':[],'IdHidreletrica':[]}
    for s in range(NC_simu):
        df[f'IdCenario_{s}']=[]
    t_aux=start_month;ind=1
    for t in range(T):
        for h in hidros.index:
            df['Periodo'].append(t_aux)
            df['Id_Estagio'].append(ind)
            df['IdHidreletrica'].append(hidros.COMPATIBILIDADE_SPT.loc[h])
            for s in range(NC_simu):
                if h in hidros[(hidros.USINA_FIO_DAGUA==0)].index:
                    df[f'IdCenario_{s}'].append(cen[s][f'vsult{h}'][t])
                else:df[f'IdCenario_{s}'].append(0)
        t_aux+=relativedelta(months=1);ind+=1
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(path1,'varVF_3_primal.csv'),sep = ';',index=None)
    # writer.save()
    return mean_custo