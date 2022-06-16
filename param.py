# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:12:41 2020

@author: rehpe
"""
import pandas as pd
import pandas as pd
import gurobipy as gb
import numpy as np
import copy
from heapq import nlargest, nsmallest
import numpy_financial as npf
import math as mt
import timeit
from itertools import product
import random as rd
import datetime as dt
import os
from dateutil.relativedelta import relativedelta

NC_simu = 16
start_month = dt.datetime(2020,1,1)
horas = 1
tax = 0
# v0 = 2000
# r=0
# tax = 0.008
beta=0
abert_ot = 2
abert = 2
risk = 0.5
phi  = 0
prob1 = (1/abert)*(1-phi)
prob2 = (1/abert)*(1-phi)+(1/abert)*(phi/risk)
## system data
SUBS = 1;PAT = 1
# conv = 3600*24*30/1000000
cz = 3600*24*30/1000000
# y0 = 3088
seed=30
ordem = 1;mc = 1;det = 0;flag_2per = 0;fph_flag = 0;simu = 0;cut_selection=0;bacia=0
sort_dist=0;lhs_flag=1;flag_2020=0;fim_percurso = 0;limit_cut = 100

path = 'C:/Users/rehpe/OneDrive/Documentos/GitHub/DRO/'
path2 = 'C:/Users/rehpe/OneDrive/Documentos/GitHub/DRO/sistema_grande'
path1 = 'D:/Renata/Doutorado/DRO/resultado_artigo'
#################### Dados FPH #################################
# linhas = [28,28,28,20]
# usinas = ['Campos Novos','Barra Grande', 'Machadinho', 'Itá']
# coef_fph={}
# for h in range(1,NH+1):
#     coef_fph[h] = pd.read_excel(os.path.join(path,'FPH_reduzida.xlsx'), header = None, sheet_name=usinas[h-1], skiprows = [*range(0,2),*range(linhas[h-1],100+linhas[h-1])], usecols  = [*range(0, 3)])
###############################################################
######### PERÍODO - ESTÁGIO #############################################
file = os.path.join(path2,'DADOS_periodo_estagio.csv')
g = pd.read_csv(file,sep=";",index_col=0,header=0)
g.astype('int32').dtypes
T = int(g.duracao.sum())
#########################################################################
########################################## Leitura de Dados #######################################################################
file = os.path.join(path2,'HIDRELETRICA.csv')
data = pd.read_csv(file,sep=";",index_col=0,header=0) 
hidros = data[(data.USINA_EM_OPERACAO == 1)]
reservatorios = hidros[(hidros.USINA_FIO_DAGUA==0)].index
### TERMELÉTRICAS ############################################################################
file = os.path.join(path2,'TERMELETRICA.csv')
data = pd.read_csv(file,sep=";",index_col=0,header=0)
termos = data[(data.USINA_EM_OPERACAO == 1)]

########### DEMANDA #################################################
file = os.path.join(path2,'DEMANDA_ESTAGIO_v5.csv')
demanda = pd.read_csv(file,sep=";",index_col=0,header=0)
demanda['DATA'] = [dt.datetime(demanda['ANO'][c],demanda['MES'][c],1) for c in demanda.index]
demanda = demanda.drop(columns=['ANO','MES'])
demanda = demanda.set_index('DATA')

########### PROCESSO ESTOCASTICO #################################################
if bacia == 0:
    file = os.path.join(path,f'ProcessoEstocasticoHidrologico_usina/VARIAVEL_ALEATORIA_coeficiente_linear_auto_correlacao.csv')
    f = pd.read_csv(file,sep=";",index_col=0,usecols=[0,2,3],header=0).fillna(0)
    f.Iteradores = pd.to_datetime(f.Iteradores)
    f = f[(f.index.isin(hidros.COMPATIBILIDADE_SPT.values))]
    file = os.path.join(path,f'ProcessoEstocasticoHidrologico_usina/VARIAVEL_ALEATORIA_INTERNA_grau_liberdade.csv')
    gl = pd.read_csv(file,sep=";",index_col=0,header=0)
    gl= gl[(gl.index.isin(hidros.COMPATIBILIDADE_SPT.values))]
    file = os.path.join(path,f'ProcessoEstocasticoHidrologico_usina/VARIAVEL_ALEATORIA_INTERNA_tendencia_temporal.csv')
    y_hist = pd.read_csv(file,sep=";",index_col=0,header=0)
    y_hist.columns = y_hist.columns[:3].tolist() + pd.to_datetime(y_hist.columns[3:]).tolist()
    y_hist = y_hist[(y_hist.index.isin(hidros.COMPATIBILIDADE_SPT.values))]
    # for i in hidros.COMPATIBILIDADE_SPT.values:
    #     y_hist.loc[i,'novo index'] = hidros[hidros.COMPATIBILIDADE_SPT==i].index[0]
    #     gl.loc[i,'novo index'] = hidros[hidros.COMPATIBILIDADE_SPT==i].index[0]
    #     f.loc[i,'novo index'] = hidros[hidros.COMPATIBILIDADE_SPT==i].index[0]
    # y_hist.reset_index(inplace=True)
    # y_hist = y_hist.set_index('novo index')
    # gl.reset_index(inplace=True)
    # gl = gl.set_index('novo index')
    # f.reset_index(inplace=True)
    # f = f.set_index('novo index')
    h_agreg = hidros.index
else:
    file = os.path.join(path,'ProcessoEstocasticoHidrologico_bacia/VARIAVEL_ALEATORIA_coeficiente_linear_auto_correlacao.csv')
    f = pd.read_csv(file,sep=";",index_col=0,header=0).fillna(0)
    f.Iteradores = pd.to_datetime(f.Iteradores)
    file = os.path.join(path,'ProcessoEstocasticoHidrologico_bacia/VARIAVEL_ALEATORIA_INTERNA_grau_liberdade.csv')
    gl = pd.read_csv(file,sep=";",index_col=0,header=0)
    gl_aux = gl.sum(level=0)
    gl = gl.set_index('idVariavelAleatoriaInterna')
    # gl.columns = gl.columns[:2].tolist() + pd.to_datetime(gl.columns[2:]).tolist()
    file = os.path.join(path,'ProcessoEstocasticoHidrologico_bacia/VARIAVEL_ALEATORIA_INTERNA_tendencia_temporal.csv')
    y_hist = pd.read_csv(file,sep=";",index_col=0,header=0)
    y_hist.columns = y_hist.columns[:3].tolist() + pd.to_datetime(y_hist.columns[3:]).tolist()
    y_hist = y_hist.sum(level=0)
    y_hist = y_hist.drop(y_hist.columns[:2], axis=1)
    # h_agreg = hidros.BACIA.unique()
    h_agreg = np.sort(hidros.BACIA.unique())
    for t in y_hist.columns:
        y_hist[t]=y_hist[t]-gl_aux.valor
########### FPH #################################################
## FPH SPT
# file = os.path.join(path,'Sistema Fredo/HIDRELETRICA_FPH.csv')
# fph = pd.read_csv(file,sep=";",index_col=0,header=0)
## FPH Fredo
file = os.path.join(path2,'FUNCAO_PRODUCAO_HIDRELETRICA_FPH1.csv')
fph = pd.read_csv(file,sep=";",index_col=0,header=0)
####### produtividade constante ##################################
file = os.path.join(path2,'produtividade.csv')
prod = pd.read_csv(file,sep=";",index_col=0,header=0)
file = os.path.join(path2,'COEFICIENTE_PARTICIPACAO.csv')
coef_part = pd.read_csv(file,sep=";",index_col=0,header=0)
################# RESÍDUOS ########################################
if bacia == 0:
    file = os.path.join(path,f'ProcessoEstocasticoHidrologico_usina/VARIAVEL_ALEATORIA_residuo_espaco_amostral.csv')
    res = pd.read_csv(file,sep=";",index_col=0,header=0)
    res.Iteradores = pd.to_datetime(res.Iteradores)
    res = res[(res.index.isin(hidros.COMPATIBILIDADE_SPT.values))]
    # for i in hidros.COMPATIBILIDADE_SPT.values:
    #     res.loc[i,'novo index'] = hidros[hidros.COMPATIBILIDADE_SPT==i].index[0]
    # res.reset_index(inplace=True)
    # res = res.set_index('novo index')
else:
    file = os.path.join(path,'ProcessoEstocasticoHidrologico_bacia/VARIAVEL_ALEATORIA_residuo_espaco_amostral.csv')
    res = pd.read_csv(file,sep=";",index_col=0,header=0)
    res.Iteradores = pd.to_datetime(res.Iteradores)
# ################# MATRIZ DISTÂNCIAS ################################
### matriz potência
totalpot = hidros['POTENCIA_MAXIMA'].sum()
pot = np.zeros((len(hidros.index)))
for h in hidros.index:
    pot[h-1] = hidros['POTENCIA_MAXIMA'][h]/totalpot
#######################

########### produtibilidade acumulada #####################
file = os.path.join(path2,'produtividade_acumulada.csv')
prod_acu = pd.read_csv(file,sep=";",index_col=0,header=0) 
prod_acu = prod_acu['prod_acum'].values
##################################################
matrix_dist = np.zeros((abert,abert,T))
M =  np.eye(len(hidros),len(hidros)) #matriz identidade
# M =  inverse_corr # matriz correlação inversa
# M = pot*np.eye(len(hidros),len(hidros)) # matriz potência
# M =  prod_acu*np.eye(len(hidros),len(hidros))
stages = len(g)
t=start_month
j=1;
for j in range(stages):
    # corr_matrix = hist[(hist.index.month==t.month)].corr(method='pearson')
    # inverse_corr = np.linalg.inv(corr_matrix.values)
    # M =  inverse_corr
    for i in range(abert):
        for a in range(abert):
            x = res[res['Iteradores']==t][f"{i+1}"].values-res[res['Iteradores']==t][f"{a+1}"].values
            mat = np.dot(x,M)
            matrix_dist[i,a,j]=np.sqrt(np.dot(mat,x))
    t+=relativedelta(months=+1)

# hist['DATA'] = [dt.datetime(hist['ANO'][c],hist['MES'][c],1) for c in hist.index]
# hist = hist.drop(columns=['ANO','MES'])
# hist = hist.set_index('DATA')
# corr_matrix = hist.corr(method='pearson')
# inverse_corr = pd.DataFrame(np.linalg.pinv(corr_matrix.values), corr_matrix.columns, corr_matrix.index)
#### PATAMAR DE CARGA ###############################################
file = os.path.join(path2,'SUBMERCADO_AttMatrizPremissa_PorPeriodoPorIdPatamarCarga_sempat.csv')
patamar = pd.read_csv(file,sep=";",index_col=0,header=0)
patamar.Iteradores = pd.to_datetime(patamar.Iteradores)
file = os.path.join(path2,'DADOS_AttMatrizOperacional_PorPeriodoPorIdPatamarCarga_sempat.csv')
duracao_patamar = pd.read_csv(file,sep=";",index_col=0,header=0)
duracao_patamar['Iteradores'] = pd.to_datetime(duracao_patamar['Iteradores'])
file = os.path.join(path2,'SUBMERCADO_PATAMAR_DEFICIT_AttMatrizOperacional_PorPeriodoPorIdPatamarCarga.csv')
cd =  pd.read_csv(file,sep=";",index_col=0,header=0)
cd = cd[(cd.AttMatriz == 'custo')]
cd.Iteradores = pd.to_datetime(cd.Iteradores)

