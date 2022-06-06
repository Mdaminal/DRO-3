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
# from datetime import date
import datetime as dt
import os

# T = 2
# start_month = dt.datetime(2020,7,1)
start_month = dt.datetime(2020,10,1)
# start_month = dt.datetime(2019,8,1)
# start_month = dt.date(2019,8,1)
horas = 1
tax = 0
v0 = 8000
# v0 = 2000  ## caso sem AR
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
conv = 3600*24*30/1000000
cz = 2.592
# cz = 1 ## caso sem AR
# y0 = 3088 ## caso sem AR
y0 = 756
seed=25
ordem = 1;mc = 0;det = 0;flag_2per = 0;fph_flag = 0;simu = 0;cut_selection=0
NC_simu = 8
path = 'D:/Renata/Doutorado/DRO/Dados'
path1 = 'D:/Renata/Doutorado/DRO/'
path2 = 'D:/Renata/Doutorado/DRO/results'
#################### Dados FPH #################################
# linhas = [28,28,28,20]
# usinas = ['Campos Novos','Barra Grande', 'Machadinho', 'Itá']
# coef_fph={}
# for h in range(1,NH+1):
#     coef_fph[h] = pd.read_excel(os.path.join(path,'FPH_reduzida.xlsx'), header = None, sheet_name=usinas[h-1], skiprows = [*range(0,2),*range(linhas[h-1],100+linhas[h-1])], usecols  = [*range(0, 3)])
###############################################################

########################################## Leitura de Dados #######################################################################
file = os.path.join(path,'HIDRELETRICA_v1.csv')## caso sem AR
# file = os.path.join(path,'HIDRELETRICA.csv')
data = pd.read_csv(file,sep=";",index_col=0,header=0) 
hidros = data[(data.USINA_EM_OPERACAO == 1)]
### TERMELÉTRICAS ############################################################################
file = os.path.join(path,'TERMELETRICA_v1.csv')## caso sem AR
# file = os.path.join(path,'TERMELETRICA.csv')## caso sem AR
data = pd.read_csv(file,sep=";",index_col=0,header=0)
termos = data[(data.USINA_EM_OPERACAO == 1)]

########### DEMANDA #################################################
file = os.path.join(path,'DEMANDA_ESTAGIO_v1.csv')
# file = os.path.join(path,'DEMANDA_ESTAGIO.csv')## caso sem AR
demanda = pd.read_csv(file,sep=";",index_col=0,header=0)
demanda['DATA'] = [dt.datetime(demanda['ANO'][c],demanda['MES'][c],1) for c in demanda.index]
demanda = demanda.drop(columns=['ANO','MES'])
demanda = demanda.set_index('DATA')

########### PROCESSO ESTOCASTICO #################################################
# file = os.path.join(path,'VARIAVEL_ALEATORIA_coeficiente_linear_auto_correlacao.csv')
# f = pd.read_csv(file,sep=";",index_col=0,header=0).fillna(0)
file = os.path.join(path,'Afluências.xlsx')
df=pd.read_excel(file, sheet_name=1)
# coef_fph=pd.read_excel('Afluências.xlsx', sheet_name=2)
f0 = df.iloc[:,1]

########### FPH #################################################
## FPH SPT
# file = os.path.join(path,'Sistema Fredo/HIDRELETRICA_FPH.csv')
# fph = pd.read_csv(file,sep=";",index_col=0,header=0)
## FPH Fredo
# file = os.path.join(path,'FUNCAO_PRODUCAO_HIDRELETRICA_FPH1.csv')
# fph = pd.read_csv(file,sep=";",index_col=0,header=0)
####### produtividade constante ##################################
file = os.path.join(path,'produtividade_v1.csv') ## caso  sem AR
# file = os.path.join(path,'produtividade.csv') ## caso  sem AR
prod = pd.read_csv(file,sep=";",index_col=0,header=0)


#### HISTÓRICO AFLUÊNCIA ########################################
file = os.path.join(path,'HISTORICO_AFLUENCIA_MES_ANO.csv')
hist = pd.read_csv(file,sep=";",index_col=0,header=0,usecols=[0,1,2,30])
reservatorios = hidros[(hidros.USINA_FIO_DAGUA==0)].index
#### PATAMAR DE CARGA ###############################################
file = os.path.join(path,'PATAMAR_ESTAGIO.csv')
patamar = pd.read_csv(file,sep=";",index_col=0,header=0)
patamar['DATA'] = [dt.datetime(patamar['ANO'][c],patamar['MES'][c],1) for c in patamar.index]
patamar = patamar.drop(columns=['ANO','MES'])
patamar = patamar.set_index('DATA')
file = os.path.join(path,'DADOS_AttMatrizOperacional_PorPeriodoPorIdPatamarCarga.csv')
duracao_patamar = pd.read_csv(file,sep=";",index_col=0,header=0)
duracao_patamar['Iteradores'] = pd.to_datetime(duracao_patamar['Iteradores'])
file = os.path.join(path,'SUBMERCADO_PATAMAR_DEFICIT_AttMatrizOperacional_PorPeriodoPorIdPatamarCarga.csv')
cd =  pd.read_csv(file,sep=";",index_col=0,header=0)
cd = cd[(cd.AttMatriz == 'custo')]
cd.Iteradores = pd.to_datetime(cd.Iteradores)
######### PERÍODO - ESTÁGIO #############################################
file = os.path.join(path,'DADOS_periodo_estagio.csv')
g = pd.read_csv(file,sep=";",index_col=0,header=0)
g.astype('int32').dtypes
T = int(g.duracao.sum())
#########################################################################
