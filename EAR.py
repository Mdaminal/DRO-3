# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 09:14:34 2021

@author: rehpe
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
conversor_vazao_volume = 2.6298 
path = 'D:/Renata/Doutorado/DRO/Dados'
########### hidros ########################################
# file = os.path.join(path,'HIDRELETRICA_v2.csv')
file = os.path.join(path,'HIDRELETRICA_v1.csv')
data = pd.read_csv(file,sep=";",index_col=0,header=0) 
hidros = data[(data.USINA_EM_OPERACAO == 1)]
reservatorios = hidros[(hidros.USINA_FIO_DAGUA==0)].index
###########################################################
########### produtibilidade acumulada #####################
# file = os.path.join(path,'produtividade_acumulada.csv')
# file = os.path.join(path,'produtividade_v1.csv')
# prod_acu = pd.read_csv(file,sep=";",index_col=0,header=0) 
# prod_acu = prod_acu['prod_acum'].values
##################################################
EAR_max = 0
for h in reservatorios:
    EAR_max += ((hidros.loc[h]['VOLUME_MAXIMO_OPERACIONAL']-hidros.loc[h]['VOLUME_MINIMO_OPERACIONAL'])/conversor_vazao_volume)*0.2732
# stages = 60;seed = 30
# NC_simu = 2000
# stages = 3;seed = 25
# NC_simu = 4
# df = {'matrix':np.array([]),'r':np.array([]),'scenario':np.array([]),'periodo':np.array([]),'EAR':np.array([])}
df = {'rd':np.array([]),'scenario':np.array([]),'periodo':np.array([]),'EAR':np.array([]),'CMO':np.array([])}
# path = 'D:/Renata/Doutorado/DRO/matrizes'
path = 'D:/Renata/Doutorado/DRO/resultado_exemplo'
# matrix = ['inverse_correlation','identity','power','accumulated_productivity']
# r1 = [0,50,100];
r1 = [0,100,1000,2000,4000]
# for n in matrix:
for r in r1:
        # file = os.path.join(path,f'{n}/volumes_{r}_0.csv')
        file = os.path.join(path,f'volumes_{r}.csv')
        data_volume = pd.read_csv(file,sep=";",header=0) 
        file = os.path.join(path,f'CMO_{r}.csv')
        data_cmo = pd.read_csv(file,sep=";",header=0) 
        for h in reservatorios:
            data_volume[f'v{h}'] = (data_volume[f'v{h}']/conversor_vazao_volume)*0.2732
        list_columns = data_volume.columns[2:]
        data_volume['EAR'] = data_volume[list_columns].sum(axis=1)/EAR_max
        df['scenario'] = np.append(df['scenario'],data_volume.scenario.values)
        df['periodo'] = np.append(df['periodo'],data_volume.periodo.values)
        df['EAR'] = np.append(df['EAR'],data_volume.EAR.values)
        df['CMO'] = np.append(df['CMO'],data_cmo.CMO.values)
        # df['matrix'] = np.append(df['matrix'],np.array(len(data_volume.EAR)*[f'{n}']))
        df['rd'] = np.append(df['rd'],np.array(len(data_volume.EAR)*[f'{r}']))
df = pd.DataFrame(df)
df.to_csv(os.path.join(path,'EAR.csv'),sep = ';',index=False) 
################# gráfico #############################################################

# path = 'D:/Renata/Doutorado/DRO/matrizes'
file = os.path.join(path,'EAR.csv')
data = pd.read_csv(file,sep=";")
# data = data[(data.r==50)]
ax = sns.boxplot(x="periodo", y="EAR",data=data,hue='rd',showmeans=True, meanprops={"marker":"x", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":"5"}, linewidth = 0.5, flierprops = dict(markerfacecolor = '0.50', markersize = 0.5))
# ax = sns.boxplot(x="periodo", y="EAR",data=data,hue='matrix',showmeans=True, meanprops={"marker":"x", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":"5"}, linewidth = 0.5, flierprops = dict(markerfacecolor = '0.50', markersize = 0.5))
ax.set(xlabel='stage', ylabel='stored energy (%)')

# ax = sns.boxplot(x="periodo", y="CMO",data=data,hue='rd',showmeans=True, meanprops={"marker":"x", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":"5"}, linewidth = 0.5, flierprops = dict(markerfacecolor = '0.50', markersize = 0.5))
# # ax = sns.boxplot(x="periodo", y="EAR",data=data,hue='matrix',showmeans=True, meanprops={"marker":"x", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":"5"}, linewidth = 0.5, flierprops = dict(markerfacecolor = '0.50', markersize = 0.5))
# ax.set(xlabel='stage', ylabel='CMO ($/MWh)')
# ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
# legend_label = ['inverse correlation', 'identity', 'power']
# ax.legend(labels=legend_label)
# n = 0
# for i in legend_label:
#     ax.legend_.texts[n].set_text(i)
#     n += 1
# plt.show(g)



df = {'rd':np.array([]),'scenario':np.array([]),'cost':np.array([])}
stages = 3
path = 'D:/Renata/Doutorado/DRO/resultado_exemplo'
# matrix = ['inverse_correlation','identity','power','accumulated_productivity']
# r1 = [0,50,100];
r1 = [0,100,1000,2000,4000]
# for n in matrix:
for r in r1:
        # file = os.path.join(path,f'{n}/volumes_{r}_0.csv')
        file = os.path.join(path,f'custo_simulação_{stages}_{r}.csv')
        data_custo = pd.read_csv(file,sep=";",header=0) 
        list_columns = data_custo.columns[2:]
        df['scenario'] = np.append(df['scenario'],data_custo.s.values)
        df['cost'] = np.append(df['cost'],data_custo.custo)
        # df['matrix'] = np.append(df['matrix'],np.array(len(data_volume.EAR)*[f'{n}']))
        df['rd'] = np.append(df['rd'],np.array(len(data_custo.values)*[f'{r}']))
df = pd.DataFrame(df)
df.to_csv(os.path.join(path,'custo_simu.csv'),sep = ';',index=False) 
################# gráfico #############################################################
import seaborn as sns
# path = 'D:/Renata/Doutorado/DRO/matrizes'
file = os.path.join(path,'custo_simu.csv')
data = pd.read_csv(file,sep=";")
# data = data[(data.r==50)]
ax = sns.boxplot(x="", y="cost",data=data,hue='rd',showmeans=True, meanprops={"marker":"x", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":"5"}, linewidth = 0.5, flierprops = dict(markerfacecolor = '0.50', markersize = 0.5))
# ax = sns.boxplot(x="periodo", y="EAR",data=data,hue='matrix',showmeans=True, meanprops={"marker":"x", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":"5"}, linewidth = 0.5, flierprops = dict(markerfacecolor = '0.50', markersize = 0.5))
ax.set(xlabel='stage', ylabel='cost $')