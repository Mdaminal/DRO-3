# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:40:12 2019

@author: renat
"""

from param import *
import time
# for g in [2,3,6]:
# def PDDE(seed):

np.random.seed(3)
from sorteio import sorteio
# from inflow_tree_res import inflow_tree_res
from forward_beta import forward
from backward import backward
import os
from modelo import modelo
from dateutil.relativedelta import relativedelta
# def cut_selection_func(num_cut_ant,cortes):
#     a_ma = np.array([0]);corrected_list = np.array([]);
#     for k in range(1,len(cortes)-num_cut_ant):
#         a_ma = np.append(a_ma,cortes[k][-3] + np.inner(cortes[num_cut_ant][1:2],cortes[k][0:len(reservatorios)]) - cortes[k][-2]*cortes[k][-1])
#     # print(a_ma)
#     vetor_v = np.amax(a_ma)
#     for i in range(num_cut_ant,len(cortes)+1):
#         a_ma1 = cortes[i][-3] + np.inner(cortes[num_cut_ant][1:2],cortes[i][0:len(reservatorios)]) - cortes[i][-2]*cortes[i][-1]
#         if a_ma1 > vetor_v:
#             corrected_list = np.append(corrected_list, i)
#     return corrected_list
custo_inf_v = [];itera_v = [];ncon = [];nvar = [];nrc = []
# seed_arvore = [25,100,54,33,2,96,21,68,47,83]
seed_arvore = [25]
# r1 = 0
# for seed in seed_arvore:
# r1 = [0,0.5,1,10,100,1000,10000]
# r1 = [0,1000,10000]
# r1=[0,1000,10000]
r1=[0]
for r in r1:
    rd.seed(seed)
    res = sorteio(T,abert,start_month,hidros,seed)
    
    stages = len(g)
    stage = modelo(g,res,0,r)

    start_program = time.time();
    o=0;time_stop = time.time() - start_program;time_ot = [];o_aux = 0;NC_PDDE=5;cuts = {}
    custo_inf = [];para = 1;tol = [];custo_ant = 0;o=0;tempo = {};num_cut={};restr_corte=0
    PI_s = {}; vsult_s = {}; custo_s = {};custo_sup = []; PI_ys = {};ysult_s = {};num_cut_ant = {};gama_s = {}
    # list_1 = list(product(range(0,abert),repeat = T-1))
    # if selecao_proc == 0:vetor_i = {};vetor_v = {};corrected_list={};cortes[j]={}
    cortes={}
    for j in range(1,len(g)): 
        cuts[j]={};num_cut[j]={};num_cut_ant[j]={}
        for i in range(abert):
            cuts[j][i]={};num_cut[j][i]=0;num_cut_ant[j][i] = 0
            for a in range(abert):cuts[j][i][a]=np.array([])
        cortes[j]={}
        for i in range(abert):cortes[j][i]={}
        # if selecao_proc == 0:
        #     vetor_v[j]={};vetor_i[j]={};cortes[j]={}
        #     for i in range(abert):vetor_v[j][i]={};vetor_i[j][i]={};cortes[j][i]={}
    # scenarios = {}
    # for i in range(len(list_1)):
    #     scenarios[i] = (1,)+list_1[i]
    
    # while time_stop<10:
    for o in range(200):
    # for o in range(5):
        if para==0 and o>1:break
        vsult_s[o] = {}; custo_s[o] = {};ysult_s[o] = {};
        sampled_scenarios = np.random.randint(abert,size=(NC_PDDE,T))
        
        # sortea = rd.sample(list(np.arange(0,len(scenarios))),NC_PDDE)
        # print(sortea)
        cen = {}
        for s in range(NC_PDDE):
                cen[s]={};
                # cen[s]['y'] = scenarios[s]
                cen[s]['y'] = sampled_scenarios[s]
                # cen[s]['y'] = scenarios[sortea[s]]
        if o>=3:
            # r=r1
            for j in range(1,len(g)):
                for i in range(abert):
                    stage[j]['nu'][i].LB = -gb.GRB.INFINITY
                    # stage[j]['gama'].Obj = r
        # runs a forward phase
        cen, stage,tempo_aux,prob,distr = forward(stage,res,stages,cen,g,det,cuts,o)
    
        ## lower bound calculation
    
        # if rank ==0:
    
        custo_inf.append(abs(cen[0]['custo'][1][0]))
            
        tol.append((abs(cen[0]['custo'][1][0])-custo_ant)/(abs(cen[0]['custo'][1][0])+0.00000001))
    
        custo_ant = copy.deepcopy(abs(cen[0]['custo'][1][0]))
        if o>3:
            if (sum(tol[-5:])/5) <= 0.00001: para=0 
            
            # if (sum(tol[-5:])/5) <= 0.00001: para=0 
        # Bachward
    
        for j in range(1,len(g)+1):
            for s in cen:
                tempo_aux[j][s][1]={} ## 1 indica backward
        
        PI_s[o]={};PI_ys[o]={};custo_s[o]={};vsult_s[o] = {};ysult_s[o] = {};gama_s[o]={}
        for h in hidros.index: PI_s[o][h]={};PI_ys[o][h]={};vsult_s[o][h]={};ysult_s[o][h]={}
        t=start_month+relativedelta(months=T);
        for aux in range(1,len(g)):
            j = stages - aux + 1; t-=relativedelta(months=1);     
            custo_s[o][j-1]={};gama_s[o][j-1]={};
            for h in hidros.index:PI_s[o][h][j-1] = {};PI_ys[o][h][j-1] = {};vsult_s[o][h][j-1]={};ysult_s[o][h][j-1]={}
            PIs,vsults,custos,tempo_aux,gama,PIys,ysults =  backward(t,j,tempo_aux,stage,res,cen)
            for s in cen:
                for i in range(abert):
                    for a in range(abert):
                        num_cut[j-1][i]+=1
                        cortes[j-1][i][num_cut[j-1][i]] = np.hstack((PIs[s][a],vsults[s],custos[s][a],gama[s],np.linalg.norm(res[h][i+1].loc[t]-res[h][a+1].loc[t]),a))
                #     PI_s[o][h][j-1][s]={};custo_s[o][j-1][s]={};gama_s[o][j-1][s]={}
                #     for q in range(ordem):PI_ys[o][h][j-1][s]={};
                # for i in range(abert):
                #      custo_s[o][j-1][s][i] = custos[s][i]
                #      gama_s[o][j-1][s][i] = gama[s][0]
                #      for h in hidros.index:
                #          PI_s[o][h][j-1][s][i] = PIs[s][i][h-1]
                #          for q in range(ordem):
                #              PI_ys[o][h][j-1][s][i] = PIy[s][i][h-1]
                # for q in range(ordem): ysult_s[o][h][j-1][s] = cen[s][f'ysult{h}_{q}'][j-2]
                # vsult_s[o][h][j-1][s] = cen[s][f'vsult{h}'][j-2]             
                  
                for i in range(abert):
                    a_ma = np.array([]);corrected_list = []
                    if o>0:
                        for k in range(num_cut_ant[j-1][i]+1,len(cortes[j-1][i])+1):
                            a_ma = np.append(a_ma,cortes[j-1][i][k][-4] + np.inner(vsults[s],cortes[j-1][i][k][0:len(reservatorios)]) - cortes[j-1][i][k][-3]*cortes[j-1][i][k][-2])
                            # print(cortes[j-1][i][k][-4],vsults[s],cortes[j-1][i][k][0:len(reservatorios)], cortes[j-1][i][k][-3],cortes[j-1][i][k][-2],cortes[j-1][i][k][-4] + np.inner(vsults[s],cortes[j-1][i][k][0:len(reservatorios)]) - cortes[j-1][i][k][-3]*cortes[j-1][i][k][-2])
                        vetor_v = np.amax(a_ma)
                        vetor_i = np.argmax(a_ma)+num_cut_ant[j-1][i]+1
                        # print(cen[s][f'nu{i}'][j-2],vetor_v,vetor_i, corrected_list)
                        if vetor_v>cen[s][f'nu{i}'][j-2]: corrected_list=[vetor_i]
                    else: corrected_list = np.arange(num_cut_ant[j-1][i]+1,len(cortes[j-1][i])+1)
                    # print(j,i,a_ma,np.arange(num_cut_ant[j-1][i]+1,len(cortes[j-1][i])+1),corrected_list)
                    for k in corrected_list:
                        # print(cortes[j-1][i][k][-4])
                        cuts[j-1][i][int(cortes[j-1][i][k][-1])] = np.append(cuts[j-1][i][int(cortes[j-1][i][k][-1])],stage[j-1][f'm{j-1}'].addConstr(cortes[j-1][i][k][-2]*stage[j-1]['gama'] + stage[j-1]['nu'][i]>= \
                                          cortes[j-1][i][k][-4] + gb.quicksum(stage[j-1]['v'][h]*cortes[j-1][i][k][h-1] for h in reservatorios)))
                        if (j-1)==1:restr_corte+=1;
                    num_cut_ant[j-1][i] = num_cut[j-1][i]
                stage[j-1][f'm{j-1}'].update()
                # stage[j-1][f'm{j-1}'].write(f'D:/Renata/Doutorado/DRO/modelo_{j-1}{s}_{cut_selection}.lp')
        time_ot.append(time.time() - start_program)    
        time_stop = time.time() - start_program;o+=1;o_aux +=1
    end_program = time.time() - start_program;

    
    t1=pd.DataFrame([end_program], columns = ['tempo total'])
    t2=pd.DataFrame([o], columns = ['iteracoes'])
    t3=pd.DataFrame(custo_inf, columns = ['limite_inferior'])
    t4=pd.DataFrame(time_ot, columns = ['tempo por iteracao'])
    t5=pd.DataFrame(tol, columns = ['tolerancia'])
    t = pd.concat([t1,t2,t3,t4,t5],axis=1)
    t.to_csv(os.path.join(path1,f'evolution_{stages}_{seed}_{r}_{beta}.csv'),sep = ';')

    df = {'j':[],'i':[],'a':[],'custo':[],'distancia':[]}
    for h in hidros.index:
          df[f'PI{h}']=[]
          # df[f'PIy{h}']=[]
          df[f'vsult{h}']=[]
          # df[f'ysult{h}']=[]

    for j in cortes:
        for i in range(abert):
            for k in cortes[j][i]:
              # df['o'].append(k)
              df['j'].append(j)
              # df['s'].append(s)
              df['i'].append(i)
              # df['tempo'].append(time_ot[k])
              df['a'].append(cortes[j][i][k][-1])
              df['custo'].append(cortes[j][i][k][-4])
              df['distancia'].append(cortes[j][i][k][-2])
              for h in hidros.index:
                  df[f'PI{h}'].append(cortes[j][i][k][0])
                  # if ordem>=1:
                  #     df[f'PIy{h}'].append(PI_ys[k][h][j][s][i])
                  #     df[f'ysult{h}'].append(ysult_s[k][h][j][s])
                  # else:
                  #     df[f'PIy{h}'].append(0)
                  #     df[f'ysult{h}'].append(0)
                  df[f'vsult{h}'].append(cortes[j][i][k][h])
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(path1,f'cortes_DRO_{stages}_{seed}_{r}_{beta}.csv'),sep = ';')
    custo_inf_v.append(custo_inf[-1]);
    itera_v.append(o);
    ncon.append(stage[1]['m1'].NumConstrs);
    nvar.append(stage[1]['m1'].NumVars);
    nrc.append(restr_corte)
    # if o>1:
    #     df = {'j':[],'a':[],'k':[],'i':[],'z':[]}
    #     for j in range(1,len(g)):
    #         for a in range(abert):
    #             for i in range(abert):
    #                 for k in range(len(z[j][a][i])):
                    
    #                 # for i in range(1,len(z[j][a][k])):
    #                     df['j'].append(j)
    #                     df['a'].append(a)
    #                     df['k'].append(k)
    #                     df['i'].append(i)
    #                     df['z'].append(z[j][a][i][k]) 
    #     df = pd.DataFrame(df)
    #     df.to_csv(os.path.join(path1,f'z_{stages}_{seed}_{r}.csv'),sep = ';')
t1 = pd.DataFrame(custo_inf_v,columns = ['LI (R$)'])
t2 = pd.DataFrame(itera_v,columns = ['Iteracao'])
t3 = pd.DataFrame(ncon,columns = ['NRST'])
t4 = pd.DataFrame(nvar,columns = ['NVAR'])
t5 = pd.DataFrame(nrc,columns = ['NCORTE'])
t = pd.concat([t1,t2,t3,t4,t5],axis=1)
t.to_csv(os.path.join(path1,f'otimi_DRO_{stages}_{seed}_{beta}.csv'),sep = ';')
