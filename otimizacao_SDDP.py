# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:40:12 2019

@author: renat
"""

from param import *
import time
import os
# from resultados_simu_small_system import resultados
from resultados_simu import resultados
np.random.seed(3)
# from sorteio_big_system import sorteio
from forward_SDDP  import forward
from modelo_SDDP import modelo
from backward_SDDP  import backward
from dateutil.relativedelta import relativedelta
from mpi4py import MPI
# from modelo_big_system import modelo
from dateutil.relativedelta import relativedelta

custo_inf_v = [];itera_v = [];ncon = [];nvar = [];nrc = []
# seed_arvore = [25,100,54,33,2,96,21,68,47,83]
seed_arvore = [30]
# r1 = 0
# for seed in seed_arvore:
# r1 = [0,0.5,1,10,100,1000,10000]
# r1 = [0,1000]
def selecao(cortes,num_cut,vetor_v,vetor_i):
    a_ma = np.array([]);
    for k in range(1,num_cut+1):
        if k < num_cut:
            aux_vetor = cortes[num_cut][-1] + np.inner(cortes[k][len(reservatorios):2*len(reservatorios)],cortes[num_cut][0:len(reservatorios)])\
                        + sum(np.inner(cortes[k][2*len(reservatorios)+(q+ordem)*len(hidros.index):2*len(reservatorios)+(q+ordem+1)*len(hidros.index)],cortes[num_cut][2*len(reservatorios)+q*len(hidros.index):2*len(reservatorios)+(q+1)*len(hidros.index)]) for q in range(ordem))
            if vetor_v[k]<aux_vetor:
                vetor_v[k]=aux_vetor
                vetor_i[k]=num_cut
        a_ma = np.append(a_ma,cortes[k][-1] + np.inner(cortes[num_cut][len(reservatorios):2*len(reservatorios)],cortes[k][0:len(reservatorios)])\
               + sum(np.inner(cortes[num_cut][2*len(reservatorios)+(q+ordem)*len(hidros.index):2*len(reservatorios)+(q+ordem+1)*len(hidros.index)],cortes[k][2*len(reservatorios)+q*len(hidros.index):2*len(reservatorios)+(q+1)*len(hidros.index)]) for q in range(ordem)))
        vetor_v[num_cut] = np.amax(a_ma)
        vetor_i[num_cut] = np.argmax(a_ma)+1
    corrected_list = set(val for val in vetor_i.values())  
    return corrected_list,vetor_v,vetor_i


rd.seed(seed)
# if bacia==1:
#     res = sorteio(T,abert,start_month,hidros,seed)
# else:
#     file = os.path.join(path,'ProcessoEstocasticoHidrologico/VARIAVEL_ALEATORIA_residuo_espaco_amostral.csv')
#     res = pd.read_csv(file,sep=";",index_col=0,header=0)
#     res.Iteradores = pd.to_datetime(res.Iteradores)
    
stages = len(g)
stage = modelo(g,0)

start_program = time.time();
o=0;time_stop = time.time() - start_program;time_ot = [];NC_PDDE=1;cuts = {};o=0;
custo_inf = [];para = 1;tol = [];custo_ant = 0;tempo = {};num_cut={};restr_corte=0;#rest={}
num_cut_add = {};tempo_selecao_violation={};tempo_mod_add_cut={};
# PI_s = {}; vsult_s = {}; custo_s = {};custo_sup = []; PI_ys = {};ysult_s = {};num_cut_ant = {};gama_s = {}

if cut_selection == 1:vetor_i = {};vetor_v = {};corrected_list={};
cortes={};cortes_add = {}
for j in range(1,len(g)): 
    cuts[j]={};num_cut[j]={};num_cut_add[j]={};#rest[j-1]={}
    for i in range(abert):
        cuts[j][i]=[];num_cut[j][i]=0;num_cut_add[j][i]={}
    cortes[j]={};cortes_add[j] = {}
    if mc==1: 
        for i in range(abert):cortes[j][i]={};cortes_add[j][i] = {}
# list_1 = list(product(range(0,abert),repeat = T-1))
# scenarios = {}
# for i in range(len(list_1)):
#     scenarios[i] = (1,)+list_1[i]

# while time_stop<100:
for o in range(50):
    # if para==0 and o>1:break
    # vsult_s[o] = {}; custo_s[o] = {};ysult_s[o] = {};
    sampled_scenarios = np.random.randint(abert,size=(NC_PDDE,T))
    # print(sampled_scenarios)
    # sortea = rd.sample(list(np.arange(0,len(scenarios))),NC_PDDE)
    # print(sortea)
    cen = {}
    for s in range(NC_PDDE):
            cen[s]={};
            # cen[s]['y'] = scenarios[s]
            cen[s]['y'] = sampled_scenarios[s]
            # cen[s]['y'] = scenarios[sortea[s]]
    # runs a forward phase
    cen, stage,tempo_aux,prob,distr = forward(stage,res,stages,cen,g,det,cuts,o)
    if o==0:
        j=1; rest = stage[j][f'm{j}'].NumConstrs;
        # for j in range(1,len(g)+1):
        #     rest[j] = stage[j][f'm{j}'].NumConstrs;
    ## lower bound calculation

    # if rank ==0:

    custo_inf.append(abs(cen[0]['custo'][1][0]))
        
    tol.append((abs(cen[0]['custo'][1][0])-custo_ant)/(abs(cen[0]['custo'][1][0])+0.00000001))

    custo_ant = copy.deepcopy(abs(cen[0]['custo'][1][0]))
    if o>3:
        if (sum(tol[-10:])/10) <= 0.00001: para=0 
        
        # if (sum(tol[-5:])/5) <= 0.00001: para=0 
    # Bachward
    tempo_selecao_violation[o]={};tempo_mod_add_cut[o]={};
    for j in range(1,len(g)+1):
        tempo_selecao_violation[o][j]={};tempo_mod_add_cut[o][j]={};
        for s in cen:
            tempo_aux[j][s][1]={} ## 1 indica backward

    t=start_month+relativedelta(months=T);
    for aux in range(1,len(g)):
        j = stages - aux + 1; t-=relativedelta(months=1);     
        PIs,vsults,custos,PIys,ysults,tempo_aux =  backward(t,j,tempo_aux,stage,res,cen)
        for s in cen:
            if mc==1:
                for i in range(abert):
                    num_cut[j-1][i]+=1
                    cortes[j-1][i][num_cut[j-1][i]] = np.hstack((PIs[s][i],vsults[s],PIys[s][i],ysults[s],custos[s][i]))

                for i in range(abert):
                    if cut_selection==1:
                        corrected_list,vetor_v[j-1][i],vetor_i[j-1][i] = selecao(cortes[j-1][i],num_cut[j-1][i],vetor_v[j-1][i],vetor_i[j-1][i])
                        
                        ind_cut = 0;
                        for ind in corrected_list:  
                            if (ind_cut+1)<=len(cuts[j-1][i]):
                                cuts[j-1][i][ind_cut].RHS = cortes[j-1][i][ind][-1]
                                for h in range(len(reservatorios)):
                                    stage[j-1][f'm{j-1}'].chgCoeff(cuts[j-1][i][ind_cut],stage[j-1]['v'][reservatorios[h]],-cortes[j-1][i][ind][h])
                                for h in hidros.index:
                                    for q in range(ordem):
                                        stage[j-1][f'm{j-1}'].chgCoeff(cuts[j-1][i][ind_cut],stage[j-1]['yp'][h,q],-cortes[j-1][i][ind][(h-1)+2*len(reservatorios)+q*len(hidros.index)])
                                ind_cut+=1
                            else:
                                nonZerosv = np.where(np.abs(cortes[j-1][i][ind][:len(reservatorios)]) > 0)[0]
                                nonZerosy = {} 
                                for q in range(ordem):
                                    nonZerosy[q] = np.where(np.abs(cortes[j-1][i][ind][2*len(reservatorios)+q*len(hidros.index):2*len(reservatorios)+(q+1)*len(hidros.index)]) > 0)[0]
                                cuts[j-1][i] = np.append(cuts[j-1][i],stage[j-1][f'm{j-1}'].addConstr(stage[j-1]['alfa'][i] - \
                                                gb.quicksum(stage[j-1]['v'][reservatorios[h]]*cortes[j-1][i][ind][h] for h in nonZerosv)-\
                                                gb.quicksum(gb.quicksum(stage[j-1]['yp'][h+1,q]*cortes[j-1][i][ind][h+2*len(reservatorios)+q*len(hidros.index)] for h in nonZerosy[q]) for q in range(ordem))>= \
                                                cortes[j-1][i][ind][-1]))
                                ind_cut +=1
                    else:
                        nonZerosv = np.where(np.abs(cortes[j-1][i][num_cut[j-1][i]][:len(reservatorios)]) > 0)[0]
                        nonZerosy = {} 
                        for q in range(ordem):
                            nonZerosy[q] = np.where(np.abs(cortes[j-1][i][num_cut[j-1][i]][2*len(reservatorios)+q*len(hidros.index):2*len(reservatorios)+(q+1)*len(hidros.index)]) > 0)[0]
                        cuts[j-1][i] = np.append(cuts[j-1][i],stage[j-1][f'm{j-1}'].addConstr(stage[j-1]['alfa'][i] - \
                                        gb.quicksum(stage[j-1]['v'][reservatorios[h]]*cortes[j-1][i][num_cut[j-1][i]][h] for h in nonZerosv)-\
                                        gb.quicksum(gb.quicksum(stage[j-1]['yp'][h+1,q]*cortes[j-1][i][num_cut[j-1][i]][h+2*len(reservatorios)+q*len(hidros.index)] for h in nonZerosy[q]) for q in range(ordem))>= \
                                        cortes[j-1][i][num_cut[j-1][i]][-1]))
            else:

                num_cut[j-1]+=1
                cortes[j-1][num_cut[j-1]] = np.hstack((PIs[s],vsults[s],PIys[s],ysults[s],custos[s]))


                if cut_selection==1:
                    corrected_list,vetor_v[j-1],vetor_i[j-1] = selecao(cortes[j-1],num_cut[j-1],vetor_v[j-1],vetor_i[j-1])
                    
                    ind_cut = 0;
                    for ind in corrected_list:  
                        if (ind_cut+1)<=len(cuts[j-1]):
                            cuts[j-1][ind_cut].RHS = cortes[j-1][ind][-1]
                            for h in range(len(reservatorios)):
                                stage[j-1][f'm{j-1}'].chgCoeff(cuts[j-1][ind_cut],stage[j-1]['v'][reservatorios[h]],-cortes[j-1][ind][h])
                            for h in hidros.index:
                                for q in range(ordem):
                                    stage[j-1][f'm{j-1}'].chgCoeff(cuts[j-1][ind_cut],stage[j-1]['yp'][h,q],-cortes[j-1][ind][(h-1)+2*len(reservatorios)+q*len(hidros.index)])
                            ind_cut+=1
                        else:
                            nonZerosv = np.where(np.abs(cortes[j-1][ind][:len(reservatorios)]) > 0)[0]
                            nonZerosy = {} 
                            for q in range(ordem):
                                nonZerosy[q] = np.where(np.abs(cortes[j-1][ind][2*len(reservatorios)+q*len(hidros.index):2*len(reservatorios)+(q+1)*len(hidros.index)]) > 0)[0]
                            cuts[j-1] = np.append(cuts[j-1],stage[j-1][f'm{j-1}'].addConstr(stage[j-1]['alfa'] - \
                                            gb.quicksum(stage[j-1]['v'][reservatorios[h]]*cortes[j-1][ind][h] for h in nonZerosv)-\
                                            gb.quicksum(gb.quicksum(stage[j-1]['yp'][h+1,q]*cortes[j-1][ind][h+2*len(reservatorios)+q*len(hidros.index)] for h in nonZerosy[q]) for q in range(ordem))>= \
                                            cortes[j-1][ind][-1]))
                            ind_cut +=1
                else:
                        nonZerosv = np.where(np.abs(cortes[j-1][num_cut[j-1]][:len(reservatorios)]) > 0)[0]
                        nonZerosy = {} 
                        for q in range(ordem):
                            nonZerosy[q] = np.where(np.abs(cortes[j-1][num_cut[j-1]][2*len(reservatorios)+q*len(hidros.index):2*len(reservatorios)+(q+1)*len(hidros.index)]) > 0)[0]
                        cuts[j-1][i] = np.append(cuts[j-1],stage[j-1][f'm{j-1}'].addConstr(stage[j-1]['alfa'] - \
                                        gb.quicksum(stage[j-1]['v'][reservatorios[h]]*cortes[j-1][num_cut[j-1]][h] for h in nonZerosv)-\
                                        gb.quicksum(gb.quicksum(stage[j-1]['yp'][h+1,q]*cortes[j-1][num_cut[j-1]][h+2*len(reservatorios)+q*len(hidros.index)] for h in nonZerosy[q]) for q in range(ordem))>= \
                                        cortes[j-1][num_cut[j-1]][-1]))
                


            # for i in range(abert):
            #     nonZerosv = np.where(np.abs(cortes[j-1][i][num_cut[j-1][i]][:len(reservatorios)]) > 0)[0]
            #     nonZerosy = {} 
            #     for q in range(ordem):
            #         nonZerosy[q] = np.where(np.abs(cortes[j-1][i][num_cut[j-1][i]][2*len(reservatorios)+q*len(hidros.index):2*len(reservatorios)+(q+1)*len(hidros.index)]) > 0)[0]
            #     cuts[j-1][i] = np.append(cuts[j-1][i],stage[j-1][f'm{j-1}'].addConstr(stage[j-1]['alfa'][i] - \
            #                     gb.quicksum(stage[j-1]['v'][reservatorios[h]]*cortes[j-1][i][num_cut[j-1][i]][h] for h in nonZerosv)-\
            #                     gb.quicksum(gb.quicksum(stage[j-1]['yp'][h+1,q]*cortes[j-1][i][num_cut[j-1][i]][h+2*len(reservatorios)+q*len(hidros.index)] for h in nonZerosy[q]) for q in range(ordem))>= \
            #                     cortes[j-1][i][num_cut[j-1][i]][-1]))

                
        # tempo_mod_add_cut[o][j] = time.time()-tempo_selecao_aux     
        stage[j-1][f'm{j-1}'].update()
        # stage[j-1][f'm{j-1}'].write(f'D:/Renata/Doutorado/DRO/modelo_PDDE_ss{j-1}{s}.lp')
        # stage[j-1][f'm{j-1}'].write(f'D:/Renata/Doutorado/DRO/modelo_{j-1}{s}_{cut_selection}_{o}.lp')
    time_ot.append(time.time() - start_program)    
    tempo[o] = tempo_aux
    time_stop = time.time() - start_program;o+=1;
    
end_program = time.time() - start_program;
# df = {'o':[],'j':[],'i':[],'tempo':[]}
# for k in range(o):
#     for j in tempo_selecao_violation[k]:
#         if j==1:continue
#         for i in range(abert):
#             df['o'].append(k)
#             df['j'].append(j)
#             df['i'].append(i)
#             df['tempo'].append(tempo_selecao_violation[k][j][i])
# df = pd.DataFrame(df)
# df.to_csv(os.path.join(path1,f'tempo_selecao_violation_{r}.csv'),sep = ';')
# df = {'o':[],'j':[],'i':[],'tempo':[]}
# for k in range(o):
#     for j in tempo_mod_add_cut[k]:
#         if j==1:continue
#         for i in range(abert):
#             df['o'].append(k)
#             df['j'].append(j)
#             df['i'].append(i)
#             df['tempo'].append(tempo_mod_add_cut[k][j][i])
# df = pd.DataFrame(df)
# df.to_csv(os.path.join(path1,f'tempo_mod_add_cut_{r}.csv'),sep = ';')
# df = {'o':[],'j':[],'tempo':[]}
# for k in range(o):
#     for j in tempo_mod_add_cut[k]:
#         if j==1:continue
#         # for i in range(abert):
#         df['o'].append(k)
#         df['j'].append(j)
#         # df['i'].append(i)
#         df['tempo'].append(tempo_mod_add_cut[k][j])
# df = pd.DataFrame(df)
# df.to_csv(os.path.join(path1,f'tempo_selecao_{r}.csv'),sep = ';')

t1=pd.DataFrame([end_program], columns = ['tempo total'])
t2=pd.DataFrame([o], columns = ['iteracoes'])
t3=pd.DataFrame(custo_inf, columns = ['limite_inferior'])
t4=pd.DataFrame(time_ot, columns = ['tempo por iteracao'])
t5=pd.DataFrame(tol, columns = ['tolerancia'])
t = pd.concat([t1,t2,t3,t4,t5],axis=1)
t.to_csv(os.path.join(path1,f'evolution_{stages}_{seed}.csv'),sep = ';')

df = {'j':[],'i':[],'custo':[]}
for h in reservatorios:
      df[f'PI{h}']=[]
      df[f'vsult{h}']=[]
for h in hidros.index:
    for q in range(ordem):
         df[f'PIy{h}{q}']=[]
         df[f'ysult{h}{q}']=[]
# for k in range(o):
for j in cortes:
    for i in range(abert):
        for k in cortes[j][i]:
          # df['o'].append(k)
          df['j'].append(j)
          # df['s'].append(s)
          df['i'].append(i)
          # df['tempo'].append(time_ot[k])
          df['custo'].append(cortes[j][i][k][-1])
          for h in range(len(reservatorios)):
              df[f'PI{reservatorios[h]}'].append(cortes[j][i][k][h])
              df[f'vsult{reservatorios[h]}'].append(cortes[j][i][k][h+len(reservatorios)-1])
          for h in hidros.index:
              for q in range(ordem):
                  df[f'PIy{h}{q}'].append(cortes[j][i][k][h+2*len(reservatorios)+q*len(hidros.index)-1])
                  df[f'ysult{h}{q}'].append(cortes[j][i][k][h+2*len(reservatorios)+(ordem+q)*len(hidros.index)-1])
df = pd.DataFrame(df)
df.to_csv(os.path.join(path1,f'cortes_{stages}_{seed}.csv'),sep = ';')
custo_inf_v.append(custo_inf[-1]);
itera_v.append(o);
ncon.append(stage[1]['m1'].NumConstrs);
nvar.append(stage[1]['m1'].NumVars);
nrc.append(stage[1]['m1'].NumConstrs - rest)
# nrc.append(restr_corte)
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
df = {'o':[],'j':[],'s':[],'fase':[],'abertura':[],'tempo':[]}
for k in range(o):
    for j in tempo[k]:
        for s in tempo[k][j]:
            for fase in [0,1]:
                if fase==1 and j==1: continue
                if fase==0 and j==len(g): continue                    
                if fase == 0:
                    df['o'].append(k)
                    df['j'].append(j)
                    df['s'].append(s)
                    df['fase'].append(fase)
                    df['abertura'].append(0)
                    df['tempo'].append(tempo[k][j][s][fase])
                else:
                    for i in range(abert):
                        df['o'].append(k)
                        df['j'].append(j)
                        df['s'].append(s)
                        df['fase'].append(fase)
                        df['abertura'].append(i)
                        df['tempo'].append(tempo[k][j][s][fase][i])
df = pd.DataFrame(df)
df.to_csv(os.path.join(path1,f'tempo_{stages}_{seed}_{ordem}.csv'),sep = ';')



############# Simulação ###########################################
# simu=1
# # mean_custo_siduru = np.array([])
# start_program = time.time();
# stages = len(g)
# # mean_simu = np.array([]);std_simu = np.array([])
# # for siduru in seed_out:
# come_program = time.time();
# # file = os.path.join(path,'ProcessoEstocasticoHidrologico_ous/VARIAVEL_ALEATORIA_residuo_espaco_amostral.csv')
# file = os.path.join(path,'ProcessoEstocasticoHidrologico/VARIAVEL_ALEATORIA_residuo_espaco_amostral.csv')
# res = pd.read_csv(file,sep=";",index_col=0,header=0)
# res.Iteradores = pd.to_datetime(res.Iteradores)
# list_1 = list(product(range(0,abert),repeat = T-1))
# scenarios = {}
# for i in range(len(list_1)):
#     scenarios[i] = (0,)+list_1[i]
# cen = {};
# for s in range(NC_simu):
#     cen[s]={};
#     # cen[s]['y'] = T*[s]
#     cen[s]['y'] = scenarios[s]
# cen, stage,tempo_aux,prob,distr = forward(stage,res,stages+1,cen,g,simu,cuts,o)
# mean_custo = resultados(cen,path1,stages,seed,0,fph_flag,ordem,0)
# # mean_custo_siduru = np.append(mean_custo_siduru,mean_custo)


# t1 = pd.DataFrame([mean_custo],columns = ['media'])

# t = pd.concat([t1],axis=1)
# t.to_csv(os.path.join(path1,f'result_simu_{stages}_{seed}.csv'),sep = ';')
# t1 = pd.DataFrame(custo_inf_v,columns = ['LI (R$)'])
# t2 = pd.DataFrame(itera_v,columns = ['Iteracao'])
# t3 = pd.DataFrame(ncon,columns = ['NRST'])
# t4 = pd.DataFrame(nvar,columns = ['NVAR'])
# t5 = pd.DataFrame(nrc,columns = ['NCORTE'])
# t = pd.concat([t1,t2,t3,t4,t5],axis=1)
# t.to_csv(os.path.join(path1,f'otimi_DROh_{stages}_{seed}_{beta}.csv'),sep = ';')
