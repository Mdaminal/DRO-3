# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:40:12 2019

@author: renat
"""

############################################################################
#Programa com cortes separados por j
############################################################################
import os
import time
from dateutil.relativedelta import relativedelta
from param import *
from resultados_simu import resultados
from forward import forward
from backward import backward
from modelo import modelo
# from sorteio_big_system import sorteio
# from inflow_tree_res import inflow_tree_res
# from modelo_big_system import modelo


np.random.seed(3)
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
seed_arvore = [30]
# r1 = 0
rd.seed(seed)
# if bacia==1:
#     res = sorteio(T,abert,start_month,hidros,seed)
# else:
#     file = os.path.join(path,'ProcessoEstocasticoHidrologico/VARIAVEL_ALEATORIA_residuo_espaco_amostral.csv')
#     res = pd.read_csv(file,sep=";",index_col=0,header=0)
#     res.Iteradores = pd.to_datetime(res.Iteradores)
#     # res = res[(res.index.isin(hidros.COMPATIBILIDADE_SPT.values))]
# matrix_dist = np.zeros((abert,abert,T))
# stages = len(g)
# t=start_month
# for j in range(stages):
#     for i in range(abert):
#         for a in range(abert):
#             matrix_dist[i,a,j]=np.linalg.norm([res[res['Iteradores']==t][f"{i+1}"][hidros.COMPATIBILIDADE_SPT.loc[h]]-res[res['Iteradores']==t][f"{a+1}"][hidros.COMPATIBILIDADE_SPT.loc[h]] for h in hidros.index])
#     t+=relativedelta(months=+1)
# print(matrix_dist)    
# r1 = [0,0.5,1,10,100,1000,10000]
# r1 = [0,100,1000]
r1=[0]
for r in r1:
    stage = modelo(g,0,r)
    start_program = time.time();
    o=0;time_stop = time.time() - start_program;time_ot = [];NC_PDDE=1;cuts = {};o=0;
    custo_inf = [];para = 1;tol = [];custo_ant = 0;tempo = {};num_cut={};restr_corte=0;#rest={}
    num_cut_ant = {};num_cut_add = {};tempo_selecao_violation={};tempo_mod_add_cut={};
    # PI_s = {}; vsult_s = {}; custo_s = {};custo_sup = []; PI_ys = {};ysult_s = {};num_cut_ant = {};gama_s = {}
    # list_1 = list(product(range(0,abert),repeat = T-1))
    # if selecao_proc == 0:vetor_i = {};vetor_v = {};corrected_list={};cortes[j]={}
    cortes={};cortes_add = {}
    for j in range(1,len(g)): 
        cuts[j]={};num_cut[j]={};num_cut_ant[j]={};num_cut_add[j]={};#rest[j-1]={}
        for i in range(abert):
            cuts[j][i]={};num_cut[j][i]=0;num_cut_ant[j][i] = 0;num_cut_add[j][i]={}
            for a in range(abert):cuts[j][i][a]=np.array([]);num_cut_add[j][i][a]=0
        cortes[j]={};cortes_add[j] = {}
        for i in range(abert):cortes[j][i]={};cortes_add[j][i] = {}
        # if selecao_proc == 0:
        #     vetor_v[j]={};vetor_i[j]={};cortes[j]={}
        #     for i in range(abert):vetor_v[j][i]={};vetor_i[j][i]={};cortes[j][i]={}
    # scenarios = {}
    # for i in range(len(list_1)):
    #     scenarios[i] = (1,)+list_1[i]
    
    # while time_stop<0:
    for o in range(50):
        # if para==0 and o>1:break
        # vsult_s[o] = {}; custo_s[o] = {};ysult_s[o] = {};
        sampled_scenarios = np.random.randint(abert,size=(NC_PDDE,T))
        
        # sortea = rd.sample(list(np.arange(0,len(scenarios))),NC_PDDE)
        # print(sortea)
        cen = {}
        for s in range(NC_PDDE):
                cen[s]={};
                # cen[s]['y'] = scenarios[s]
                cen[s]['y'] = sampled_scenarios[s]
                # cen[s]['y'] = scenarios[sortea[s]]
        if o>=5:
            # r=r1
            for j in range(1,len(g)):
                for i in range(abert):
                    stage[j]['nu'][i].LB = -gb.GRB.INFINITY
                    stage[j]['gama'].Obj = r
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
            PIs,vsults,custos,PIys,ysults,tempo_aux,gama,RHS =  backward(t,j,tempo_aux,stage,res,cen)
            ## 110 jeito 1
            # for i in range(abert):
            #     for a in range(abert):
            #         if num_cut_add[j-1][i][a]>=110:
            #             for con in range(int(num_cut_add[j-1][i][a]*0.1)):
            #                 stage[j-1][f'm{j-1}'].remove(cuts[j-1][i][a][con])
            #                 cuts[j-1][i][a]=np.delete(cuts[j-1][i][a],con)
            #         num_cut_add[j-1][i][a] = len(cuts[j-1][i][a])
            # ## 110 jeito 2
            # if stage[j][f'm{j}'].NumConstrs-rest[j]>=6:
            #     stage[j-1][f'm{j-1}'].remove(stage[j-1][f'm{j-1}'].getConstrs()[rest[j]:rest[j]+2])
                # stage[j-1][f'm{j-1}'].remove(stage[j-1][f'm{j-1}'].getConstrs()[rest[j]:rest[j]+(abert**abert)])
            tempo_selecao_aux = time.time()
            for s in cen:
                for i in range(abert):
                    for a in range(abert):
                        num_cut[j-1][i]+=1
                        if bacia==1:
                            cortes[j-1][i][num_cut[j-1][i]] = np.hstack((PIs[s][a],vsults[s],custos[s][a],gama[s],np.linalg.norm([res[b][i+1].loc[t]-res[b][a+1].loc[t] for b in hidros.BACIA.unique()]),a))
                        else: cortes[j-1][i][num_cut[j-1][i]] = np.hstack((PIs[s][a],vsults[s],PIys[s][a],ysults[s],RHS[s][a],custos[s][a],gama[s],matrix_dist[i,a,j-1],a))
                for i in range(abert):
                    # tempo_selecao_aux = time.time()
                    a_ma = np.array([]);
                    if o>0:
                        for k in range(num_cut_ant[j-1][i]+1,len(cortes[j-1][i])+1):
                            # a_ma = np.append(a_ma,cortes[j-1][i][k][-4] + np.inner(vsults[s],cortes[j-1][i][k][0:len(reservatorios)]) - cortes[j-1][i][k][-3]*cortes[j-1][i][k][-2])
                            a_ma = np.append(a_ma,cortes[j-1][i][k][-4] - cortes[j-1][i][k][-3]*cortes[j-1][i][k][-2])
                        vetor_v = np.amax(a_ma)
                        vetor_i = np.argmax(a_ma)+num_cut_ant[j-1][i]+1
                        if vetor_v>cen[s][f'nu{i}'][j-2]: corrected_list=[vetor_i]
                    else: corrected_list = np.arange(num_cut_ant[j-1][i]+1,len(cortes[j-1][i])+1)
                    # tempo_selecao_violation[o][j][i] = time.time()-tempo_selecao_aux
                    # tempo_selecao_aux2 = time.time()
                    for k in corrected_list:
                        a = int(cortes[j-1][i][k][-1])
                        if len(cuts[j-1][i][a])>=limit_cut:
                            cuts[j-1][i][a][num_cut_add[j-1][i][a]].RHS = cortes[j-1][i][k][-4]- cortes[j-1][i][k][-5]
                            for h in range(len(reservatorios)):
                                stage[j-1][f'm{j-1}'].chgCoeff(cuts[j-1][i][a][num_cut_add[j-1][i][a]],stage[j-1]['v'][reservatorios[h]],-cortes[j-1][i][k][h])
                            for h in hidros.index:
                                for q in range(ordem):
                                    stage[j-1][f'm{j-1}'].chgCoeff(cuts[j-1][i][a][num_cut_add[j-1][i][a]],stage[j-1]['yp'][h,q],-cortes[j-1][i][k][(h-1)+2*len(reservatorios)+q*len(hidros.index)])
                            stage[j-1][f'm{j-1}'].chgCoeff(cuts[j-1][i][a][num_cut_add[j-1][i][a]],stage[j-1]['gama'],cortes[j-1][i][k][-2])
                            num_cut_add[j-1][i][a]+=1
                            if num_cut_add[j-1][i][a] == limit_cut: num_cut_add[j-1][i][a]=0
                        else: 
                            cortes_add[j-1][i]
                            nonZerosv = np.where(np.abs(cortes[j-1][i][k][:len(reservatorios)]) > 0)[0]
                            nonZerosy = {} 
                            for q in range(ordem):
                                nonZerosy[q] = np.where(np.abs(cortes[j-1][i][k][2*len(reservatorios)+q*len(hidros.index):2*len(reservatorios)+(q+1)*len(hidros.index)]) > 0)[0]
                            cuts[j-1][i][a] = np.append(cuts[j-1][i][a],stage[j-1][f'm{j-1}'].addConstr(cortes[j-1][i][k][-2]*stage[j-1]['gama'] + stage[j-1]['nu'][i] - \
                                                  gb.quicksum(stage[j-1]['v'][reservatorios[h]]*cortes[j-1][i][k][h] for h in nonZerosv)-\
                                                  gb.quicksum(gb.quicksum(stage[j-1]['yp'][h+1,q]*cortes[j-1][i][k][h+2*len(reservatorios)+q*len(hidros.index)] for h in nonZerosy[q]) for q in range(ordem))>= \
                                                  cortes[j-1][i][k][-4]- cortes[j-1][i][k][-5]))

                    # tempo_mod_add_cut[o][j][i] = time.time()-tempo_selecao_aux2
                    num_cut_ant[j-1][i] = num_cut[j-1][i]
                    
            tempo_mod_add_cut[o][j] = time.time()-tempo_selecao_aux     
            stage[j-1][f'm{j-1}'].update()
            # stage[j-1][f'm{j-1}'].write(f'D:/Renata/Doutorado/DRO/modelo_{j-1}{s}.lp')
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
    # df.to_csv(os.path.join(path1,f'tempo_selecao_violation_{r}_v2.csv'),sep = ';')
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
    # df.to_csv(os.path.join(path1,f'tempo_mod_add_cut_{r}_v2.csv'),sep = ';')
    df = {'o':[],'j':[],'tempo':[]}
    for k in range(o):
        for j in tempo_mod_add_cut[k]:
            if j==1:continue
            # for i in range(abert):
            df['o'].append(k)
            df['j'].append(j)
            # df['i'].append(i)
            df['tempo'].append(tempo_mod_add_cut[k][j])
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(path1,f'tempo_selecao_{r}.csv'),sep = ';')
    
    t1=pd.DataFrame([end_program], columns = ['tempo total'])
    t2=pd.DataFrame([o], columns = ['iteracoes'])
    t3=pd.DataFrame(custo_inf, columns = ['limite_inferior'])
    t4=pd.DataFrame(time_ot, columns = ['tempo por iteracao'])
    t5=pd.DataFrame(tol, columns = ['tolerancia'])
    t = pd.concat([t1,t2,t3,t4,t5],axis=1)
    t.to_csv(os.path.join(path1,f'evolution_{stages}_{seed}_{r}_{beta}_DRO.csv'),sep = ';')

    df = {'j':[],'i':[],'a':[],'custo':[],'distancia':[]}
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
              df['a'].append(cortes[j][i][k][-1])
              df['custo'].append(cortes[j][i][k][-4])
              df['distancia'].append(cortes[j][i][k][-2])
              for h in range(len(reservatorios)):
                  df[f'PI{reservatorios[h]}'].append(cortes[j][i][k][h])
                  df[f'vsult{reservatorios[h]}'].append(cortes[j][i][k][h+len(reservatorios)-1])
              for h in hidros.index:
                  for q in range(ordem):
                  # if ordem>=1:
                      df[f'PIy{h}{q}'].append(cortes[j][i][k][h+2*len(reservatorios)+q*len(hidros.index)-1])
                      df[f'ysult{h}{q}'].append(cortes[j][i][k][h+2*len(reservatorios)+(ordem+q)*len(hidros.index)-1])
                  # else:
                  #     df[f'PIy{h}'].append(0)
                  #     df[f'ysult{h}'].append(0)
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(path1,f'cortes_{stages}_{seed}_{r}_{beta}_DRO.csv'),sep = ';')
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
    df.to_csv(os.path.join(path1,f'tempo_{stages}_{seed}_{ordem}_{r}_DRO.csv'),sep = ';')

#     print('inicia simulação')
#     o=0
#     simu=1
#     mean_custo_siduru = np.array([])
#     start_program = time.time();
#     stages = len(g)
#     beta=0
#     come_program = time.time();
#     file = os.path.join(path,'ProcessoEstocasticoHidrologico_usina_ous/VARIAVEL_ALEATORIA_residuo_espaco_amostral.csv')
#     # file = os.path.join(path,'ProcessoEstocasticoHidrologico/VARIAVEL_ALEATORIA_residuo_espaco_amostral.csv')
#     res = pd.read_csv(file,sep=";",index_col=0,header=0)
#     res.Iteradores = pd.to_datetime(res.Iteradores)
#     list_1 = list(product(range(0,abert),repeat = T-1))
#     # scenarios = {}
#     # for i in range(len(list_1)):
#     #     scenarios[i] = (0,)+list_1[i]
#     cen = {};
#     for s in range(NC_simu):
#         cen[s]={};
#         cen[s]['y'] = T*[s]
#         # cen[s]['y'] = scenarios[s]
#     cen, stage,tempo_aux,prob,distr = forward(stage,res,stages+1,cen,g,simu,cuts,o)
#     mean_custo = resultados(cen,path1,stages,seed,0,fph_flag,ordem,r)
#     end_come = time.time() - come_program;
#     print(end_come)
#     t1 = pd.DataFrame([mean_custo],columns = ['media'])
#     t = pd.concat([t1],axis=1)
#     t.to_csv(os.path.join(path1,f'result_simu_{stages}_{r}_{seed}_DRO.csv'),sep = ';')
# t1 = pd.DataFrame(custo_inf_v,columns = ['LI (R$)'])
# t2 = pd.DataFrame(itera_v,columns = ['Iteracao'])
# t3 = pd.DataFrame(ncon,columns = ['NRST'])
# t4 = pd.DataFrame(nvar,columns = ['NVAR'])
# t5 = pd.DataFrame(nrc,columns = ['NCORTE'])
# t = pd.concat([t1,t2,t3,t4,t5],axis=1)
# t.to_csv(os.path.join(path1,f'otimi_{stages}_{seed}_{beta}_DRO.csv'),sep = ';')