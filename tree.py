# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:11:27 2020

@author: rehpe
"""
def tree(T,abert):
    tree = {} # Inicialização da árvore
    arest = {} # Inicialização da aresta
    period = sum([abert**(i) for i in range(T)]) #todos os nós no período de tempo
    nnos = [p for p in range(1,period+1)] #lista com todos os nós
    ##Primeiro ano da árvore
    tree[1]={} #criando dicionário dentro do ano
    #Restante da árvore
    ind = 0
    cont_1 = 0
    for p in nnos[1:]:
        tree[p]={}
    for p in range(2,period+1):
        tup=(nnos[ind],p)
        cont_1+=1
        if cont_1>=abert: 
            ind+=1
            cont_1 = 0
        tree[p]={}
        arest[tup]=None
    # Atribuição dos pais de cada nó. Os pais ficam armazenados na árvore
    for n in range(1,period+1):
        tree[n]['pais']=[]
    for n0, n1 in arest.keys():
        tree[n1]['pais'].append(n0)
    for n in sorted([i for i in tree.keys()]):
        if len(tree[n]['pais'])==0: continue
        tree[n]['pais'] = sorted(tree[n]['pais'] + tree[tree[n]['pais'][0]]['pais'],reverse=True)        
    return tree