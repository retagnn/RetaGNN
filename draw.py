


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
import scipy.sparse as sparse
from scipy.sparse import coo_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt





def Draw_Bipartite_Graph(user,seq_item,attn_value,name,save=True):

 

    if True:
        fig = plt.figure(name)
        entity2ids,index_user,index_item = dict(),0,0
        for i in range(len(user)):
            if user[i] not in entity2ids:
                entity2ids[user[i]] = index_user
                index_user +=1
        for i in range(len(seq_item)):
            if seq_item[i] not in entity2ids:
                entity2ids[seq_item[i]] = index_item
                index_item +=1

        row = [entity2ids[user[i]] for i in range(len(user))]
        col = [entity2ids[seq_item[i]] for i in range(len(seq_item))]
        X_name = [i for i in range(len(set(row)))]
        Y_name = [i+len(set(row)) for i in range(len(set(col)))]
        a_matrix = coo_matrix((attn_value, (row, col))).toarray()
        a_matrix = coo_matrix(a_matrix)
        G = bipartite.from_biadjacency_matrix(a_matrix, create_using=None, 
                                                edge_attribute='weight')
        pos = dict()
        Y_len = int((len(Y_name) - 1)*10)
        X_unit_len = int(Y_len/(len(X_name)+1))
        pos.update((n, (0, (i+1)*X_unit_len)) for i, n in enumerate(X_name))
        pos.update((n, (0.5, i*10)) for i, n in enumerate(Y_name))

        num_edges = G.number_of_edges()
        num_nodes = G.number_of_nodes()
        color_map = []
        for node in G:
            if node < len(set(user)):
                color_map.append('xkcd:red')
            else: 
                color_map.append('xkcd:blue')

        nx.draw(G, pos=pos,#with_labels=True,
                edge_color=attn_value, 
                edge_cmap=plt.get_cmap('rainbow'), 
                node_color=color_map,
                cmap=plt.get_cmap('Reds'))
        plt.savefig('/home/hsucheng/DRS/code/RS_2/graph/draw_test-'+str(name)+'.png')
        plt.close(name)








