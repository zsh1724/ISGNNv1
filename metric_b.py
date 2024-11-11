import torch
from torch_geometric.data import DataLoader
# import scipy.sparse
# import scipy.sparse as sp
# from collections import Counter
import numpy as np
import itertools
import sys
# from csv import writer

# from sklearn.cluster import SpectralClustering KMeans

from sklearn.cluster import SpectralClustering
from sklearn.utils import check_array
import math
import random



max1 = sys.maxsize


import numpy as np
from sklearn.cluster import SpectralClustering



def CSE_cut(adj_matrix,steps,gn):
    degrees = np.sum(adj_matrix, axis=1)
    segmented_graphs = []
    saved_graphs = []  

    for step in range(steps):
        if step == 0:
            max_degree = np.max(degrees)
            max_degree_nodes = np.where(degrees == max_degree)[0]
            start_node = random.choice(max_degree_nodes) 
            neighbors = np.nonzero(adj_matrix[start_node])[0]  
            
            A1 = set([start_node] + list(neighbors))
            A2 = A1.copy()
            A1D=np.array([x for x in A1])
            adjA1=adj_matrix[A1D]
            DEG1=np.sum(adjA1, axis=0)
            while True:
                degree_one_nodes = [node for node in A1 if DEG1[node] == 1]
                if not degree_one_nodes:
                    break
                A2.difference_update(degree_one_nodes)
                DEG1[list(degree_one_nodes)] = 0
            if len(A2)==1:
                A2 = A1.copy()
            segmented_graphs.extend([A1, A2])

            
        else:
            prev_graphs = segmented_graphs[-2:]  
            A_prev = prev_graphs[0].copy()
            for node in prev_graphs[0]:
                neighbors = np.nonzero(adj_matrix[node])[0]
                A_prev.update(neighbors)
            A_prev2 = A_prev.copy()
            APD=np.array([x for x in A_prev])
            adjAP=adj_matrix[APD]
            DEGP=np.sum(adjAP, axis=0)
            degree_one_nodes = []
            while True:
                degree_one_nodes = [node for node in A_prev2 if DEGP[node] == 1]
                if not degree_one_nodes:
                    break
                A_prev2.difference_update(degree_one_nodes)
                if len(A_prev2)<=2:
                    A_prev2=[]
                    break
                degrees[list(degree_one_nodes)] = 0
            if len(A_prev)<gn:
                segmented_graphs.extend([A_prev])
            if len(A_prev2)<gn:
                segmented_graphs.extend([A_prev2])
            if len(A_prev)>gn:
                break
            if len(segmented_graphs[-1])>gn:
                break
    saved_graphs.extend(segmented_graphs)
    new_list = [saved_graphs[0]] + saved_graphs[1::2]
    if type(new_list)==set:
        ctg = [list(new_list)]
    else:
        ctg = [list(s) for s in new_list]
    new_listt = [sublist for sublist in ctg if len(sublist)<gn]
    if not new_listt:
        min_len = min(len(sublist) for sublist in ctg)
        new_listt = [sublist for sublist in ctg if len(sublist) == min_len]
    if len(new_list)!=1:
        unique_list = []
        seen = set()
        for sublist in new_list:
            tuple_sublist = tuple(sublist)
            if tuple_sublist not in seen:
                unique_list.append(list(tuple_sublist))
                seen.add(tuple_sublist)
    else:
        unique_list = []
        unique_list=new_list[0]
    filtered_list = [sublist for sublist in unique_list if sublist]
    return filtered_list

def remove_edges(A, subgraphs):
    N = A.shape[0]
    new_A = np.zeros((N, N))
    if len(subgraphs)==1:
        for i in subgraphs:
            for j in subgraphs:
                new_A[i, j] = A[i, j]
    else:
        for subgraph in subgraphs:
            for i in subgraph:
                for j in subgraph:
                    new_A[i, j] = A[i, j]
    return new_A





def Dijkstra(G, start):

    start = start - 1
    inf = float('inf')
    node_num = len(G)

    visited = [0] * node_num

    dis = {node: G[start][node] for node in range(node_num)}

    parents = {node: -1 for node in range(node_num)}

    visited[start] = 1

    last_point = start

    for i in range(node_num - 1):

        min_dis = inf
        for j in range(node_num):
            if visited[j] == 0 and dis[j] < min_dis:
                min_dis = dis[j]

                last_point = j

        visited[last_point] = 1

        if i == 0:
            parents[last_point] = start + 1
        for k in range(node_num):
            if G[last_point][k] < inf and dis[k] > dis[last_point] + G[last_point][k]:

                dis[k] = dis[last_point] + G[last_point][k]
                parents[k] = last_point + 1

    return dis


def get_metric_basis(edge_index,num_nodes,steps,un):

    edge_index = edge_index.numpy()

    row = edge_index[0]
    col = edge_index[1]
    vertices_number = num_nodes
    am = np.zeros((num_nodes,num_nodes))
    for i in range(len(row)):
        am[row[i]][col[i]] = 1  
    if vertices_number>2:
        ctg = CSE_cut(am,steps,un)
        new_A = remove_edges(am, ctg)
        am =  new_A
        ls = np.eye(vertices_number)
        am[am==0] = -1
        adjacency_matrix=am+ls
        inf=float('inf')
        adjacency_matrix[adjacency_matrix==-1]=inf  
        d_m=np.zeros((vertices_number,vertices_number))
        for i in range(1,num_nodes+1):
            dis = Dijkstra(adjacency_matrix,i)
            for j in range(num_nodes):
                d_m[i-1][j]=dis[j]
        ad_num_nodes=0
        ad_edges = 0
        ad_hyper_node_list = []
        ad_node_list =[]
        node_list=[]
        for gni in ctg:
            lt=[]
            d_m1 = d_m[:,gni]
            vertices_number = len(gni)
            l1 = [ii for ii in range(vertices_number)]
            for i in range(1,vertices_number+1):
                combinations = []
                combinations.extend(list(itertools.combinations(gni, i)))
                for ii in range(len(combinations)):
                    l2=list(combinations[ii])
                    dm_l=d_m1[l2,:]
                    unique_rows = np.unique(dm_l, axis=1)
                    if unique_rows.shape[1]==len(gni):
                        lt.append(l2)
                if lt!=[]:
                    break
            if type(lt) ==  list:
                node_list=np.concatenate(lt)
                new_array_list = []
                value = num_nodes+ad_num_nodes
                for arr in lt:
                    new_array = np.full_like(arr, value, dtype=np.int64)
                    new_array_list.append(new_array)
                    value += 1
                hyper_node_list=np.concatenate(new_array_list)
                a_num_nodes =  value - num_nodes
                edges = sum([len(al) for al in lt])
                lt=[]
            ad_num_nodes=0
            ad_num_nodes=ad_num_nodes+a_num_nodes
            ad_edges = ad_edges + edges
            ad_node_list = np.concatenate((ad_node_list,node_list))
            ad_hyper_node_list = np.concatenate((ad_hyper_node_list,hyper_node_list))
    else:
        ad_hyper_node_list = []
        ad_node_list =[]
        ad_edges = 1
        a_num_nodes = 1 
        ad_node_list = np.array([0])
        ad_hyper_node_list=np.array([1])
    return ad_edges,a_num_nodes,(torch.IntTensor(ad_node_list),torch.IntTensor(ad_hyper_node_list))

    

