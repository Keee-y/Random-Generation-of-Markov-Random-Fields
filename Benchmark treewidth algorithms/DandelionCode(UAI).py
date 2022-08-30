# Reference - Caminiti, S., Fusco, E. G., & Petreschi, R. (2010). 
# Bijective linear time coding and decoding for k-trees. 
# Theory of Computing Systems, 46(2), 284-300.

import numpy as np

# Decoding process
# Input: a code (Q, S) in A_n,k
# Output: a k-tree T_n,k
# add pair (0, \varepsilon) to the S -> reverse Generalized Dandelion Code T -> R_n,k -> T_n,k 
# def S2T(): Program 2 Generalized Dandelion Decoding; S -> T
# def cycle(): Program 3 Identify Cycles (auxiliary function)
# def TRnk(): Program 8 Rebuild R_n,k; T -> R_n,k
# def reverse_permutation(): R_n,k -> T_n,k

# Progeam 5 Order Adjeacency Lists
def order_adj(tree):
    # input: 
    #     tree: the adjacency lists of the k-tree
    # output:
    #     sort_tree: for each node, its adjacent neighbors are in increasing order
    sort_tree = {}
    for i in range(1, len(tree) + 1):
        for j in tree[i]:
            if j in sort_tree.keys():
                sort_tree[j].append(i)
            else:
                sort_tree[j] = [i]         
    return sort_tree

# T_n,k -> R_n,k

# Program 4 Compute \phi
# mapping nodes of T_n,k to R_n,k
def mapping_nodes(k, n, root):
    # input: 
    #     k: treewidth
    #     n: the number of nodes
    #     root: the node set
    # output:
    #     mapping: the mapping function of nodes    
    mapping = {}
    root.sort()
    for i in root:
        mapping[i] = n - k + root.index(i) + 1
    for i in range(1, n - k + 1):
        j = i
        while j in mapping.keys():
            j = mapping[j]
        mapping[j] = i
    return mapping

# R_n,k -> T_n,k
def reverse_permutation(k, n, Q):
    # input: 
    #     k: treewidth
    #     n: the number of nodes
    #     Q: the k_clique adjacent to the maximum labeled leaf l_M of T_n,k and the first part of A_code
    # output:
    #     mapping: the mapping function of nodes
    #     re_mapping: the re_mapping function of nodes  
    
    mapping = mapping_nodes(k, n, Q)
    re_mapping = {}
    for (key, value) in mapping.items():
        re_mapping[value] = key
        
    return mapping, re_mapping
    
# Program 8 Rebuild R_n,k
# T -> R_n,k
def TRnk(k, n, k_parent, k_label):
    # input: 
    #     k: treewidth
    #     n: the number of nodes
    #     k_parent: the dictionary of parent of the nodes 
    #     k_label: the dictionary of label of the nodes
    # output: 
    #     ordering: the perfect ordering of this k-tree
    #     r_tree: the adjacency lists of the Renyi k-tree
    #     k_cliques: a list of all k-cliques in this k-tree
    #     k_add1_cliques: a list of all (k+1)-cliques in this k-tree
    #     tree_decomposition: the tree decomposition of the k-tree

    root = list(range(n - k + 1, n + 1))
    k_sons = {}
    for (key, value) in k_parent.items():
        if value in k_sons.keys():
            k_sons[value].append(key)
        else:
            k_sons[value] = [key]
#     print(k_sons)
    breadth_first_order = []
    idx = v = 0
    while len(breadth_first_order) < n - k:
        if v in k_sons.keys():
            breadth_first_order.extend(k_sons[v])
        v = breadth_first_order[idx]
        idx += 1
#     print(breadth_first_order)
    ordering = root + breadth_first_order
    k_parents = {}
    k_cliques = [root]
    k_add1_cliques = []
    for i in breadth_first_order:
        if k_parent[i] == 0:
            k_parents[i] = root
            for j in range(len(root)):
                clique = root[:j] + root[j+1:] + [i]
#                 clique.sort()
                k_cliques.append(clique)
        else:
            k_parents[i] = [j for j in k_parents[k_parent[i]] if j != k_parents[k_parent[i]][k_label[i]-1]] + [k_parent[i]]
            k_parents[i].sort()
            for j in range(len(k_parents[i])):
                clique = k_parents[i][:j] + k_parents[i][j+1:] + [i]
#                 clique.sort()
                k_cliques.append(clique)
        clique = k_parents[i] + [i]
        k_add1_cliques.append(clique)
    
    nbfo = breadth_first_order + [0]
    tree_decomposition = [(nbfo.index(i) + 1, nbfo.index(k_parent[i]) + 1) for i in breadth_first_order]
    
    r_tree = {}
#     print(k_parents)
    for (key, value) in k_parents.items():
        if key in r_tree.keys():
            r_tree[key].extend(value)
        else:
            r_tree[key] = [i for i in value]

        for i in value:
            if i in r_tree.keys():
                r_tree[i].append(key)
            else:
                r_tree[i] = [key]
        
    for i in root:
        r_tree[i].extend([j for j in root if j != i])
        
    return ordering, r_tree, k_cliques, k_add1_cliques, tree_decomposition

# Program 3 Identify Cycles
# identify all cycles in G
def analyze(node, cycle, k_status, k_parent):
    # input:
    #     node: the processing node
    #     cycle: the set of maximal nodes in cycles
    #     k_status: the dictionary of status of the nodes
    #     k_parent: the dictionary of parent of the nodes
    # output: 
    #     (updated) cycle: the set of maximal nodes in cycles
    #     (updated) k_status: the dictionary of status of the nodes
    #     (updated) k_parent: the dictionary of parent of the nodes

    if k_status[node] != "processed":
        k_status[node] = "inProgress"
        if k_status[k_parent[node]] == "inProgress":
            if node > k_parent[node]:
                cycle.append(node)
            else:
                cycle.append(k_parent[node])
        else:
            analyze(k_parent[node], cycle, k_status, k_parent)
        k_status[node] = "processed"
        
# Program 2 Generalized Dandelion Decoding
# add pair (0, \varepsilon) to the S -> reverse Generalized Dandelion Code T
def S2T(k, n, x, k_parent, k_label):
    # input: 
    #     k: treewidth
    #     n: the number of nodes
    #     x: node
    #     k_parent: the dictionary of parent of the nodes 
    #     k_label: the dictionary of label of the nodes
    # output: 
    #     (updated) k_parent: the dictionary of parent of the nodes 
    #     (updated) k_label: the dictionary of label of the nodes    
    
    k_status = {}
    cycle = []
    for i in range(1, n - k + 1):
        if k_parent[i] == 0:
            k_status[i] = "processed"
        else:
            k_status[i] = "unprocessed"
    for i in range(1, n - k + 1):
        analyze(i, cycle, k_status, k_parent)
    cycle.sort()
    for i in cycle:
        pa = k_parent[i]
        la = k_label[i]
        k_parent[i] = k_parent[x]
        k_label[i] = k_label[x]
        k_parent[x] = pa
        k_label[x] = la
    
#     print(k_parent)
#     print(k_label)

    return k_parent, k_label

def decoding_k_tree(k, n, A_code, save = False, path = None):
    # input: 
    #     k: treewidth
    #     n: the number of nodes
    #     A_code: the code of the k-tree in A_n,k
    #     save: save the tree decomposition of k-tree into a .gr file
    #     paht: the path to save the .gr file
    # output:
    #     k_tree: the adjacency lists of the k-tree
    #     k_cliques_decode: a list of all decodec k-cliques in this k-tree
    #     k_add1_cliques_decode: a list of all decoded (k+1)-cliques in this k-tree
    #     ordering_decode: the decoded ordering of the k-tree
    
    Q = A_code[0]
    mapping, re_mapping = reverse_permutation(k, n, Q)
    
    # add pair (0, \varepsilon) to the S
    code = A_code[1]
    r = 0
    x = mapping[min([i for i in range(1, n + 1) if i not in Q])]
    internal_nodes = [i[0] for i in code]
    internal_label = [i[1] for i in code]
    
    l_M = 0
    for i in range(1, n - k + 1):
        if i not in internal_nodes and re_mapping[i] > l_M:
            l_M = re_mapping[i]

    count = 0
    k_parent = {}
    k_label = {}
    for i in range(1, n - k + 1):
        if i == x or i == mapping[l_M]:
            k_parent[i] = 0
            k_label[i] = 'w'
        else:
            k_parent[i] = internal_nodes[count]
            k_label[i] = internal_label[count]
            count += 1

#     print(k_parent)
#     print(k_label)
#     sys.exit(0)

    # add pair (0, \varepsilon) to the S -> reverse Generalized Dandelion Code T
    k_parent, k_label = S2T(k, n, x, k_parent, k_label)
    
    # T -> R_n,k
    ordering, r_tree, k_cliques, k_add1_cliques, tree_decomposition = TRnk(k, n, k_parent, k_label)
    
#     print(tree_decomposition)
    # R_n,k -> T_n,k
    k_tree = {}
    for (key, value) in r_tree.items():
        k_tree[re_mapping[key]] = [re_mapping[i] for i in value]
    k_tree = order_adj(k_tree)
    
    k_cliques_decode = []
    for kcl in k_cliques:
        k_cliques_decode.append([re_mapping[i] for i in kcl])
    
    k_add1_cliques_decode = []
    for kcl in k_add1_cliques:
        k_add1_cliques_decode.append([re_mapping[i] for i in kcl])
    
    if save:
        with open(path + ".gr", 'w') as f:
            line = "p" + " tw "
            num_edge = k * (k - 1) / 2 + k * (n - k)
            line += str(int(n)) + " " + str(int(num_edge)) + "\n"
            f.write(line)

            for (key, value) in k_tree.items():
                for node in value:
                    f.write(str(key) + " " + str(node) + "\n")
            
        with open(path + ".td", 'w') as f:
            line = "s" + " td"
            line += " " + str(len(k_add1_cliques) + 1) + " " + str(int(k + 1)) + " " + str(int(n)) + "\n"
            f.write(line)

            for i, bag in enumerate(k_add1_cliques_decode):
                line = "b " + str(i + 1) 
                for j in bag:
                    line += " " + str(j)
                f.write(line + "\n")

            line = "b " + str(i + 2)
            for j in k_cliques_decode[0]:
                line += " " + str(j)
            f.write(line + "\n")
                
            for edge in tree_decomposition:
                f.write(str(edge[0]) + " " + str(edge[1]) + "\n")
    
    ordering_decode = [re_mapping[i] for i in reversed(ordering)]
    
    return k_tree, k_cliques_decode, k_add1_cliques_decode, ordering_decode

def create_A_code(k, n):
    ls_root = list(range(1, n + 1))
    root = np.random.choice(ls_root, k, replace = False)
    ls_code = [(0, "w")] + [(i, j) for i in range(1, n - k + 1) for j in range(1, k + 1)]
    code_idx = np.random.choice(range(len(ls_code)), n - k - 2, replace = False)
    code = [ls_code[i] for i in code_idx]
    return (root.tolist(), code)

def get_aktree_by_code(k, n, save = False, path = None):
    if k > n:
        raise ValueError(f"the tree-width is greater than the number of vertices in the graph.")
    A_code = create_A_code(k, n)
    k_tree, k_cliques, k_add1_cliques, ordering = decoding_k_tree(k, n, A_code, save, path)
    return k_tree, k_cliques, k_add1_cliques, ordering

# disorder the vertices
def disorder(nodes):
    """
    input: nodes: a list of nodes
    outputs: nodes_en: a encoding dictionary with the nodes as keys and the corresponding codes as values
    """
    nodes_dis = np.random.permutation(np.arange(len(nodes)))
    nodes_en = {nodes[i]: nodes_dis[i] + 1 for i in range(len(nodes))}
    return nodes_en

import copy

def create_aktree(k, n):
    k_tree = {}
    k_cliques = []
    k_add1_cliques = []
    if k > n:
        raise ValueError(f"the tree-width is greater than the number of vertices in the graph.")
    else:
        root = list(np.linspace(1, k, k, dtype = int))
        nodes = 1
        for i in range(k):
            k_tree[nodes] = root[0:root.index(nodes)] + root[root.index(nodes)+1:]
            nodes += 1 
        k_cliques.append(root)
        select_clique = 0
        while nodes <= n:
            k_tree[nodes] = copy.deepcopy(k_cliques[select_clique])
            k_add1_cliques.append(k_tree[nodes] + [nodes])
            for i in k_cliques[select_clique]:
                k_tree[i].append(nodes)
                cl = k_cliques[select_clique][0:k_cliques[select_clique].index(i)] \
                    + k_cliques[select_clique][k_cliques[select_clique].index(i)+1:] \
                    + [nodes]
                k_cliques.append(cl)
#             if np.random.rand() < 0.2:
#                 select_clique = 0
#             else:
#                 select_clique = np.random.randint(0, len(k_cliques) - 1)
            select_clique = np.random.randint(0, len(k_cliques) - 1)
            nodes += 1

    ls_nodes = list(range(1, n + 1))
    nodes_en = disorder(ls_nodes)
    k_tree_encode = {}
    for (key, value) in k_tree.items():
        k_tree_encode[nodes_en[key]] = [nodes_en[i] for i in value]

    k_cliques_encode = []
    for kcl in k_cliques:
        k_cliques_encode.append([nodes_en[i] for i in kcl])
    
    k_add1_cliques_encode = []
    for kcl in k_add1_cliques:
        k_add1_cliques_encode.append([nodes_en[i] for i in kcl])

    ordering = [nodes_en[i] for i in reversed(ls_nodes)]
        
    return k_tree_encode, k_cliques_encode, k_add1_cliques_encode, ordering

# check the connectivity of the created Markov random fields
# Disjoint-set data structure
def find(dic, nod):
    """
    inputs: dic: a dictionary with nodes as keys and the smallest node in the set their belong to as values
            nod: the node that need to find its father, its father is the smallest node in the set it belongs to
    output: the father
    """
    if dic[nod] == nod:
        father = nod
    else:
        father = find(dic, dic[nod])
    return father
        
def check_subgraphs(nodes, edges):
    """
    inputs: nodes: the list of nodes in the graph
            edges: the list of edges in the graph
    output: the largest father among all subsets
    """
    dic = {}
    for nod in nodes:
        dic[nod] = nod
    for edg in edges:
        fa = [find(dic, edg[0]), find(dic, edg[1])]
        if fa[0] == fa[1]:
            continue
        else:
            idx = fa.index(max(fa))
            dic[fa[idx]] = min(fa)
            dic[edg[idx]] = min(fa)
    for nod in nodes:
        dic[nod] = find(dic, dic[nod])
    
    return len(set(dic.values()))

from itertools import combinations
import networkx as nx
import math

def removing_edges(nodes_o, edges_o, prob_re, k_tree, k_cliques, kk, random = False, clique = False):
    len_edges = len(edges_o)
    if clique:
        selected_clique = np.random.randint(0, len(k_cliques) - 1)
        fixed_clique = k_cliques[selected_clique]
        fixed_edges = set(combinations(sorted(fixed_clique), 2))
#         fixed_edges = {(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in fixed_edges}
        edges_o = edges_o - fixed_edges
        
    connectivity = False
    count = 0
    while not connectivity:
        count += 1
        num_re = int(len_edges * prob_re)
        
        if random:
            idx_re = np.random.permutation(np.arange(len(edges_o)))[:num_re]
            edges = set([list(edges_o)[i] for i in range(len(edges_o)) if i not in idx_re])
        else:
            len_ed_re = []
            for m in range(6):
                h = 2 - m * 0.2
                edge_re_ca = [edg for edg in edges_o if len(k_tree[edg[0]]) > h * kk and len(k_tree[edg[1]]) > h * kk]
                len_ed_re.append(len(edge_re_ca))
                if len(edge_re_ca) > num_re:
                    break
            if len_ed_re[-1] < num_re:
                for m in range(5):
                    h = 1.4 - m * 0.1
                    edge_re_ca = [edg for edg in edges_o if len(k_tree[edg[0]]) > h * kk and len(k_tree[edg[1]]) >= h * kk]
                    if len(edge_re_ca) > num_re:
                        break

            prob_edg = [(math.pow(len(k_tree[edg[0]]), 2) + math.pow(len(k_tree[edg[1]]), 2)) for edg in edge_re_ca]
            prob_edg = [pro/sum(prob_edg) for pro in prob_edg]
            idx_re = np.random.choice(len(edge_re_ca), num_re, replace = False, p = prob_edg)
            edges = edges_o - set([edge_re_ca[i] for i in idx_re])
            
        if count > 100:
            num_add = len_edges - num_re - len(nodes_o) + 1
            edges_o.update(fixed_edges)
            G = nx.Graph(edges_o)
            T = nx.minimum_spanning_tree(G)
            edges_st = {(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in T.edges(data = False)}
            num_add_extra = len(fixed_edges - edges_st)
            remain = edges_o - edges_st - fixed_edges
            idx_re = np.random.permutation(np.arange(len(remain)))[:(num_add - num_add_extra)]
            edges = set([list(remain)[i] for i in range(len(remain)) if i in idx_re])
            edges.update(edges_st)
        
        if clique:
            edges.update(fixed_edges)
            
        dic = {}
        for nod in nodes_o:
            dic[nod] = []
        for edg in edges:
            dic[edg[0]].append(edg[1])
            dic[edg[1]].append(edg[0])

        nodes_re = []
        for (k, v) in dic.items():
            if len(v) == 0:
                nodes_re.append(k)

        nodes = [nod for nod in nodes_o if nod not in nodes_re]
        
        if check_subgraphs(nodes, edges) == 1:
            connectivity = True
            
    return nodes, edges

def check_cliques(k_cliques, edges, nodes):
    cliques = set()
    cli_edg = set()
    cli_edges = set()
    for clique in k_cliques:
        clique.sort()
        remove_edges = set(combinations(clique, 2)) - edges
        if len(remove_edges) == 0:
            cli_edges.update(set(combinations(clique, 2)))
            cliques.add(tuple(clique))
            continue
        else:
            temp_dic = {}
            for edge in remove_edges:
                for i in range(2):
                    if edge[i] in temp_dic.keys():
                        temp_dic[edge[i]].append(edge[1 - i])
                    else:
                        temp_dic[edge[i]] = [edge[1 - i]]
            for key in temp_dic.keys():
                clique_n = sorted(set(clique) - set(temp_dic[key])) # list
#                 sys.exit()
                if len(clique_n) == 1:
                    if clique_n[0] in nodes:
                        cliques.add(tuple(clique_n))
                    else:
                        continue
                elif len(clique_n) == 2:
                    cli_edg.add(tuple(clique_n))
                elif len(set(combinations(clique_n, 2)) - edges) == 0:
                    cli_edges.update(set(combinations(clique_n, 2)))
                    cliques.add(tuple(clique_n))
                else:
                    cli_edg.update(set(combinations(clique_n, 2)).intersection(edges))

    for clique in cli_edg:
        if clique not in cli_edges:
            cliques.add(clique)
    
    return cliques

def remove_node(MRF_dic, edges, node):
    neighbors = MRF_dic[node]['neighbors']
    add_edges = set(combinations(sorted(neighbors), 2)) - set(edges)
    for edge in MRF_dic[node]['neigh_edg']:
        if edge in edges:
            edges.remove(edge)
        else:
            continue
    edges.extend(list(add_edges))
    for neighbor in neighbors:
        MRF_dic[neighbor]['neighbors'].remove(node)
        add_neighbors = list(set(neighbors) - set([neighbor]) - set(MRF_dic[neighbor]['neighbors']))
#         MRF_dic[neighbor]['neighbors'] = list(set(MRF_dic[neighbor]['neighbors']).union(set(neighbors) - set(neighbor)))
#         MRF_dic[neighbor]['neighbors'].extend([x for x in neighbors if x not in MRF_dic[neighbor]['neighbors'] and x != neighbor])
        MRF_dic[neighbor]['neighbors'].extend(add_neighbors)
        MRF_dic[neighbor]['neigh_edg'].remove((min(node, neighbor), max(node, neighbor)))
        for an in add_neighbors:
            MRF_dic[neighbor]['neigh_edg'].append((min(an, neighbor), max(an, neighbor)))
    del MRF_dic[node]

def get_MinNeighbors(MRF_dic, node):
    return len(MRF_dic[node]['neighbors'])

def get_MinFill(MRF_dic, node):
    neighbors = MRF_dic[node]['neighbors']
    exist_edges = set()
    for neighbor in neighbors:
        exist_edges.update(set(MRF_dic[neighbor]['neigh_edg']))
    add_edges = set(combinations(sorted(neighbors), 2)) - set(exist_edges)
    return len(add_edges)

def get_elimination_order(nodes_o, edges_o, cost):
    # cost: get_MinNeighbors, get_MinWeight, get_MinFill, get_WeightedMinFill
    nodes = list(copy.deepcopy(nodes_o))
    edges = list(copy.deepcopy(edges_o))
    
    ordering = []
    dic = {}
    for nod in nodes:
        dic[nod] = {'neighbors': [], 'neigh_edg': []}
    for edg in edges_o:
        dic[edg[0]]['neigh_edg'].append(edg)
        dic[edg[1]]['neigh_edg'].append(edg)
        dic[edg[0]]['neighbors'].append(edg[1])
        dic[edg[1]]['neighbors'].append(edg[0])
    
#     max_factor = 0
    while len(nodes) > 1:
        scores = {node: cost(dic, node) for node in nodes}
        min_score_node = min(scores, key = scores.get)
#         if max_factor < len(dic[min_score_node]['neighbors']):
#             max_factor = len(dic[min_score_node]['neighbors'])       
        ordering.append(min_score_node)
#         print(min_score_node)
        nodes.remove(min_score_node)
        remove_node(dic, edges, min_score_node)
    
    ordering.extend(nodes)
#     print(max_factor)
    return ordering

def max_fac(nodes, edges, eli_ord):
    dic = {}
    edges_n = list(copy.deepcopy(edges))

    for node in nodes:
        dic[node] = {'neighbors': [], 'neigh_edg': []}
    for edge in edges_n:
        dic[edge[0]]['neighbors'].append(edge[1])
        dic[edge[1]]['neighbors'].append(edge[0])
        dic[edge[0]]['neigh_edg'].append(edge)
        dic[edge[1]]['neigh_edg'].append(edge)

#     for k, v in dic.items():
#         print(k)
#         for key, value in v.items():
#             print(key, value)

    max_factor = 0
    for i in eli_ord:
        if max_factor < len(dic[i]['neighbors']):
            max_factor = len(dic[i]['neighbors'])
#         print(i, len(dic[i]['neighbors']))
        remove_node(dic, edges_n, i)
#     print(max_factor)
    return max_factor