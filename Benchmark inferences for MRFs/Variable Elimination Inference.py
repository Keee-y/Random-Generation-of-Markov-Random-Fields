import signal
import time
def set_timeout(num):
    def wrap(func):
        def handle(signum, frame):  # 收到信号 SIGALRM 后的回调函数，第一个参数是信号的数字，第二个参数是the interrupted stack frame.
            raise RuntimeError

        def to_do(graph, var, evid, eli_order, joint, show_progress):
            try:
                signal.signal(signal.SIGALRM, handle)  # 设置信号和回调函数
                signal.alarm(num)  # 设置 num 秒的闹钟
#                 print('start alarm signal.')
                r = func(graph, var, evid, eli_order, joint, show_progress)
#                 print('close alarm signal.')
                signal.alarm(0)  # 关闭闹钟
                return r
            except RuntimeError as e:
                return 30, 30  # timeout
        return to_do
    return wrap

# https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
import numpy as np

def random(idx, n):
    # idx: select random methods
    # n: size
    if idx == 1:
        # Draw samples from the Dirichlet distribution.
        return np.random.dirichlet(np.ones(n), 1)
    elif idx == 2:
        # Draw samples from an exponential distribution.
        return np.random.exponential(1, n)
    elif idx == 3:
        # Draw samples from a Beta distribution.
        return np.random.beta(2, 3, n)
    elif idx == 4:
        # Draw samples from a chi-square distribution.
        return np.random.chisquare(2, n)
    elif idx == 5:
        # Draw samples from a uniform distribution.
        return np.random.uniform(10, 100, n)
    elif idx == 6:
        # Draw samples from a uniform distribution over [0, 1).
        return np.random.rand(n)
    elif idx == 7:
        # Draw random samples from a log-normal distribution.
        return np.random.lognormal(0, 1, n)
    else:
        # Draw random samples from a normal (Gaussian) distribution. - problematic: values should be larger than 0.
#         return np.random.normal(0, 1, n) 
        return np.ones(n)



import copy
def shuffle_nodes(nodes):
    nodes_shuffle = copy.deepcopy(nodes)
    np.random.shuffle(nodes_shuffle)
    node2shuffle = {node: shuffle for (node, shuffle) in zip(nodes, nodes_shuffle)}
    shuffle2node = {shuffle: node for (node, shuffle) in zip(nodes, nodes_shuffle)}
    return node2shuffle, shuffle2node

# # create a k-tree incursively
# def create_aktree(k, n):
#     # input:
#     #     k: treewidth
#     #     n: the number of nodes
#     # output:
#     #     k_cliques: all cliques in the k-tree
#     #     k_tree: the adjacency lists of the k-tree
#     k_tree = {}
#     k_cliques = []
#     if k > n:
#         return k_cliques, k_tree
#     else:
#         root = list(np.linspace(n, n-k+1, k, dtype = int))
#         nodes = n
#         for i in range(k):
#             k_tree[nodes] = root[0:root.index(nodes)] + root[root.index(nodes)+1:]
#             nodes -= 1 
#         k_cliques.append(root)
#         select_clique = 0
#         while nodes > 0:
#             k_tree[nodes] = copy.copy(k_cliques[select_clique])
#             for i in k_cliques[select_clique]:
#                 k_tree[i].append(nodes)
#                 cl = k_cliques[select_clique][0:k_cliques[select_clique].index(i)] \
#                     + k_cliques[select_clique][k_cliques[select_clique].index(i)+1:] \
#                     + [nodes]
#                 k_cliques.append(cl)
#             select_clique = np.random.randint(0, len(k_cliques)-1)
#             nodes -= 1
#         return k_cliques, k_tree

# create a k-tree recursively
def create_aktree(k, n):
    # input:
    #     k: treewidth
    #     n: the number of nodes
    # output:
    #     k_cliques: all cliques in the k-tree
    #     k_tree: the adjacency lists of the k-tree
    k_tree = {}
    k_cliques = []
    if k > n:
        return k_cliques, k_tree
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
            for i in k_cliques[select_clique]:
                k_tree[i].append(nodes)
                cl = k_cliques[select_clique][0:k_cliques[select_clique].index(i)] \
                    + k_cliques[select_clique][k_cliques[select_clique].index(i)+1:] \
                    + [nodes]
                k_cliques.append(cl)
            if np.random.rand() < 0.2:
                select_clique = 0
            else:
                select_clique = np.random.randint(0, len(k_cliques)-1)
            nodes += 1
        return k_cliques, k_tree

def cal_edges(n, k):
    return int(k * (k - 1) / 2 + k * (n - k))

def cal_cli(n, k):
    return int(1 + (n - k) * k)

def cal_param(n, k):
    return cal_cli(n, k) * int(math.pow(2, k))

def sample(low, high, size):
    resultList = set()
    while len(resultList) < size:
        tempInt = np.random.randint(low, high)
        resultList.add(tempInt)
    return list(resultList)

from functools import reduce

def preparation(graph, variables, evidence, elimination_order, joint = True):

    # get working factors
    working_factors = {
        node: {(DiscreteFactor(factor.scope(), factor.cardinality, np.log(factor.values)), None) for factor in graph.get_factors(node)} # set
        for node in graph.nodes}

    to_eliminate = (
        set(graph.nodes)
        - set(variables)
        - set(evidence.keys() if evidence else [])
    )

    # get elimination order
    # Step 1: If elimination_order is a list, verify it's correct and return.
    # Step 1.1: Check that not of the `variables` and `evidence` is in the elimination_order.
    if hasattr(elimination_order, "__iter__") and (not isinstance(elimination_order, str)):
        if any(var in elimination_order for var in set(variables).union(set(evidence.keys() if evidence else []))):
            raise ValueError(
                "Elimination order contains variables which are in"
                " variables or evidence args"
            )
        # Step 1.2: Check if elimination_order has variables which are not in the model.
        elif any(var not in graph.nodes() for var in elimination_order):
            elimination_order = list(filter(lambda t: t in graph.nodes(), elimination_order))

        # Step 1.3: Check if the elimination_order has all the variables that need to be eliminated.
        elif to_eliminate != set(elimination_order):
            raise ValueError(
                f"Elimination order doesn't contain all the variables"
                f"which need to be eliminated. The variables which need to"
                f"be eliminated are {to_eliminate}")

    # Step 2: If elimination order is None or a Markov model, return a random order.
    elif elimination_order is None:
        elimination_order = to_eliminate
    else:
        elimination_order = None

    # marginal
    if not variables:
        variables = []

    common_vars = set(evidence if evidence is not None else []).intersection(
        set(variables if variables is not None else [])
    )
    if common_vars:
        raise ValueError(f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}")

    # variable elimination
    # Step 1: Deal with the input arguments.
    if isinstance(variables, str):
        raise TypeError("variables must be a list of strings")
    if isinstance(evidence, str):
        raise TypeError("evidence must be a list of strings")

    # Dealing with the case when variables is not provided.
    if not variables:
        all_factors = []
        for factor_li in graphs.get_factors():
            all_factors.extend(factor_li)
        if joint:
            return factor_product(*set(all_factors))
        else:
            return set(all_factors)
            
    return working_factors, elimination_order


def factor_product(*args):
        
    return reduce(lambda phi1, phi2: phi1 + phi2, args)

# Compute the most probable explanation (MPE)
def maximize(phi_o, variables, inplace=True):

    if isinstance(variables, str):
        raise TypeError("variables: Expected type list or array-like, got type str")

    phi = phi_o if inplace else phi_o.copy()

    for var in variables:
        if var not in phi.variables:
            raise ValueError(f"{var} not in scope.")

    # get the indices of the input variables
    var_indexes = [phi.variables.index(var) for var in variables]

    # get the indices of the rest variabels
    index_to_keep = sorted(set(range(len(phi_o.variables))) - set(var_indexes))
    # the new factor with the rest variables
    phi.variables = [phi.variables[index] for index in index_to_keep]
    # the new factor with the cardinality of the rest variables
    phi.cardinality = phi.cardinality[index_to_keep]
    # delete the eliminated variables
    phi.del_state_names(variables)

    var_assig = np.argmax(phi.values, axis = var_indexes[0])
    phi.values = np.max(phi.values, axis = tuple(var_indexes))

    if not inplace:
        return phi, var_assig

# @set_timeout(30)
def cal_mpe(graph, variables, evidence, elimination_order, joint = True, show_progress = True):

    # Step 2: Prepare data structures to run the algorithm.
    eliminated_variables = set()
    # Get working factors and elimination order
    working_factors, elimination_order = preparation(graph, variables, evidence, elimination_order, joint = True)
    # elimination_order = elimination_order

    assignments = {node: None for node in graph.nodes}
    eliminated_assignments = {node: (None, None) for node in elimination_order}

    # Step 3: Run variable elimination
    if show_progress:
        pbar = tqdm(elimination_order)
    else:
        pbar = elimination_order

    for var in pbar:
        if show_progress:
            pbar.set_description(f"Eliminating: {var}")
        # Removing all the factors containing the variables which are
        # eliminated (as all the factors should be considered only once)
        factors = [factor for factor, _ in working_factors[var] if not set(factor.scope()).intersection(eliminated_variables)]
        phi = factor_product(*factors)
        phi, var_assignment = maximize(phi, [var], inplace = False)
        # phi = getattr(phi, operation)([var], inplace=False)
        del working_factors[var]
        for variable in phi.variables:
            working_factors[variable].add((phi, var))
        eliminated_variables.add(var)
        eliminated_assignments[var] = (var_assignment, phi.variables)

    # Step 4: Prepare variables to be returned.
    final_distribution = set()
    for node in working_factors:
        for factor, origin in working_factors[node]:
            if not set(factor.variables).intersection(eliminated_variables):
                final_distribution.add((factor, origin))
    final_distribution = [factor for factor, _ in final_distribution]

    if joint:
        if isinstance(graph, BayesianNetwork):
            final_distribution = factor_product(*final_distribution).normalize(inplace=False)
        else:
            final_distribution = factor_product(*final_distribution)
    else:
        query_var_factor = {}
        for query_var in variables:
            phi = factor_product(*final_distribution)
            query_var_factor[query_var] = phi.marginalize(list(set(variables) - set([query_var])), inplace=False).normalize(inplace=False)
        final_distribution = query_var_factor
    
    max_assign = np.unravel_index(np.argmax(final_distribution.values, axis = None), final_distribution.values.shape)  
    for (node, assign) in zip(final_distribution.variables, max_assign):
        assignments[node] = assign
    elimination_order.reverse()
    for node in elimination_order:
        ind = []
        for variable in eliminated_assignments[node][1]:
            ind.append(assignments[variable])
        assignments[node] = eliminated_assignments[node][0][tuple(ind)]
    # max_assign = np.argmax(final_distribution.values)
    max_prob = np.max(final_distribution.values)
    return max_prob, assignments
#     return assignments

# Compute the partition function (PR)
def marginalize(phi_o, variables, inplace=True):
    
    if isinstance(variables, str):
        raise TypeError("variables: Expected type list or array-like, got type str")

    phi = phi_o if inplace else phi_o.copy()

    for var in variables:
        if var not in phi.variables:
            raise ValueError(f"{var} not in scope.")

    # get the indices of the input variables
    var_indexes = [phi.variables.index(var) for var in variables]

    # get the indices of the rest variabels
    index_to_keep = sorted(set(range(len(phi_o.variables))) - set(var_indexes))
    # the new factor with the rest variables
    phi.variables = [phi.variables[index] for index in index_to_keep]
    # the new factor with the cardinality of the rest variables
    phi.cardinality = phi.cardinality[index_to_keep]
    # delete the eliminated variables
    phi.del_state_names(variables)

#     phi.values = np.log(np.sum(np.exp(phi.values), axis = tuple(var_indexes)))
    values = np.split(phi.values, 2, axis = var_indexes[0])
    phi.values = np.logaddexp(values[0].reshape([2 for i in range(len(phi.variables))]), values[1].reshape([2 for i in range(len(phi.variables))]))

    if not inplace:
        return phi

# @set_timeout(30)
def cal_pr(graph, variables, evidence, elimination_order, joint = True, show_progress = False):
    
    # Step 2: Prepare data structures to run the algorithm.
    eliminated_variables = set()
    # Get working factors and elimination order
    working_factors, elimination_order = preparation(graph, variables, evidence, elimination_order, joint = True)
    # elimination_order = elimination_order

    # Step 3: Run variable elimination
    if show_progress:
        pbar = tqdm(elimination_order)
    else:
        pbar = elimination_order

    for var in pbar:
        if show_progress:
            pbar.set_description(f"Eliminating: {var}")
        # Removing all the factors containing the variables which are
        # eliminated (as all the factors should be considered only once)
        factors = [factor for factor, _ in working_factors[var] if not set(factor.scope()).intersection(eliminated_variables)]
#         if len(factors) == 0:
#             print(var)
#             print(factors)
        phi = factor_product(*factors)
        phi = marginalize(phi, [var], inplace = False)
        # phi = getattr(phi, operation)([var], inplace=False)
        del working_factors[var]
        for variable in phi.variables:
            working_factors[variable].add((phi, var))
        eliminated_variables.add(var)
#         eliminated_assignments[var] = (var_assignment, phi.variables)

    # Step 4: Prepare variables to be returned.
    final_distribution = set()
    for node in working_factors:
        for factor, origin in working_factors[node]:
            if not set(factor.variables).intersection(eliminated_variables):
                final_distribution.add((factor, origin))
    final_distribution = [factor for factor, _ in final_distribution]

    if joint:
        if isinstance(graph, BayesianNetwork):
            final_distribution = factor_product(*final_distribution).normalize(inplace=False)
        else:
            final_distribution = factor_product(*final_distribution)
    else:
        query_var_factor = {}
        for query_var in variables:
            phi = factor_product(*final_distribution)
            query_var_factor[query_var] = phi.marginalize(list(set(variables) - set([query_var])), inplace=False).normalize(inplace=False)
        final_distribution = query_var_factor

    return final_distribution

@set_timeout(60)
def compute(graph, var, evid, eli_order, joint = True, show_progress = False):
    q = cal_pr(graph, variables = var, evidence = evid, elimination_order = eli_order, joint = True, show_progress = False)
    max_prob, assignments = cal_mpe(graph, variables = var, evidence = evid, elimination_order = eli_order, joint = True, show_progress = False)
    return q, assignments

from itertools import combinations

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

def get_MinWeight(MRF_dic, node):
    return np.sum([np.log(MRF_dic[neighbor]['card']) for neighbor in MRF_dic[node]['neighbors']])

def get_MinFill(MRF_dic, node):
    neighbors = MRF_dic[node]['neighbors']
    exist_edges = set()
    for neighbor in neighbors:
        exist_edges.update(set(MRF_dic[neighbor]['neigh_edg']))
    add_edges = set(combinations(sorted(neighbors), 2)) - set(exist_edges)
    return len(add_edges)

def get_WeightedMinFill(MRF_dic, node):
    neighbors = MRF_dic[node]['neighbors']
    exist_edges = set()
    for neighbor in neighbors:
        exist_edges.update(set(MRF_dic[neighbor]['neigh_edg']))
    add_edges = set(combinations(sorted(neighbors), 2)) - set(exist_edges)
    return np.sum([MRF_dic[edge[0]]['card'] * MRF_dic[edge[1]]['card'] for edge in add_edges])

def get_elimination_order(MRF, cost):
    # cost: get_MinNeighbors, get_MinWeight, get_MinFill, get_WeightedMinFill
    nodes = list(MRF.nodes)
    edges_o = list(MRF.edges)
    edges = list()
    
    ordering = []
    dic = {}
    for nod in nodes:
        dic[nod] = {'card': MRF.get_cardinality(nod),'neighbors': [], 'neigh_edg': []}
    for edg in edges_o:
        edge = (min(edg[0], edg[1]), max(edg[0], edg[1]))
        edges.append(edge)
        dic[edg[0]]['neigh_edg'].append(edge)
        dic[edg[1]]['neigh_edg'].append(edge)
        dic[edg[0]]['neighbors'].append(edg[1])
        dic[edg[1]]['neighbors'].append(edg[0])
    
    while len(nodes) > 1:
        scores = {node: cost(dic, node) for node in nodes}
        min_score_node = min(scores, key = scores.get)
        ordering.append(min_score_node)
#         print(min_score_node)
        nodes.remove(min_score_node)
        remove_node(dic, edges, min_score_node)
    
    ordering.extend(nodes)
    return ordering

# convert MRF to BN
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.models import MarkovNetwork, BayesianNetwork

def convert(G):
    nodes = list(G.nodes)
    edges = list(G.edges)
    BN = BayesianNetwork()
    s_nodes = [str(node) for node in nodes]
    z_nodes = [str(len(nodes) + 1 + i) for i in range(len(edges))]
    BN.add_nodes_from(s_nodes + z_nodes)
    for i in s_nodes:
        card = G.get_cardinality(node = int(i))
        cpd = TabularCPD(i, card, [[1/card] for j in range(card)])
        BN.add_cpds(cpd)  

    for (edge, z) in zip(edges, z_nodes):
        BN.add_edges_from([(str(l), z) for l in edge])
        factor = [fa for fa in G.get_factors(edge[0]) if edge[1] in fa.scope()][0]
        phi = factor.copy()
        phi.normalize()
        phi_reshape = phi.values.reshape((1, -1))
        cpd = TabularCPD(z, 2, np.vstack((phi_reshape, 1 - phi_reshape)), evidence = [str(edge[0]), str(edge[1])], \
                         evidence_card = list(factor.get_cardinality(edge).values()))
        BN.add_cpds(cpd)  
    
    return BN

def get_graph(path):
#     reader = UAIReader(path = path)
    with open(path, 'r') as f:

        # Preamble
        lines = [x.strip() for x in f.readlines() if x.strip()]
#         assert lines[0] == 'MARKOV'
        if lines[0] == 'MARKOV':
            G = MarkovNetwork()
        else:
            raise ValueError("This is not a Markov random field.")
            
        n_vars = int(lines[1])
        nodes = list(range(1, n_vars + 1))
        cardinalities = [int(x) for x in lines[2].split()]
        n_cliques = int(lines[3])
        edges = set()
        factor = []
        G.add_nodes_from(nodes)
        for i in range(n_cliques):
            edge = [y + 1 for y in [int(x) for x in lines[i + 4].split()][1:]]
#             print(edge)
            edges_in_cliq = set(combinations(sorted(edge), 2))
            edges.update(edges_in_cliq)
            factor.append(DiscreteFactor(edge, [cardinalities[node - 1] for node in edge], [float(x) for x in lines[i * 2 + 1 + n_cliques + 4].split()]))
        
        G.add_edges_from(edges)
        G.add_factors(*factor)
        
    return G

def max_fac(graph, eli_ord):
    dic = {}
    edges_n = copy.deepcopy(list(graph.edges))

    for node in graph.nodes:
        dic[node] = {'neighbors': [], 'neigh_edg': []}
    for edge in edges_n:
        dic[edge[0]]['neighbors'].append(edge[1])
        dic[edge[1]]['neighbors'].append(edge[0])
        dic[edge[0]]['neigh_edg'].append(edge)
        dic[edge[1]]['neigh_edg'].append(edge)

    max_factor = 0
    for i in eli_ord:
        if max_factor < len(dic[i]['neighbors']):
            max_factor = len(dic[i]['neighbors'])
#         print(i, len(dic[i]['neighbors']))
        remove_node(dic, edges_n, i)
#     print(max_factor)
    return max_factor

import math
def create_aMRF(kk, n, idx, remove_edges = False, prob_re = 0.0):
    
#     kk = 4
#     n = 10
#     idx = 2
#     remove_edges = True
#     prob_re = 0.1

    k_cliques, k_tree = create_aktree(kk, n)

    nodes_o = [i+1 for i in range(n)]
    edges_o = set()
    for (k, v) in k_tree.items():
        for l in v:
            edges_o.add((min(l, k), max(l, k)))
            
    # print(nodes)
    # print(edges)

    if remove_edges and prob_re > 2e-5:
        num_re = int(len(edges_o) * prob_re)
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
        
    else:
        nodes = nodes_o
        edges = edges_o
    
    G2 = MarkovNetwork()
    G2.add_nodes_from(nodes)
    G2.add_edges_from(edges)
    # factor_edges = [DiscreteFactor(edge, [2 for i in range(kk)], random(idx, int(math.pow(2, kk)))) for edge in k_cliques]
    factor_edges = [DiscreteFactor(edge, [2, 2], random(idx, 4)) for edge in edges]
    G2.add_factors(*factor_edges)
    
    return G2


def create_MRFs(kk, n, idx, remove_edges = False, prob_re = 0.0):
    
#     kk = 4
#     n = 10
#     idx = 2
#     remove_edges = True
#     prob_re = 0.1

    k_cliques, k_tree = create_aktree(kk, n)

    nodes_o = [i+1 for i in range(n)]
    edges_o = set()
    cliques_o = k_cliques
    for (k, v) in k_tree.items():
        for l in v:
            edges_o.add((min(l, k), max(l, k)))
            
    # print(nodes)
    # print(edges)

    if remove_edges and prob_re > 2e-5:
        num_re = int(len(edges_o) * prob_re)
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
    else:
        nodes = nodes_o
        edges = edges_o
        cliques = cliques_o

    G1 = MarkovNetwork()
    G1.add_nodes_from(nodes)
    G1.add_edges_from(edges)
    # factor_edges = [DiscreteFactor(edge, [2 for i in range(kk)], random(idx, int(math.pow(2, kk)))) for edge in k_cliques]
    factor_edges = [DiscreteFactor(clique, [2 for i in range(len(clique))], random(idx, int(math.pow(2, len(clique))))) for clique in cliques]
    G1.add_factors(*factor_edges)
    
    G2 = MarkovNetwork()
    G2.add_nodes_from(nodes)
    G2.add_edges_from(edges)
    # factor_edges = [DiscreteFactor(edge, [2 for i in range(kk)], random(idx, int(math.pow(2, kk)))) for edge in k_cliques]
    factor_edges = [DiscreteFactor(edge, [2, 2], random(idx, 4)) for edge in edges]
    G2.add_factors(*factor_edges)
    
    return G1, G2

def find(dic, nod):
    if dic[nod] == nod:
        father = nod
    else:
        father = find(dic, dic[nod])
    return father
        

def check_subgraphs(nodes, edges):
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
    
    return max(dic.values())

