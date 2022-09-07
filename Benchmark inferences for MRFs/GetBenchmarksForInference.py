from pgmpy.readwrite import UAIWriter
from DandelionCode import * 
from VariableEliminationInference import *

def benchmark(idx, kk, n, path):
    
    k_tree, k_cliques, k_add1_cliques, ordering = get_aktree_by_code(kk, n, save = False, path = None)

    nodes_o = [l+1 for l in range(n)]
    edges_o = set()
    for (k, v) in k_tree.items():
        for l in v:
            edges_o.add((min(l, k), max(l, k)))

    for l in range(5):
        prob_re = l * 0.1 + 0.1

    nodes, edges = removing_edges(nodes_o, edges_o, prob_re, k_tree, k_add1_cliques, kk, random = False, clique = False)

    G = MarkovNetwork()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    factor_edges = [DiscreteFactor(edge, [2, 2], random(idx, 4)) for edge in edges]
    G.add_factors(*factor_edges)

    factor = cal_pr(G, variables = [ordering[-1]], evidence = None, elimination_order = ordering[:-1], joint = True, show_progress = False)
    max_prob, assignments = cal_mpe(G, variables = [ordering[-1]], evidence = None, elimination_order = ordering[:-1], joint = True, show_progress = False)

    writer = UAIWriter(G)
    writer.write_uai(path)
    pr = np.logaddexp(factor.values[0], factor.values[1]) / np.log(10)
    mpe = assignments
    with open(path+'.PR', 'w') as f:
        f.write("PR\n")
        f.write(str(pr))
    with open(path+'.MPE', 'w') as f:
        f.write("MPE\n")
        ls_mpe = list(mpe.values())
        ls_mpe.insert(0, len(mpe))
        str_mpe = ' '.join(str(v) for v in ls_mpe)
        f.write(str_mpe)