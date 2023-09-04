import numpy as np
import networkx as nx
import sys
d = 3
n = int(sys.argv[1])
seed = int(sys.argv[2])
np.random.seed(seed)

g = nx.random_regular_graph(d, n, seed=seed)
for (u,v) in g.edges():
    g.edges[u,v]['weight'] = np.random.rand()
nx.write_weighted_edgelist(G=g, path='{}_{}_{}'.format(d,n,seed), delimiter=',')
