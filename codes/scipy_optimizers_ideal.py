import tensorcircuit as tc
import jax
import cotengra as ctg
import networkx as nx
import time 
import numpy as np
import torch
import matplotlib.pyplot as plt
import os 
import sys
from scipy.optimize import minimize
K = tc.set_backend("jax")

# We use cotengra to speedup the experiments
opt_ctg = ctg.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel="ray",
    minimize="combo",
    max_time=1200,
    max_repeats=128,
    progbar=True,
)
tc.set_contractor("custom", optimizer=opt_ctg, preprocessing=True)

# Define hyperparamters
d = 3
n = 16
total_cycle=1000
ncircuits = 1  # the number of circuits with different initial parameters
methods = sys.argv[1] # L-BFGS-B or COBYLA or Nelder-Mead or SPSA
graph_id = int(sys.argv[2])
seed = int(sys.argv[3])
nlayers = int(sys.argv[4])  # the number of layers
np.random.seed(seed)
i = graph_id
# Load the presaved initializations & the graphs
init_X = np.load('initialization/X_init_weight_graph_p{}.npy'.format(nlayers), allow_pickle=True)[seed]
example_graph = nx.read_weighted_edgelist(path='graph/{}_{}_{}'.format(d,n,i), delimiter=',', nodetype=int)

# Define QAOA ansatz
def QAOAansatz(params, g=example_graph):
    n = len(g.nodes)  # the number of nodes
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    # PQC
    for j in range(nlayers):
        # U_j
        for e in g.edges:
            c.exp1(e[0], e[1], unitary=tc.gates._zz_matrix,
                theta=g[e[0]][e[1]].get("weight") * params[2 * j],)
        # V_j
        for i in range(n):
            c.rx(i, theta=params[2 * j + 1])
    # calculate the loss function
    loss = 0.0
    for e in g.edges:
        loss += g[e[0]][e[1]].get("weight") * c.expectation_ps(z=[e[0], e[1]])
    return K.real(loss)

# A optimization interface
y = []
n=0
y = []

def new_fscipy(*args):
    global y
    new = f_scipy(*args)
    print(new)
    y.append(new)
    return new

def new_fscipy_nm(*args):
    global y
    new = f_scipy(*args)
    print(new)
    y.append(new)
    return float(new[0])

f_scipy = tc.interfaces.scipy_optimize_interface(QAOAansatz, shape=[2*nlayers], jit=True)

if methods == 'L-BFGS-B':
    r = minimize(new_fscipy, init_X, method=methods, jac=True, options={'maxfun':total_cycle})
else:
    if methods == 'COBYLA':
        r = minimize(new_fscipy, init_X.ravel(), method=methods, options={'maxiter':total_cycle, 'tol': 0.0001})
    elif methods == 'Nelder-Mead':
        r = minimize(new_fscipy_nm, init_X.ravel(), method=methods, options={'maxfev':total_cycle})
    elif methods == 'SPSA':
        bounds = []
        for i in range(2*nlayers):
            bounds.append([0, 2*np.pi])
        bounds = np.array(bounds)
        print(bounds)
        from noisyopt import minimizeSPSA
        r = minimizeSPSA(new_fscipy_nm, bounds=bounds, x0=init_X.ravel(), a=0.01, c=0.01, niter=500, paired=False)
print(r)
np.save('../results/{}/{}_{}_{}_{}_1000_v16p{}'.format(methods, methods, 'weight', graph_id, seed, nlayers), y)
