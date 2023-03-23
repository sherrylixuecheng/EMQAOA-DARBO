import sys
from functools import partial
import numpy as np
from scipy import optimize
import networkx as nx
import optax
import cotengra as ctg
import tensorcircuit as tc
from tensorcircuit import experimental as E
from tensorcircuit.applications.graphdata import maxcut_solution_bruteforce

# note this script only supports jax backend
K = tc.set_backend("jax")

# We use cotengra to speedup the experiments
opt_ctg = ctg.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel="ray",
    minimize="combo",
    max_time=10,
    max_repeats=128,
    progbar=True,
)
tc.set_contractor("custom", optimizer=opt_ctg, preprocessing=True)

# Define hyperparamters
d=3
n=16
methods=str(sys.argv[1])
graph_id = int(sys.argv[2])
nlayers=int(sys.argv[3])
seed = int(sys.argv[4])
noiselevel = int(sys.argv[5])
np.random.seed(seed)

rkey = K.get_random_state(seed)
g = nx.read_weighted_edgelist(path='graph/3_16_{}'.format(graph_id), delimiter=',', nodetype=int)
init = np.load('initialization/X_init_weight_graph_p{}.npy'.format(nlayers), allow_pickle=True)[seed].reshape(nlayers,2)

def get_graph(n, d, weights=None):
    g = nx.random_regular_graph(d, n)
    if weights is not None:
        i = 0
        for e in g.edges:
            g[e[0]][e[1]]["weight"] = weights[i]
            i += 1
    return g

def get_exact_maxcut_loss(g):
    cut, _ = maxcut_solution_bruteforce(g)
    totalw = 0
    for e in g.edges:
        totalw += g[e[0]][e[1]].get("weight", 1)
    loss = totalw - 2 * cut
    return loss

def get_pauli_string(g):
    n = len(g.nodes)
    pss = []
    ws = []
    for e in g.edges:
        l = [0 for _ in range(n)]
        l[e[0]] = 3
        l[e[1]] = 3
        pss.append(l)
        ws.append(g[e[0]][e[1]].get("weight", 1))
    return pss, ws

def generate_circuit(param, g, n, nlayers):
    # construct the circuit ansatz
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    for j in range(nlayers):
        c = tc.templates.blocks.QAOA_block(c, g, param[j, 0], param[j, 1])
    return c

def ps2xyz(psi):
    # ps2xyz([1, 2, 2, 0]) = {"x": [0], "y": [1, 2], "z": []}
    xyz = {"x": [], "y": [], "z": []}
    for i, j in enumerate(psi):
        if j == 1:
            xyz["x"].append(i)
        if j == 2:
            xyz["y"].append(i)
        if j == 3:
            xyz["z"].append(i)
    return xyz

@partial(K.jit, static_argnums=(2))
def exp_val(param, key, shots=noiselevel):
    # expectation with shot noise
    # ps, w: H = \sum_i w_i ps_i
    # describing the system Hamiltonian as a weighted sum of Pauli string
    c = generate_circuit(param, g, n, nlayers)
    loss = 0
    s = c.state()
    mc = tc.quantum.measurement_counts(
        s,
        counts=shots,
        format="sample_bin",
        random_generator=key,
        jittable=True,
        is_prob=False,
    )
    for psi, wi in zip(pss, ws):
        xyz = ps2xyz(psi)
        loss += wi * tc.quantum.correlation_from_samples(xyz["z"], mc, c._nqubits)
    return K.real(loss)

@K.jit
def exp_val_analytical(param):
    c = generate_circuit(param, g, n, nlayers)
    loss = 0
    for psi, wi in zip(pss, ws):
        xyz = ps2xyz(psi)
        loss += wi * c.expectation_ps(**xyz)
    return K.real(loss)

print("QAOA with shot noise")

def exp_val_wrapper(param):
    global rkey
    rkey, skey = K.random_split(rkey)
    # maintain stateless randomness in scipy optimize interface
    return exp_val(param, skey)


pss, ws = get_pauli_string(g)

exp_val_sp = tc.interfaces.scipy_interface(
    exp_val_wrapper, shape=[nlayers, 2], gradient=False
)
x = []
y = []

def new_fscipy_spsa(*args):
    global y
    new = exp_val_sp(*args)
    x.append(*args)
    print(new)
    return float(new)

def new_fscipy(*args):
    global y, x
    new = exp_val_sp(*args)
    x.append(*args)
    return new

if methods == 'COBYLA':
    r = optimize.minimize(
        new_fscipy,
        init,
        method="COBYLA",
        options={"maxiter": 1000},
    )
    print(r)
    x = np.array(x).reshape(len(x), nlayers, 2)
    ytrue = [float(exp_val_analytical(o)) for o in x] 
    print(ytrue)
elif methods == 'SPSA':
    from noisyopt import minimizeSPSA
    r = minimizeSPSA(new_fscipy_spsa, x0=init.ravel(), a=0.01, c=0.01, niter=500, paired=False)
    print(r)
    x = np.array(x).reshape(len(x), nlayers, 2)
    ytrue = [float(exp_val_analytical(o)) for o in x] 
    print(ytrue)
np.save('../results/{}/{}_noise_{}_{}_{}_{}_v16p{}'.format(methods, methods, 'weight', graph_id, seed, noiselevel, nlayers), ytrue)
