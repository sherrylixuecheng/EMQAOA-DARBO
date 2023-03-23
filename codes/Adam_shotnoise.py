from functools import partial
import numpy as np
from scipy import optimize
import networkx as nx
import optax
import cotengra as ctg
import tensorcircuit as tc
from tensorcircuit import experimental as E
from tensorcircuit.applications.graphdata import maxcut_solution_bruteforce
import sys
# note this script only supports jax backend
K = tc.set_backend("jax")


opt_ctg = ctg.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel="ray",
    minimize="combo",
    max_time=120,
    max_repeats=128,
    progbar=True,
)
tc.set_contractor("custom", optimizer=opt_ctg, preprocessing=True)


mode = str(sys.argv[1])
graph_id = int(sys.argv[2])
seed = int(sys.argv[3])
nlayers = int(sys.argv[4])  # the number of layers
ncircuits = 1  # the number of circuits with different initial parameters
noiselevel =  int(sys.argv[5])
np.random.seed(seed)
i = graph_id
d = 3
n = 16
total_cycle=1000


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


rkey = K.get_random_state(42)


g = nx.read_weighted_edgelist(path='graph/{}_{}_{}'.format(d,n,i), delimiter=',', nodetype=int)
pss, ws = get_pauli_string(g)
init = np.load('initialization/X_init_weight_graph_p{}.npy'.format(nlayers), allow_pickle=True)[seed].reshape(nlayers,2)


@partial(K.jit, static_argnums=(2))
def exp_val(param, key, shots=noiselevel):
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

def exp_val_wrapper(param):
    global rkey
    rkey, skey = K.random_split(rkey)
    return exp_val(param, skey)

# QAOA with finite shot noise: gradient based
exponential_decay_scheduler = optax.exponential_decay(
    init_value=1e-2, transition_steps=500, decay_rate=0.9
)
opt = K.optimizer(optax.adam(exponential_decay_scheduler))
param = tc.array_to_tensor(init, dtype=tc.rdtypestr)
exp_grad = E.parameter_shift_grad_v2(
    exp_val, argnums=0, random_argnums=1, shifts=(0.001, 0.002)
)
# parameter shift doesn't directly apply in QAOA case
rkey = K.get_random_state(seed)
tot_loss = []
true_loss = []
for i in range(1000):
    rkey, skey = K.random_split(rkey)
    gs = exp_grad(param, skey)
    param = opt.update(gs, param)
    rkey, skey = K.random_split(rkey)
    tot_loss.append(float(exp_val(param, skey)))
    true_loss.append(float(exp_val_analytical(param)))
    print(true_loss[-1])
# the real energy position after optimization
print("converged as:", true_loss[-1])
np.save('../results/Adam_noise_{}_{}_{}_{}_v16p{}'.format('weight', graph_id, seed, noiselevel, nlayers), tot_loss)
np.save('../results/Adam_noise_true_{}_{}_{}_{}_v16p{}'.format('weight', graph_id, seed, noiselevel, nlayers), true_loss)
