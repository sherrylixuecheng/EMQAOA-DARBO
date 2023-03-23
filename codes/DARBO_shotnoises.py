"""
QAOA with finite measurement shot noise
"""
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



mode = str(sys.argv[1])
graph_id = int(sys.argv[2])
seed = int(sys.argv[3])
nlayers = int(sys.argv[4])  # the number of layers
ncircuits = 1  # the number of circuits with different initial parameters
noiselevel =  int(sys.argv[5])

#np.random.seed(seed)
i = graph_id
d = 3
n = 16
total_cycle=1000


opt_ctg = ctg.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel="ray",
    minimize="combo",
    max_time=120,
    max_repeats=128,
    progbar=True,
)

tc.set_contractor("custom", optimizer=opt_ctg, preprocessing=True)


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

def eval_objective(x, example_graph):
    a = tc.array_to_tensor(x, dtype=tc.rdtypestr).reshape(nlayers, 2)
    m = exp_val_wrapper(a)
    return -torch.Tensor(np.array(m))
def eval_objective_true(x, example_graph):
    a = tc.array_to_tensor(x, dtype=tc.rdtypestr).reshape(nlayers, 2)
    m = exp_val_analytical(a)
    return -torch.Tensor(np.array(m))

import odbo
import os
import torch
dim = 2 * nlayers
n_init = 2
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
batch_size = 1
acqfn='ucb'
switch = 'small'
switch_counter = 0
failure_tolerance = 10
t = time.time()
tr_length= [1.6]

paras = [np.array(init)]
X_new = np.random.uniform(low= 0, high= 1, size=[1, 2 * nlayers])
paras.append([2*np.pi*X_new])
X_turbo = torch.tensor(np.vstack([np.array(init).reshape(1, 2 * nlayers)/2/np.pi, X_new]))
Y_turbo = torch.tensor([eval_objective(x*2*np.pi, g) for x in X_turbo], dtype=dtype, device=device).unsqueeze(-1)
Y_true = torch.tensor([eval_objective_true(x*2*np.pi, g) for x in X_turbo], dtype=dtype, device=device).unsqueeze(-1)
bo_best = list(np.min(np.array(-Y_true), axis = 1))
state = odbo.turbo.TurboState(dim=X_turbo.shape[1], batch_size=batch_size, length=tr_length, n_trust_regions=len(tr_length), failure_tolerance = failure_tolerance)
state.best_value = Y_turbo.max()
print(bo_best)

for i in range(total_cycle-n_init):
    if switch_counter >=4:
        if switch == 'small':
            switch = 'large'
            X_turbo = X_turbo/2
        else:
            switch = 'small'
            X_turbo = X_turbo*2
        switch_counter = 0

    beta = 0.2
    X_next, acq_value, ids = odbo.run_exp.turbo_design(state=state,X=X_turbo,Y=Y_turbo, n_trust_regions=len(tr_length), batch_size=batch_size,a=beta, acqfn=acqfn, normalize=False, verbose=False)
    X_next = torch.reshape(X_next, [len(tr_length)*batch_size, 2*nlayers])
    if switch == 'small':
        print('small')
        Y_next = torch.tensor([eval_objective(x*np.pi-np.pi/2,g) for x in X_next], dtype=dtype, device=device)
        Y_next_true = torch.tensor([eval_objective_true(x*np.pi-np.pi/2,g) for x in X_next], dtype=dtype, device=device)
    else:
        print('large')
        Y_next = torch.tensor([eval_objective(x*2*np.pi-np.pi,g) for x in X_next], dtype=dtype, device=device)
        Y_next_true = torch.tensor([eval_objective_true(x*2*np.pi-np.pi,g) for x in X_next], dtype=dtype, device=device)

    # Update state
    state = odbo.turbo.update_state(state=state, Y_next=torch.reshape(Y_next, [len(tr_length), batch_size, 1]))

    if np.max(Y_next.numpy()) < np.max(np.array(Y_turbo)):
        switch_counter = switch_counter + 1
        print(switch_counter)
    else:
        switch_counter = 0
    if switch == 'small':
        paras.append([np.array(X_next)*np.pi-np.pi/2])
    else:
        paras.append([np.array(X_next)*2*np.pi-np.pi])
    X_turbo = torch.cat((X_turbo, X_next), dim=0)
    Y_turbo = torch.cat((Y_turbo, Y_next.unsqueeze(-1)), dim=0)
    Y_true = torch.cat((Y_true, Y_next_true.unsqueeze(-1)), dim=0)
    print(Y_next, Y_next_true, acq_value)
    bo_best.append(-Y_true.max())
  
    # Print current status
    print(f"{i+1}) Best value: {state.best_value:.4e}, TR length: {state.length}")

np.save('../results/Y_DARBO_noise_{}_{}_{}_{}_v16p{}'.format('weight', graph_id, seed, noiselevel, nlayers), np.array(Y_turbo))
np.save('../results/Y_DARBO_true_{}_{}_{}_{}_v16p{}'.format('weight', graph_id, seed, noiselevel, nlayers), np.array(Y_true))


