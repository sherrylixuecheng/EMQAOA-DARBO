import tensorcircuit as tc
import tensorflow as tf
# Could also use Jax backends
#import jax 
#import optax
import cotengra as ctg
import networkx as nx
import time 
import numpy as np
import torch
import matplotlib.pyplot as plt
import os 
import sys
K = tc.set_backend("tensorflow")
#K = tc.set_backend("jax")

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
graph_id = int(sys.argv[1])
seed = int(sys.argv[2])
nlayers = int(sys.argv[3])  # the number of layers
np.random.seed(seed)
i = graph_id

init_X = np.load('initialization/X_init_weight_graph_p{}.npy'.format(nlayers), allow_pickle=True)[seed]
example_graph = nx.read_weighted_edgelist(path='graph/{}_{}_{}'.format(d,n,i), delimiter=',', nodetype=int)

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

QAOA_vvag = K.jit(QAOAansatz)

def eval_objective(x, example_graph):
    a = tf.convert_to_tensor(np.array(x).ravel())
    return -torch.Tensor([QAOA_vvag(a, example_graph).numpy()])


import odbo
dim = 2 * nlayers
n_init = 2
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
batch_size = 1
ncluster_grid=[2,3,5]
acqfn='ucb'
switch = 'small'
switch_counter = 0
failure_tolerance = 10
t = time.time()
tr_length= [1.6]


paras = [np.array(init_X)]
X_new = np.random.uniform(low= 0, high= 1, size=[1, 2 * nlayers])
paras.append([2*np.pi*X_new])
X_turbo = torch.tensor(np.vstack([np.array(init_X).reshape(1, 2 * nlayers)/2/np.pi, X_new]))
Y_turbo = torch.tensor([eval_objective(x*2*np.pi, example_graph) for x in X_turbo], dtype=dtype, device=device).unsqueeze(-1)
bo_best = list(np.min(np.array(-Y_turbo), axis = 1))
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
       Y_next = torch.tensor([eval_objective(x*np.pi-np.pi/2,example_graph) for x in X_next], dtype=dtype, device=device)
    else:
       print('large')
       Y_next = torch.tensor([eval_objective(x*2*np.pi-np.pi,example_graph) for x in X_next], dtype=dtype, device=device)

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
    print(Y_next, acq_value)
    bo_best.append(-Y_turbo.max())
  
    # Print current status
    print(f"{i+1}) Best value: {state.best_value:.4e}, TR length: {state.length}")

np.save('../results/Y_DARBO_{}_{}_{}_1000_v16p{}'.format('weight', graph_id, seed, nlayers), np.array(Y_turbo))

