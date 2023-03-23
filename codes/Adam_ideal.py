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
mode = str(sys.argv[1])
BO_method = sys.argv[2]
graph_id = int(sys.argv[3])
seed = int(sys.argv[4])
nlayers = int(sys.argv[5])  # the number of layers
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
QAOA_vvag1 = K.jit(tc.backend.vvag(QAOAansatz, argnums=0, vectorized_argnums=0))

params = tf.Variable(init_X.reshape(1, 2*nlayers))
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=100,
    decay_rate=0.95,
    staircase=False)
opt = K.optimizer(tf.keras.optimizers.Adam(learning_rate=lr_schedule))

ids = 0
tot_loss = []
for i in range(total_cycle):
    loss, grads = QAOA_vvag1(params, example_graph)
    print(i, K.numpy(loss))
    tot_loss.append(loss.numpy())
    params = opt.update(grads, params)  # gradient descent
tot_loss = np.array(tot_loss)
np.save('../results/Adam/Adam_{}_{}_{}_1000_v16p{}'.format(mode, graph_id, seed, nlayers), tot_loss)

