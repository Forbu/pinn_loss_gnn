# training / eval for burger equation

"""
This is the main file to train the GNN on the burger equation

The burger equation is a simple PDE that can be solved with a GNN (and a pinn loss).
The PDE is :
    u_t + u*u_x = 0

With the following boundary conditions:
    u(0, t) = 0
    u(1, t) = 0
    u(x, 0) = sin(pi*x)

The goal is to train a GNN to solve this PDE.

"""

# using pinn_loss without installing the library
import sys
sys.path.append("/home/")


import numpy as np

# from pytorch import dataset / dataloader / tensordataset
from torch.utils.data import DataLoader, TensorDataset
import torch

import mlflow
from mlflow.tracking import context, MlflowClient

from flax import linen as nn
from flax.training import train_state
from flax import serialization

from pinn_loss import trainer, models
from pinn_loss.loss_operator import BurgerLoss

import optax

import jax.numpy as jnp
import jax

import json

from functools import partial
import pickle
import argparse
from tqdm import tqdm

from matplotlib import pyplot as plt

config_trainer = {
    "batch_size": 1,
    "learning_rate": 1e-3,
    "nb_epoch": 1,
    "save_model_every_n_epoch": 10,
    "save_log_step_every_n_step": 10,
}

config_model = {
"nb_layers": 2,
"hidden_dims": 32,
"input_dims_node_encoder":  1,
"input_dims_edge_encoder":  1,
"encoder_output_dims": 32,
"input_dims_node_decoder": 32,
"output_dims_node_decoder": 1,
"mp_iteration": 5,
}

def create_graph(nb_space, delta_x, nb_nodes=None, nb_edges=None):
    """
    In the case of the burger equation, we have a 1D space : easy to create the graph (linear graph)
    """
    # create the edges features, size : ((nb_space - 1) * 2, 1)
    # the first (nb_space - 1) edges are the edges that goes from top to bottom
    # the second (nb_space - 1) edges are the edges that goes from bottom to top
    edges = np.zeros(((nb_space - 1) * 2, 1), dtype=np.float32)

    edges[:nb_space - 1, 0] = delta_x # positive edges
    edges[nb_space - 1:, 0] = -delta_x # negative edges

    # now we can create the edges_index values
    edges_index = np.zeros(((nb_space - 1) * 2, 2), dtype=np.int32)

    # we create the edges_index for the positive edges
    # the first column is the index of the node that is the source of the edge
    # the second column is the index of the node that is the target of the edge
    
    # the first (nb_space - 1) edges are the edges that goes from top to bottom
    # the second (nb_space - 1) edges are the edges that goes from bottom to top

    edges_index[:(nb_space - 1), 0] = np.arange(nb_space - 1)
    edges_index[:(nb_space - 1), 1] = np.arange(1, nb_space)

    # now we can create the edges_index for the negative edges
    edges_index[(nb_space - 1):, 0] = np.arange(1, nb_space)
    edges_index[(nb_space - 1):, 1] = np.arange(nb_space - 1)

    # we check the number of nodes and edges
    if nb_nodes is not None:
        assert nb_nodes == nb_space, "The number of nodes is not correct"
    if nb_edges is not None:
        assert nb_edges == (nb_space - 1) * 2, "The number of edges is not correct"

    return edges, edges_index

def create_burger_dataset(nb_space, delta_x, batch_size=1, size_dataset=10000):
    """
    Creation of the dataset for the burger equation
    We have to create a continuous function and the nn will have to approximate the next temporal step of the PDE
    """
    # create the space mesh
    space_mesh = np.linspace(0, 1, nb_space)

    # gausien noise
    noise = np.random.normal(0., 1., size=(size_dataset, nb_space))

    # we create the initial condition by adding a gaussian noise to create a random initial condition
    dataset = np.cumsum(noise, axis=1) * delta_x

    # now we have to create the graph associated to the dataset
    # we create the graph
    edges, edges_index = create_graph(nb_space, delta_x, nb_nodes=nb_space, nb_edges=(nb_space - 1) * 2)

    class BurgerDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, edges, edges_index):
            self.dataset = dataset
            self.edges = edges
            self.edges_index = edges_index

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return {"nodes" : self.dataset[idx], "edges" : self.edges, "edges_index" : self.edges_index}

    # we create another dataset with the some initial condition that look like a sinus
    def get_sinus_init(frequency, scale):
        initial_condition = np.sin(frequency * 2 * np.pi * space_mesh)
        return initial_condition * scale 
    
    # we generate size_dataset sinus initial condition
    # frequency are int between 1 and 5
    # scale are float between 0.5 and 5
    frequency = np.random.randint(1, 3, size=size_dataset)
    scale = np.random.uniform(0.5, 5., size=size_dataset)

    sinus_dataset = np.array([get_sinus_init(frequency[i], scale[i]) for i in range(size_dataset)])

    dataset_normal = BurgerDataset(dataset, edges, edges_index)
    dataset_sinus = BurgerDataset(sinus_dataset, edges, edges_index)

    # now we concat the two dataset
    dataset = dataset_sinus

    # create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def init_model_gnn(dataloader, delta_t=0.01, index_edge_derivator=0, index_node_derivator=0):
    """
    This function initialize the model (gnn) but also the burgerloss operator
    """

    # we retrieve only one element from the dataloader
    for batch in dataloader:
        break

    # we retrieve the data
    nodes = batch["nodes"]
    edges = batch["edges"]
    edges_index = batch["edges_index"]

    # squeeze the first dimension for all the tensors
    nodes = nodes.squeeze(0).unsqueeze(-1)
    edges = edges.squeeze(0)
    edges_index = edges_index.squeeze(0)

    # convert tensor to jnp array
    nodes = jnp.array(nodes)
    edges = jnp.array(edges)
    edges_index = jnp.array(edges_index)

    rng = jax.random.PRNGKey(0)

    # create the model
    model = models.ModelGnnPinn(**config_model)
    params = model.init(rng, nodes=nodes, edges=edges, edges_index=edges_index)["params"]

    # optimizer is chain with clip and adam
    optimizer = optax.chain(optax.clip(1.),
             optax.adam(config_trainer["learning_rate"]))

    #optimizer_accumulation = optax.MultiSteps(optimizer, every_k_schedule=8)

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer)

    # here we can also initialize the burger loss operator
    burger_loss = BurgerLoss(delta_t=delta_t, index_edge_derivator=index_edge_derivator, index_node_derivator=index_node_derivator)

    # we can also init the BurgerLoss
    params_burger = burger_loss.init(rng, nodes=nodes, edges=edges, edges_index=edges_index, nodes_t_1=nodes)

    return state, model, burger_loss, params_burger

@partial(jax.jit, static_argnums=(5, 6,))
def apply_model_derivative_target(state_main, params_burger=None, nodes=None, edges=None, edges_index=None, model_main=None, model_derivative=None):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params_main, params_derivative):
        prediction = model_main.apply({'params': params_main}, nodes=nodes, edges=edges, edges_index=edges_index)

        # compute derivative of the prediction
        loss_derivative = model_derivative.apply({'params': params_derivative}, nodes=prediction, edges=edges, edges_index=edges_index, nodes_t_1=nodes)

        loss = jnp.mean(optax.l2_loss(loss_derivative))
        return loss, prediction

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, result), grads = grad_fn(state_main.params, params_burger)
    return grads, loss

@partial(jax.jit, static_argnums=(4,))
def pred_gnn_model(params, nodes, edges, edges_index, model):
    pred = model.apply({'params': params}, nodes=nodes, edges=edges, edges_index=edges_index)
    return pred

@partial(jax.jit, static_argnums=(5, 6))
def eval_gnn_model(params, params_burger, nodes, edges, edges_index, model, model_derivative):
    pred = model.apply({'params': params}, nodes=nodes, edges=edges, edges_index=edges_index)
    loss_derivative = model_derivative.apply({'params': params_burger}, nodes=pred, edges=edges, edges_index=edges_index, nodes_t_1=nodes)
    loss = jnp.mean(optax.l2_loss(loss_derivative))
    return pred, loss

def save_params_into_file(params, path):
    """Save the params into a file using pickle"""
    with open(path, "wb") as f:
        pickle.dump(params, f)

def main_train():
    """
    This function regroup all the main functions to train the GNN on the burger equation
    """
    # we choose the discretization of the space and the time
    nb_space = 100
    nb_time = 100

    delta_x = 1.0 / nb_space
    delta_t = 1.0 / nb_time

    # we choose the batch size
    batch_size = 1

    ############### TRAINING ################
    # now we can create the dataloader
    dataloader = create_burger_dataset(nb_space, delta_x, batch_size=batch_size, size_dataset=10000)

    # training scession with the dataloader and model with the pinn loss function
    state, model, burger_loss, params_burger = init_model_gnn(dataloader, delta_t=delta_t)

    mlflow.set_tracking_uri("http://localhost:5000") # or what ever is your tracking url

    class BurgerLightningFlax(trainer.LightningFlax):
        def __init__(self, model, state, logger=None, config=None, params_burger=None):
            super().__init__(model, state, logger, config)
            self.params_burger = params_burger

        def fit_init(self):
            
            # set experiment (if not already set or already set to another experiment)
            if not mlflow.active_run() or mlflow.active_run().info.experiment_id != 1:
                mlflow.set_experiment("burger_loss")
            
            mlflow.start_run()

        def end_fit(self):
            pass

        def log_metrics(self, dict_log):
            
            # we log the metrics using mlflow log_metrics
            mlflow.log_metrics(dict_log)

        def training_step(self, batch, batch_idx):
            
            # we retrieve the data
            nodes = batch["nodes"]
            edges = batch["edges"]
            edges_index = batch["edges_index"]

            # squeeze the first dimension for all the tensors
            nodes = nodes.squeeze(0).unsqueeze(-1)
            edges = edges.squeeze(0)
            edges_index = edges_index.squeeze(0)

            # convert tensor to jnp array
            nodes = jnp.array(nodes)
            edges = jnp.array(edges)
            edges_index = jnp.array(edges_index)

            # we compute the gradient of the loss function
            grads, loss = apply_model_derivative_target(self.state, self.params_burger, nodes, edges, edges_index, self.model, burger_loss)

            # we update the parameters
            self.state = self.state.apply_gradients(grads=grads)

            return loss
        
        def validation_step(self, batch, batch_idx):
            # we retrieve the data
            nodes = batch["nodes"]
            edges = batch["edges"]
            edges_index = batch["edges_index"]

            # squeeze the first dimension for all the tensors
            nodes = nodes.squeeze(0).unsqueeze(-1)
            edges = edges.squeeze(0)
            edges_index = edges_index.squeeze(0)

            # convert tensor to jnp array
            nodes = jnp.array(nodes)
            edges = jnp.array(edges)
            edges_index = jnp.array(edges_index)

            pred, loss = eval_gnn_model(self.state.params, self.params_burger, nodes, edges, edges_index, self.model, burger_loss)

            return loss

    lightning_flax = BurgerLightningFlax(model, state, logger=mlflow, config=config_trainer, params_burger=params_burger)
    lightning_flax.fit(train_loader=dataloader, validation_loader=dataloader, config_save=config_trainer)

    # we save the model
    save_params_into_file(lightning_flax.state.params, "models_params/model_gnn.json")

def main_eval():
    # evaluation session with a custom limit condition
    # we load the model
    with open("models_params/model_gnn.json", "rb") as f:
        params = pickle.load(f)

    # we create the dataset
    nb_space = 100
    nb_time = 100

    delta_x = 1.0 / nb_space
    delta_t = 0.01

    # we choose the batch size
    batch_size = 1

    # we create space and time mesh
    space_mesh = np.linspace(0, 1, nb_space)

    # we create the initial condition
    initial_condition = np.sin(2 * np.pi * space_mesh)

    # we get the edges and the edges index
    edges, edges_index = create_graph(nb_space, delta_x=delta_x)

    # now we convert everything to jnp array
    nodes = jnp.array(initial_condition)

    # rework nodes to have the right shape (nb_nodes, 1)
    nodes = nodes.reshape(-1, 1)

    edges = jnp.array(edges)
    edges_index = jnp.array(edges_index)

    # init the model
    dataloader = create_burger_dataset(nb_space, delta_x, batch_size=batch_size, size_dataset=1000)

    # training scession with the dataloader and model with the pinn loss function
    state, model, burger_loss, params_burger = init_model_gnn(dataloader, delta_t=delta_t)

    eval_custom_initial_condition(model, params, nodes, edges, edges_index, nb_time, params_burger, burger_loss)
    eval_random_dataset(model, params, params_burger, burger_loss, dataloader)

def eval_custom_initial_condition(model, params, nodes, edges, edges_index, nb_time, params_burger, burger_loss):
    
    nb_space = nodes.shape[0]

    results = jnp.zeros((nb_space, nb_time + 1))
    results = results.at[:, 0].set(nodes[:, 0])

    # here we record the pde_loss
    pde_loss = jnp.zeros((nb_space, nb_time))

    # now we can apply the model recursively
    for i in tqdm(range(nb_time)):
        # we apply the model
        tmp = pred_gnn_model(params, nodes, edges, edges_index, model)

        # we compute the loss
        pde_loss_tmp = burger_loss.apply({'params': params_burger}, nodes=tmp, edges=edges, edges_index=edges_index, nodes_t_1=nodes)
        pde_loss = pde_loss.at[:, i].set(pde_loss_tmp)

        # we update the nodes
        nodes = tmp

        # we force the boundary condition and the first and last nodes
        nodes = nodes.at[0, 0].set(0)
        nodes = nodes.at[-1, 0].set(0)

        # we save the result using ops.index
        results = results.at[:, i + 1].set(nodes[:, 0])

    extend = 0, 1, 0, 1

    # we plot the results for the prediction
    plt.figure()
    plt.imshow(results, cmap="jet", extent=extend)

    # adding colorbar
    plt.colorbar()
    plt.show()

    # we save the image of the plot
    plt.savefig("plots/results_predictions.png")

    # we plot the results for the pde loss
    plt.figure()
    plt.imshow(pde_loss, cmap="jet", extent=extend)

    # adding colorbar
    plt.colorbar()
    plt.show()

    # we save the image of the plot
    plt.savefig("plots/results_pde_loss.png")

    # save metrics for comparaison
    pde_loss_metrics = jnp.mean(optax.l2_loss(pde_loss))

    # we save the metrics
    with open("metrics/metrics.json", "w") as f:
        json.dump({"pde_loss_custom_init": pde_loss_metrics.item()}, f)

def eval_random_dataset(model, params, params_burger, burger_loss, dataloader):
    """
    In this function we eval the performance of on a random dataset
    """
    performance_pde_loss = 0

    for i, batch in enumerate(tqdm(dataloader)):

        # we retrieve the data
        nodes = batch["nodes"]
        edges = batch["edges"]
        edges_index = batch["edges_index"]

        # squeeze the first dimension for all the tensors
        nodes = nodes.squeeze(0).unsqueeze(-1)
        edges = edges.squeeze(0)
        edges_index = edges_index.squeeze(0)

        # convert tensor to jnp array
        nodes = jnp.array(nodes)
        edges = jnp.array(edges)
        edges_index = jnp.array(edges_index)

        # directly compute the pde loss
        pred, pde_loss = eval_gnn_model(params, params_burger, nodes, edges, edges_index, model, burger_loss)

        performance_pde_loss += pde_loss.item()

    print("dataloader lenght : ", len(dataloader))

    pde_loss = performance_pde_loss / len(dataloader)
    pde_loss_dict = {"pde_loss_random_sample": pde_loss}

    print("The average pde loss is {}".format(pde_loss))

    # also adding pde_loss_dict to metrics/ folder
    with open("metrics/metrics_random_sample.json", "w") as f:
        json.dump(pde_loss_dict, f)

if __name__ == "__main__":

    # we retrieve the arguments to know if we are in training or testing mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train or test")

    args = parser.parse_args()

    if args.mode == "train":
        main_train()
    else:
        main_eval()

    mlflow.end_run()

