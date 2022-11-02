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

config_trainer = {
    "batch_size": 1,
    "learning_rate": 1e-3,
    "nb_epoch": 100,
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

def create_burger_dataset(nb_space, nb_time, delta_x, delta_t, batch_size=1, size_dataset=100000):
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

    # create the dataloader
    dataloader = DataLoader(BurgerDataset(dataset, edges, edges_index), batch_size=batch_size, shuffle=True)

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

    optimizer = optax.chain(
    optax.clip(1.0),
    optax.adam(learning_rate=config_trainer["learning_rate"]),
    )

    state, model = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer), model

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

def save_params_into_file(params, path):
    """Save the params into a file"""
    with open(path, "wb") as f:
        json.dump(params, f)

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

    # we create space and time mesh
    space_mesh = np.linspace(0, 1, nb_space)

    # we create the initial condition
    initial_condition = np.sin(np.pi * space_mesh)

    ############### TRAINING ################
    # now we can create the dataloader
    dataloader = create_burger_dataset(nb_space, nb_time, delta_x, delta_t, batch_size=batch_size)

    # training scession with the dataloader and model with the pinn loss function
    state, model, burger_loss, params_burger = init_model_gnn(dataloader)

    mlflow.set_tracking_uri("file://home/mlruns/") # or what ever is your tracking url

    class BurgerLightningFlax(trainer.LightningFlax):
        def __init__(self, model, state, logger=None, config=None, params_burger=None):
            super().__init__(model, state, logger, config)
            self.params_burger = params_burger

        def fit_init(self):
            
            # set experiment (if not already set or already set to another experiment)
            if not mlflow.active_run() or mlflow.active_run().info.experiment_id != 1:
                mlflow.set_experiment("burger_loss")
            
            mlflow.start_run()

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

    lightning_flax = BurgerLightningFlax(model, state, logger=mlflow, config=config_trainer, params_burger=params_burger)
    lightning_flax.fit(train_loader=dataloader, config_save=config_trainer)

    # now we can save the model in state.params
    dict_output = serialization.to_state_dict(state.params)

    # we save the model
    save_params_into_file(dict_output, "models_params/model_gnn.json")

if __name__ == "__main__":
    main_train()