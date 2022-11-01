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
import numpy as np

# from pytorch import dataset / dataloader / tensordataset
from torch.utils.data import DataLoader, TensorDataset
import torch

import mlflow

from pinn_loss import trainer, models

config_trainer = {
    "batch_size": 1,
    "learning_rate": 1e-3,
    "nb_epoch": 100,
    "save_model_every_n_epoch": 10,
    "save_log_step_every_n_step": 10,
}

def create_graph(nb_space, delta_x, delta_t, nb_nodes, nb_edges=None):
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
    dataset = np.cumsum(noise) * delta_x

    # now we have to create the graph associated to the dataset
    # we create the graph
    edges, edges_index = create_graph(nb_space, delta_x, delta_t)

    class BurgerDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, edges, edges_index):
            self.dataset = dataset
            self.edges = edges
            self.edges_index = edges_index

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return self.dataset[idx], self.edges, self.edges_index

    # create the dataloader
    dataloader = DataLoader(BurgerDataset(dataset, edges, edges_index), batch_size=batch_size, shuffle=True)

    return dataloader

def init_model(dataloader):

    # we retrieve only one element from the dataloader
    for batch in dataloader:
        break

    # we retrieve the data
    nodes = batch["nodes"]
    edges = batch["edges"]
    edges_index = batch["edges_index"]

    # get target
    target = batch["target"]

    # squeeze the first dimension for all the tensors
    nodes = nodes.squeeze(0)
    edges = edges.squeeze(0)
    edges_index = edges_index.squeeze(0)
    target = target.squeeze(0)

    nb_nodes = nodes.shape[0]
    nb_edges = edges.shape[0]

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

    config_input_init = {
        "nodes": (nb_nodes, 1),
        "edges": (nb_edges, 1),
        "edges_index": (nb_edges, 2),
    }

    # create the model
    state, model = trainer.create_train_state(config_model, config_trainer, config_input_init)

    return state, model

def main_train():

    """
    This function regroup all the main functions to train the GNN on the burger equation
    
    """

    # we choose the discretization of the space and the time
    nb_space = 100
    nb_time = 100

    delta_x = 1.0 / nb_space
    delta_t = 1.0 / nb_time

    # we choose the number of hidden layers and the number of hidden units
    nb_hidden_layers = 2
    nb_hidden_units = 32

    # we choose the number of iterations of the message passing
    nb_iterations = 5

    # we choose the number of epochs
    nb_epochs = 1000

    # we choose the learning rate
    learning_rate = 1e-3

    # we choose the batch size
    batch_size = 1

    # we choose the number of epochs between each save of the model
    save_model_every_n_epoch = 10
    
    # we choose the number of steps between each save of the log
    save_log_step_every_n_step = 10

    # we create space and time mesh
    space_mesh = np.linspace(0, 1, nb_space)

    # we create the initial condition
    initial_condition = np.sin(np.pi * space_mesh)

    ############### TRAINING ################
    # now we can create the dataloader
    dataloader = create_burger_dataset(nb_space, nb_time, delta_x, delta_t, batch_size=batch_size)

    # training scession with the dataloader and model with the pinn loss function
    state, model = init_model(dataloader)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow_logger = mlflow.tracking.MlflowClient()

    # we init the LightningFlax
    lightning_flax = trainer.LightningFlax(model, state, logger=mlflow_logger)

    

    ############### TESTING ################
    # here we will test the model on the custom initial condition


