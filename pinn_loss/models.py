
from tkinter import HIDDEN
import jax
import jax.numpy as jnp

import flax
from flax import nn

from pinn_loss import utils

class MLP(nn.Module):
    """A simple MLP with x hidden layers."""
    nb_layers : int
    hidden_dims : int
    input_dims: int
    output_dims: int

    def setup(self):
        self.layers = [nn.Dense(self.hidden_dims, use_bias=False) for _ in range(self.nb_layers)]
        self.output_layer = nn.Dense(self.output_dims, use_bias=False)

    def __call__(self, x, activation=nn.relu, output_activation=None):
        """Forward pass."""
        for layer in self.layers:
            x = layer(x)
            x = activation(x)
        x = self.output_layer(x)
        if output_activation is not None:
            x = output_activation(x)
        return x

class GraphProcess(nn.Module):
    """
    In this module we will define the graph processor that is used to create the MeshGraphNetwork
    TODO
    """
    def __init__(self) -> None:
        super().__init__()
        pass

    def setup(self):
        pass

    def __call__(self):
        pass

# here we will define a graph neural network using flax and jax
# we will use the same graph neural network as in the paper
class ModelGnnPinn(nn.Module):
    """
    This model takes in domain and return the solution (numerical simulation of the PDE)
    This model don't need to use synthetic data to be trained as it is trained using the PINN loss

    This is a toy model in which we will use MeshGraphNetwork to solve the PDE (couple with PINN loss)
    """
    def __init__(self, ) -> None:
        super().__init__()

    def setup(self):
        """
        We have to define the edge encoder, node encoder, the graph processor and the node decoder
        """
        self.node_encoder = MLP(nb_layers=2, hidden_dims=32, input_dims=2, output_dims=32)
        self.edge_encoder = MLP(nb_layers=2, hidden_dims=32, input_dims=2, output_dims=32)
        self.graph_processor = GraphProcess() # TODO: define the graph processor
        self.node_decoder = MLP(nb_layers=2, hidden_dims=32, input_dims=32, output_dims=1)

    def __call__(self, input_node, input_edge, graph):
        """
        input_node: (batch_size, nb_nodes, 2)
        input_edge: (batch_size, nb_edges, 2)
        graph: (batch_size, nb_nodes, nb_nodes)
        """
        # encode the node and the edge
        node = self.node_encoder(input_node)
        edge = self.edge_encoder(input_edge)

        # process the graph
        node = self.graph_processor(node, edge, graph)

        # decode the node
        node = self.node_decoder(node)

        return node


    

