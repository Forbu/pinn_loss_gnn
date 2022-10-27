"""
Main code to get a simple GNN model (MeshGraph model) to work
"""
import jax
import jax.numpy as jnp

import flax
from flax import linen as nn

# import jraph
import jraph

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

class NodeProcessor(nn.Module):
    """
    This class process the node features with all the edges features associated to it
    """
    in_dims_node: int
    in_dims_edge: int

    hidden_dims: int

    def setup(self):
        self.node_mlp = MLP(2, self.hidden_dims, self.in_dims_node + self.in_dims_edge, self.in_dims_node)

    def __call__(self, node_features, edge_features):
        """
        Forward pass
        """
        # scatter sum the edge features
        edge_features = jraph.ops.segment_sum(edge_features, node_features["edges"])

        # we concatenate the node features with the edge features
        x = jnp.concatenate([node_features, edge_features], axis=-1)
        x = self.node_mlp(x)
        return x

class EdgeProcessor(nn.Module):
    """
    This class process the node features with all the edges features associated to it
    """
    in_dims_node: int
    in_dims_edge: int

    hidden_dims: int

    def setup(self):
        self.node_mlp = MLP(2, self.hidden_dims, self.in_dims_node + self.in_dims_edge, self.in_dims_node)

    def __call__(self, nodes, sent_attributes,
                             received_attributes):
        """
        Forward pass
        """
        pass

class GraphProcess(nn.Module):
    """
    In this module we will define the graph processor that is used to create the MeshGraphNetwork
    """
    input_dims_node : int
    input_dims_edge : int

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
    nb_layers : int
    hidden_dims : int

    input_dims_node_encoder: int
    input_dims_edge_encoder: int

    input_dims_node_decoder: int
    output_dims_node_decoder: int

    output_dims: int

    mp_iteration: int

    def __init__(self, ) -> None:
        super().__init__()

    def setup(self):
        """
        We have to define the edge encoder, node encoder, the graph processor and the node decoder
        """
        self.node_encoder = MLP(nb_layers=self.nb_layers, hidden_dims=self.hidden_dims,
                                                             input_dims=self.input_dims_node_encoder, output_dims=32)
        self.edge_encoder = MLP(nb_layers=self.nb_layers, hidden_dims=self.hidden_dims,
                                                             input_dims=self.input_dims_node_encoder, output_dims=32)

        self.node_decoder = MLP(nb_layers=self.nb_layers, hidden_dims=self.hidden_dims, 
                                                             input_dims=self.input_dims_node_decoder, output_dims=self.output_dims_node_decoder)

        # TODO complete
        self.graph_processors = [jraph.GraphNetwork(update_edge_fn=EdgeProcessor(), update_node_fn=NodeProcessor()) for _ in range(self.mp_iteration)]

    def __call__(self, input_node, input_edge, graph):
        """
        input_node: (batch_size, nb_nodes, 2)
        input_edge: (batch_size, nb_edges, 2)
        graph: (batch_size, nb_nodes, nb_nodes)
        """
        # encode the node and the edge
        node = self.node_encoder(input_node)
        edge = self.edge_encoder(input_edge)

        # create the graph
        graph = jraph.GraphsTuple(nodes=node, edges=edge, globals=None,
                     n_node=jnp.array([node.shape[1]]), n_edge=jnp.array([edge.shape[1]]), senders= graph[0], receivers=graph[1])

        # process the graph
        for graph_processor in self.graph_processors:
            graph = graph_processor(graph)

        # we get the node features
        node = graph.nodes

        # decode the node
        node = self.node_decoder(node)

        return node


    

