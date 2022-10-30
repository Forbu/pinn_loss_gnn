import jax
import jax.numpy as jnp

import flax
from flax import linen as nn

import jraph

from pinn_loss import utils

def local_derivator(edges, sent_attributes, received_attributes, global_attributes):
    """
    This function compute the local derivative of the loss function
    """
    # compute the local derivative of the loss function
    # basicly 
    local_derivative = (sent_attributes - received_attributes) / edges

    return local_derivative

def global_derivator(nodes, sent_attributes,
                             received_attributes, global_attributes=None):
    """
    Simple mean between all the local derivator coming from the edges
    """
    return received_attributes

class DerivativeOperator(nn.Module):
    """
    This class compute the derivative of the function
    """
    index_edge_derivator: int
    index_node_derivator: int

    def setup(self):
        self.graph_derivative_operator = jraph.GraphNetwork(update_edge_fn=local_derivator, update_node_fn=global_derivator)

    def __call__(self, input_node, input_edge, graph_index):
        """
        Forward pass
        """
        # create the graph
        graph = jraph.GraphsTuple(nodes=input_node[:, self.index_node_derivator], edges=input_edge[:, self.index_edge_derivator], globals=None,
                                                 senders=graph_index[:, 0], receivers=graph_index[:, 1],
                                                 n_node=jnp.array([input_node.shape[0]]), n_edge=jnp.array([input_edge.shape[0]]))

        # apply the graph network
        y = self.graph_derivative_operator(graph)

        return y.nodes

class TemporalDerivativeOperator(nn.Module):
    """
    This class compute the temporal derivative of the Gnn (I/O)
    
    We suppose the input of the Gnn is the output of the previous Gnn

    This is a simple temporal derivative operator

    """
    index_node_derivator: int

    def setup(self):
        pass

    def __call__(self, input_node_t_1, input_node_t, delta_t):
        """
        Forward pass
        """
        # compute the temporal derivative
        y = (input_node_t - input_node_t_1) / delta_t

        return y[:, self.index_node_derivator]



    
    