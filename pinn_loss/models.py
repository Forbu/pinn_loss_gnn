
import jax
import jax.numpy as jnp

import flax
from flax import nn

from pinn_loss import utils

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
        pass

    def __call__(self, input_node, input_edge, graph):
        pass


    

