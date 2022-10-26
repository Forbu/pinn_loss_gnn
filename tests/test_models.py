# import lib
from pinn_loss import models

import jax.numpy as jnp

def test_MLP():
    """
    Here we test the MLP model
    """
    mlp = models.MLP(2, 2, 2, 2)

    # we test the forward pass
    x = jnp.ones((2, 2))
    y = mlp(x)

    assert y.shape == (2, 2)
    


def test_meshgraphnetmodel():

    # TODO we create the model


    # we generate a random graph data


    # we make a pass though the model and retrieve the correct dimension


    # assert the correct dimension
    pass

def test_training_mode():
    pass

def test_loss_operator():
    pass
