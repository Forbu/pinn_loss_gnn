# import lib
from pinn_loss import models
import jax.numpy as jnp
import jax

def test_MLP():
    """
    Here we test the MLP model
    """
    mlp = models.MLP(2, 2, 2, 2)

    # create PRNGKey
    rng = jax.random.PRNGKey(0)

    x = jnp.ones((2, 2))
    weights = mlp.init(rng, x)

    # we test the forward pass
    y = mlp.apply(weights, x)
    assert y.shape == (2, 2)
    
def test_meshgraphnetmodel():
    """
    Here we test the full model
    """
    pass

def test_graph_processor():
    """
    Here we test the graph processor module (the graph processor is the core of the model)
    """
    pass

def test_training_mode():
    pass

def test_loss_operator():
    pass
