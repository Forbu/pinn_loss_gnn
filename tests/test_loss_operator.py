from pinn_loss import loss_operator

import jax.tree_util as tree
import jax.numpy as jnp
import jax

def test_local_derivator():
    """
    Here we test the local derivator
    """
    # create PRNGKey
    rng = jax.random.PRNGKey(0)

    # create random edges
    edges = jax.random.normal(rng, (10,))

    # create random sent attributes
    sent_attributes = jax.random.normal(rng, (10,))

    # create random received attributes
    received_attributes = jax.random.normal(rng, (10,))

    # create random global attributes
    global_attributes = None
    # apply the local derivator
    y = loss_operator.local_derivator(edges, sent_attributes, received_attributes, global_attributes)

    assert y.shape == (10,)

def test_global_derivator():

    # create PRNGKey
    rng = jax.random.PRNGKey(0)

    # create random nodes
    nodes = jax.random.normal(rng, (10,))

    # create random edges
    edges = jax.random.normal(rng, (10,))

    # create random global attributes
    global_attributes = None

    # apply the global derivator
    y = loss_operator.global_derivator(nodes, edges,
                             edges, global_attributes=None)

    assert y.shape == (10,)


def test_loss_operator():

    # create PRNGKey
    rng = jax.random.PRNGKey(0)

    # create random nodes
    nodes = jax.random.normal(rng, (10, 2))

    # create random edges
    edges = jax.random.normal(rng, (10, 2))

    # create random global attributes
    global_attributes = jax.random.normal(rng, (10, 2))

    # create random edges_index
    edges_index = jax.random.randint(key = rng, shape = (10, 2), minval = 0, maxval = 10)

    # create random sent attributes
    sent_attributes = nodes[edges_index[:, 0]]

    # create random received attributes
    received_attributes = nodes[edges_index[:, 1]]

    # create random target
    target = jax.random.normal(rng, (10, 2))

    # init the DerivativeOperator (nn.Module)
    derivative_operator = loss_operator.DerivativeOperator(index_edge_derivator=0, index_node_derivator=1)

    # init the loss operator
    params = derivative_operator.init(rng, nodes, edges, edges_index)

    # apply the loss operator
    y = derivative_operator.apply(params, nodes, edges, edges_index)

    assert y.shape == (10,)


def test_temporal_derivative_operator():
    """
    Here we test the temporal derivative operator
    We create two vectors (nodes) and we compute the temporal derivative
    """

    # create PRNGKey
    rng = jax.random.PRNGKey(0)

    # create random nodes for t = 0
    nodes = jax.random.normal(rng, (10, 2))

    # create random nodes for t = 1
    nodes_t1 = jax.random.normal(rng, (10, 2))

    # init delta_t
    delta_t = 0.01

    # init the DerivativeOperator (nn.Module)
    derivative_operator = loss_operator.TemporalDerivativeOperator(index_node_derivator=0)

    # init the loss operator
    params = derivative_operator.init(rng, nodes, nodes_t1, delta_t)

    # apply the loss operator
    y = derivative_operator.apply(params, nodes, nodes_t1, delta_t)

    assert y.shape == (10,)


    


