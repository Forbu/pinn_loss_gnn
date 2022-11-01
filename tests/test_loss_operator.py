from pinn_loss import loss_operator

import jax.tree_util as tree
import jax.numpy as jnp
import jax

import pytest

@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)

@pytest.fixture
def nodes(rng):
    return jax.random.normal(rng, (10, 2))

@pytest.fixture
def nodes_t_1(rng):
    return jax.random.normal(rng, (10, 2))

@pytest.fixture
def edges(rng):
    return jax.random.normal(rng, (10, 2))

@pytest.fixture
def edges_index(rng):
    return jax.random.randint(key = rng, shape = (10, 2), minval = 0, maxval = 10)

def test_local_derivator(nodes, edges, edges_index):
    """
    Here we test the local derivator
    """
    # create random sent attributes
    sent_attributes = nodes[edges_index[:, 0]]

    # create random received attributes
    received_attributes = nodes[edges_index[:, 1]]

    # create random global attributes
    global_attributes = None

    index_derivator_edges = 0
    index_derivator_nodes = 1

    # apply the local derivator
    y = loss_operator.local_derivator(edges[:, index_derivator_edges],
             sent_attributes[:, index_derivator_nodes], received_attributes[:, index_derivator_nodes], global_attributes)

    assert y.shape == (10,)

def test_global_derivator(nodes, edges, edges_index):

    index_derivator_edges = 0
    index_derivator_nodes = 1

    # create random global attributes
    global_attributes = None

    # apply the global derivator
    y = loss_operator.global_derivator(nodes[:, index_derivator_nodes], edges[:, index_derivator_edges], edges[:, index_derivator_edges], global_attributes)

    assert y.shape == (10,)


def test_loss_operator(rng, nodes, edges, edges_index):

    # init the DerivativeOperator (nn.Module)
    derivative_operator = loss_operator.DerivativeOperator(index_edge_derivator=0, index_node_derivator=1)

    # init the loss operator
    params = derivative_operator.init(rng, nodes, edges, edges_index)

    # apply the loss operator
    y = derivative_operator.apply(params, nodes, edges, edges_index)

    assert y.shape == (10,)

def test_temporal_derivative_operator(rng, nodes, edges, edges_index, nodes_t_1):
    """
    Here we test the temporal derivative operator
    We create two vectors (nodes) and we compute the temporal derivative
    """

    # create random nodes for t = 1
    nodes_t1 = nodes_t_1

    # init delta_t
    delta_t = 0.01

    # init the DerivativeOperator (nn.Module)
    derivative_operator = loss_operator.TemporalDerivativeOperator(index_node_derivator=0,delta_t=delta_t)

    # init the loss operator
    params = derivative_operator.init(rng, nodes, nodes_t1)

    # apply the loss operator
    y = derivative_operator.apply(params, nodes, nodes_t1)

    assert y.shape == (10,)

def test_burger_loss():
    """
    Here we want to test the BurgerLoss operator
    
    """
    


    


