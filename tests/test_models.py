from pinn_loss import models

import jax.tree_util as tree
import jax.numpy as jnp
import jax

from jraph._src import utils

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

def test_edge_processor():

    # inti model
    edge_processor = models.EdgeProcessor(in_dims_node = 2, in_dims_edge = 2, hidden_dims = 2)

    # create PRNGKey
    rng = jax.random.PRNGKey(0)

    # create randn nodes features (10 nodes)
    nodes = jax.random.normal(rng, (10, 2))

    # create random edges between the nodes
    edges_index = jax.random.randint(key = rng, shape = (10, 2), minval = 0, maxval = 10)

    # create random edges features
    edges = jax.random.normal(rng, (10, 2))

    # retrieve the sent attributes from nodes features
    sent_attributes = nodes[edges_index[:, 0]]
    received_attributes = nodes[edges_index[:, 1]]

    # init weights
    weights = edge_processor.init(rng, edges, sent_attributes, received_attributes)

    # apply the model
    y = edge_processor.apply(weights, edges, sent_attributes, received_attributes)

    assert y.shape == (10, 2)

def test_node_processor():
    """
    Here we test the node processor part of the model (a bit more complicated)
    """

    # init model
    node_processor = models.NodeProcessor(in_dims_node = 2, in_dims_edge = 2, hidden_dims = 2)

    # create PRNGKey
    rng = jax.random.PRNGKey(0)

    # create randn nodes features (10 nodes)
    nodes = jax.random.normal(rng, (10, 2))

    # create random edges between the nodes
    edges_index = jax.random.randint(key = rng, shape = (10, 2), minval = 0, maxval = 10)

    # create random edges features
    edges = jax.random.normal(rng, (10, 2))

    nb_edge = 10

    senders = edges_index[:, 0]
    receivers = edges_index[:, 1]

    aggregate_edges_for_nodes_fn = utils.segment_sum

    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]

    # retrieve the sent attributes from edges features
    sent_attributes = tree.tree_map(
          lambda e: aggregate_edges_for_nodes_fn(e, senders, sum_n_node), edges)
    
    received_attributes = tree.tree_map(
          lambda e: aggregate_edges_for_nodes_fn(e, receivers, sum_n_node),
          edges)    

    # init weights
    weights = node_processor.init(rng, nodes, sent_attributes, received_attributes)

    # apply the model
    y = node_processor.apply(weights, nodes, sent_attributes, received_attributes)

    assert y.shape == (10, 2)


def test_graph_processor():
    """
    Here we test the graph processor module (the graph processor is the core of the model)
    """
    pass

def test_meshgraphnetmodel():
    """
    Here we test the full model
    """
    pass
