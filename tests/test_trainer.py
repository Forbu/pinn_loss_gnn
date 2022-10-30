"""
In this script we will test the training of the model with the trainer class
"""
import optax

import jax
import jax.numpy as jnp

import flax
from flax.training import train_state

from pinn_loss import trainer
from pinn_loss import models

from flax.training import checkpoints
from functools import partial

config_model = {
    "nb_layers": 2,
    "hidden_dims": 32,
    "input_dims_node_encoder":  2,
    "input_dims_edge_encoder":  2,
    "encoder_output_dims": 32,
    "input_dims_node_decoder": 32,
    "output_dims_node_decoder": 3,
    "mp_iteration": 5,
}

config_trainer = {
    "batch_size": 32,
    "learning_rate": 1e-3,
    "nb_epoch": 100,
    "save_model_every_n_epoch": 10,
    "save_log_step_every_n_step": 10,
}

config_input_init = {
    "nodes": (100, 2),
    "edges": (100, 2),
    "edges_index": (100, 2),
}

def create_train_state(rng, config_model, config_trainer, config_input_init):
    """Creates initial `TrainState`."""
    model_all = models.ModelGnnPinn(**config_model)

    nodes = jax.random.normal(rng, config_input_init["nodes"])
    edges = jax.random.normal(rng, config_input_init["edges"])
    edges_index = jax.random.randint(key = rng, shape = config_input_init["edges_index"], minval = 0, maxval = 100)

    params = model_all.init(rng, nodes=nodes, edges=edges, edges_index=edges_index)["params"]

    optimizer = optax.chain(
    optax.clip(1.0),
    optax.adam(learning_rate=config_trainer["learning_rate"]),
    )

    return train_state.TrainState.create(
        apply_fn=model_all.apply, params=params, tx=optimizer), model_all

@partial(jax.jit, static_argnums=(5,))
def apply_model(state, nodes=None, edges=None, edges_index=None, target=None, model_all=None):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
    result = model_all.apply({'params': params}, nodes=nodes, edges=edges, edges_index=edges_index)
    loss = jnp.mean(optax.l2_loss(result, target))
    return loss, result

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, result), grads = grad_fn(state.params)
  return grads, loss

def test_simple_training():
    """
    Testing simple training pass to test the performance of the model
    """
    pass
