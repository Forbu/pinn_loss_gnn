"""
In this script we will test the training of the model with the trainer class
"""
import optax

import jax
import flax

from pinn_loss import trainer
from pinn_loss import models

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

def create_train_state(rng, config_model, config_trainer):
  """Creates initial `TrainState`."""
  model_all = models.ModelGnnPinn(**config_model)
  
  params = model_all.init(rng, building=building, altitude=altitude, buildingh=buildingh)["params"]

  optimizer = optax.chain(
  optax.clip(1.0),
  optax.adam(learning_rate=config_trainer["learning_rate"]),
  )

  if not params_start:
    return train_state.TrainState.create(
        apply_fn=model_all.apply, params=params, tx=optimizer), model_all

def test_simple_training():
    """
    Testing simple training pass
    """
    pass
