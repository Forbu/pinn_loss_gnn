"""
In this script we will test the training of the model with the trainer class
"""
import optax

import jax
import jax.numpy as jnp

import flax


from pinn_loss import trainer, models

from flax.training import train_state
from functools import partial

import mlflow

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

def test_simple_training():
    """
    Testing simple training pass to test the performance of the model
    """
    
    # create PRNGKey
    rng = jax.random.PRNGKey(0)

    nb_nodes = 100

    # create random nodes
    nodes = jax.random.normal(rng, (nb_nodes, 2))

    # create random edges
    edges = jax.random.normal(rng, (nb_nodes, 2))

    # create random edges index
    edges_index = jax.random.randint(key = rng, shape = (nb_nodes, 2), minval = 0, maxval = nb_nodes)

    # create random target
    target = jax.random.normal(rng, (nb_nodes, 3))

    # create train state
    model_all = models.ModelGnnPinn(**config_model)
    params = model_all.init(rng, nodes=nodes, edges=edges, edges_index=edges_index)["params"]

    optimizer = optax.chain(
    optax.clip(1.0),
    optax.adam(learning_rate=config_trainer["learning_rate"]),
    )

    state, model_all = train_state.TrainState.create(
        apply_fn=model_all.apply, params=params, tx=optimizer), model_all

    # mlflow set tracking uri localhost:5000
    # mlflow.set_tracking_uri("http://localhost:5000")
    # mlflow_logger = mlflow.tracking.MlflowClient()

    # init trainer class
    trainer_module = trainer.LightningFlax(model_all, state, config_trainer, None)

    # init batch
    batch = {
        "nodes": nodes,
        "edges": edges,
        "edges_index": edges_index,
        "target": target,
    }

    # train model with simple training step
    # trainer_module.training_step(batch, batch_idx=0)


