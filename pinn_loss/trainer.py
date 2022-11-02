"""
Class and function to manage training of the model.
"""

import jax 
import jax.numpy as jnp

import flax
from flax import linen as nn
from flax.training import train_state

from flax import serialization

from functools import partial

from tqdm import tqdm

import numpy as np

import optax

from pinn_loss import models

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
def apply_model(state, nodes=None, edges=None, edges_index=None, target=None, model_main=None):
    """
    Computes gradients, loss and accuracy for a single batch.
    This function return the grad and loss for a single batch using classic supervised learning
    """
    def loss_fn(params):
        result = model_main.apply({'params': params}, nodes=nodes, edges=edges, edges_index=edges_index)
        loss = jnp.mean(optax.l2_loss(result, target))
        return loss, result

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, result), grads = grad_fn(state.params)
    return grads, loss

@partial(jax.jit, static_argnums=(4,))
def eval(params, nodes=None, edges=None, edges_index=None, model_all=None):
    """
    Rework for use case
    """
    result = model_all.apply({'params': params}, nodes=nodes, edges=edges, edges_index=edges_index)
    return result

class LightningFlax:
    """
    Class that manage the flax training in the same way that lightning does (but with jax this time)
    """
    def __init__(self, model, state, logger=None, config=None, log_every_n_step=100):
        
        self.model = model
        self.state = state
        self.config = config
        
        self.logger = logger
        self.log_every_n_step = log_every_n_step

        self.check_config()

    def check_config(self):
        pass

    def training_epoch(self):
        """
        Class that manage the training epoch
        """
        epoch_loss = []

        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            loss = self.training_step(batch, batch_idx)

            epoch_loss.append(loss)

            # Log metrics if loger is here
            if batch_idx % self.log_every_n_step == 0:
                self.log_metrics({"train_loss_step": loss})

        train_loss = np.mean(epoch_loss)
        
        # Log metrics
        self.log_metrics({"train_loss_epoch": train_loss})
        
        return train_loss

    def validation_epoch(self, batch, batch_idx):

        epoch_loss = []

        for batch_idx, batch in enumerate(tqdm(self.validation_loader)):
            loss = self.validation_step(batch, batch_idx)

            epoch_loss.append(loss)

            # Log metrics if loger is here
            if batch_idx % self.log_every_n_step == 0:
                self.log_metrics({"val_loss_step": loss})

        train_loss = np.mean(epoch_loss)

        # Log metrics
        self.log_metrics({"val_loss_epoch": train_loss})

        return train_loss

    def fit(self, train_loader, validation_loader=None, save_model_every_n_epoch=100, save_log_step_every_n_step=100, config_save=None):

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.save_model_every_n_epoch = save_model_every_n_epoch
        self.save_log_step_every_n_step = save_log_step_every_n_step

        validation = self.validation_loader is not None

        self.fit_init()

        if config_save is not None:
            self.config = config_save

        for epoch in range(self.config["nb_epoch"]):
            self.epoch = epoch

            train_loss = self.training_epoch()

            if validation:
                ## we loop over the validation
                valid_loss = self.validation_epoch()
                
            if self.logger is not None:
                ## we send the log to wandb
                print("training loss for the epoch {} : {}".format(epoch, train_loss))

                if validation:
                    print("validation loss for the epoch {} : {}".format(epoch, valid_loss))
                    self.logger.log({"train_loss": float(train_loss), "val_loss": float(valid_loss), "epoch": epoch})
                else:
                    self.logger.log({"train_loss": float(train_loss), "epoch": epoch})

                if (self.epoch % save_model_every_n_epoch) == 0: 
                    # save state.params using flax.serialization.to_bytes
                    dict_output = serialization.to_state_dict(self.state.params)

                    # save the dict
                    np.savez_compressed("model_epoch_{}.npz".format(self.epoch), **dict_output)

    def fit_init(self):
        """
        Init different things before running a fit (logger etc ...)
        """
        raise NotImplementedError

    def log_metrics(self, log_dict):
        """
        Log metrics to the logger
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

