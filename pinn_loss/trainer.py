import jax 
import jax.numpy as jnp

import flax
from flax import nn
from flax.training import train_state, checkpoints

import tqdm

import numpy as np

class LightningFlax:
    """
    Class that manage the flax training in the same way that lightning does (but with jax this time)
    """
    def __init__(self, model, state, config, logger):
        
        self.model = model
        self.state = state
        self.config = config
        
        self.logger = logger

        self.check_config()

    def check_config(self):
        pass

    def training_epoch(self):
        """
        Class that manage the training epoch
        """

        epoch_loss = []

        for batch_idx, batch in enumerate(tqdm(self.train_load)):
            loss = self.training_step(batch, batch_idx)

            epoch_loss.append(loss)

        train_loss = np.mean(epoch_loss)
        return train_loss

    def validation_epoch(self, batch, batch_idx):

        epoch_loss = []

        for batch_idx, batch in enumerate(tqdm(self.validation_loader)):
            loss = self.validation_step(batch, batch_idx)

            epoch_loss.append(loss)

        train_loss = np.mean(epoch_loss)
        return train_loss

    def fit(self, train_loader, validation_loader, save_model_every_n_epoch=100, save_log_step_every_n_step=100):

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.save_model_every_n_epoch = save_model_every_n_epoch
        self.save_log_step_every_n_step = save_log_step_every_n_step

        validation = self.validation_loader is not None

        for epoch in range(self.config.num_epochs):
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
                    # use checkpoints.save_checkpoint to save the model
                    checkpoints.save_checkpoint(self.config.model_dir, self.state, epoch=self.epoch)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

