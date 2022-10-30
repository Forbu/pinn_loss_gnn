"""
In this script we will test the training of the model with the trainer class
"""

def create_train_state(rng, config, params_start=None):
  """Creates initial `TrainState`."""
  model_all = FNO2d(nb_layer=config.nb_layer, modes=config.modes,
                    nb_dim_fno=config.nb_dim_fno, output_dim=1, input_dim=3, embedding_dim=config.embedding_dim)
  
  params = model_all.init(rng, building=building, altitude=altitude, buildingh=buildingh)["params"]

  scheduler_1 = optax.constant_schedule(config.learning_rate)
  scheduler_2 = optax.constant_schedule(config.learning_rate/2)
  scheduler_3 = optax.constant_schedule(config.learning_rate/4)
  scheduler_all = optax.join_schedules([scheduler_1, scheduler_2, scheduler_3], [243*200, 243*200])


  optimizer = optax.chain(
  optax.clip(1.0),
  optax.adam(learning_rate=scheduler_all),
  )

  if not params_start:
    return train_state.TrainState.create(
        apply_fn=model_all.apply, params=params, tx=optimizer), model_all

def test_simple_training():
    """
    Testing simple training pass
    """
    pass
