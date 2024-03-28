from tensorflow.keras.optimizers.schedules import ExponentialDecay


def lr_custom(learning_rate_custom, decay_steps, decay_rate):
    return ExponentialDecay(
        initial_learning_rate=learning_rate_custom,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
    )


def lr_model(learning_rate_model, decay_steps, decay_rate):
    return ExponentialDecay(
        initial_learning_rate=learning_rate_model,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
    )
