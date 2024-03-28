import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os


def unfreeze_layers(
    model, num_layers_to_unfreeze
):  # 1 // 10 for small dataset  # 1 // 3 for medium dataset  # 8.5 // 10 for large dataset
    print("Unfreezing the convolution layers of VGG16...")
    for layer in model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True

    return model


def freeze_layers(model, choice):
    if choice == "all":
        print("Freezing All Model layers")
        for layer in model.layers:
            layer.trainable = False
        return model


def model_checkpt(early_stop_model, monitor):
    return ModelCheckpoint(
        early_stop_model,
        monitor=monitor,
        save_freq="epoch",
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
    )


def early_stop(monitor, patience):
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=0,
        restore_best_weights=True,
    )


def summary(model, n):
    if n == 1:
        with open("base_model_summary.txt", "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))
    if n == 2:
        with open("model_summary.txt", "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))


def get_unique_filename(base_filename):
    name, ext = os.path.splitext(base_filename)
    index = 1
    unique_filename = base_filename
    while os.path.exists(unique_filename):
        unique_filename = f"{name}_{index}{ext}"
        index += 1
    return unique_filename
