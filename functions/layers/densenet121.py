import sys
import os
from tensorflow.keras.applications import DenseNet121 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.layers.custom import add_custom_fn, add_custom_fn_medium_shallow, add_custom_fn_medium_deep, add_custom_fn_small_shallow

def build_model(input_shape, class_labels, weights_path):
    model = DenseNet121 (
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="max"
        )

    model = add_custom_fn(model, class_labels)
    model.load_weights(weights_path)

    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
