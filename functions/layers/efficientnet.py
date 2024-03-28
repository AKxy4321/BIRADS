from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions.layers.custom import add_custom_fn



def build_model(input_shape, class_labels, weights_path):
    pretrained_weights = os.path.join('.', 'weights', 'efficientnetv2L_notop.h5')
    model = EfficientNetV2L(input_shape=input_shape, weights=pretrained_weights, pooling="max", include_top=False)
    model = add_custom_fn(model, class_labels)
    model.load_weights(weights_path)

    model.compile(
        loss="categorical_crossentropy",
        metrics="accuracy",
    )

    return model