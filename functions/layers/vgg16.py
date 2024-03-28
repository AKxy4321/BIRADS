import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.model.VGG16 import VGG16
from functions.layers.custom import add_custom

def build_model(input_shape, class_labels, weights_path):
    pretrained_weights = os.path.join('.', 'weights', 'vgg16_notop.h5')
    model = VGG16(input_shape=input_shape, weights_path=pretrained_weights)
    model = add_custom(model, class_labels)
    model.load_weights(weights_path)

    model.compile(
        loss="categorical_crossentropy",
        metrics="accuracy",
    )

    return model