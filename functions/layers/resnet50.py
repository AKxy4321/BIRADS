import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.model.ResNet50 import ResNet50
from functions.layers.custom import add_custom_fn

def build_model(input_shape, class_labels, weights_path):
    pretrained_weights = os.path.join('.', 'weights', 'resnet50_notop.h5')
    model = ResNet50(input_shape=input_shape, weights_path=pretrained_weights, choice="max")
    model = add_custom_fn(model, class_labels)
    model.load_weights(weights_path)
    
    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model