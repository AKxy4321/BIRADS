from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.model.ResNet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout

def build_model(input_shape, class_labels, weights_path):
    pretrained_weights = os.path.join('.', 'weights', 'resnet50_notop.h5')
    model = ResNet50(input_shape=input_shape, weights_path=pretrained_weights, choice="max")
    x = Flatten()(model.output)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = (Dense(len(class_labels), activation="softmax"))(x)

    model = Model(model.input, x)
    model.load_weights(weights_path)
    
    model.compile(
        loss="categorical_crossentropy",
        metrics="accuracy",
    )

    return model