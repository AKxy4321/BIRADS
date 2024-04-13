from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

def add_custom_fn(model, class_labels):
    x = Flatten()(model.output)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = (Dense(len(class_labels), activation="softmax"))(x)

    x = Model(model.input, x)
    return x

def add_custom_fn_large(model, class_labels):
    x = Flatten()(model.output)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = (Dense(len(class_labels), activation="softmax"))(x)

    x = Model(model.input, x)
    return x
