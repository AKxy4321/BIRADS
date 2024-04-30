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

def add_custom_fn_medium_deep(model, class_labels):                    #Shallow - 1 layer
    x = Flatten()(model.output)                                        #Deep - 3 or more layers
    x = Dense(2048, activation="relu")(x)                              #Small - 512, 256
    x = Dropout(0.3)(x)                                                #Medium - 1024, 2048
    x = Dense(1024, activation="relu")(x)                              #Large - 4096
    x = Dropout(0.3)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = (Dense(len(class_labels), activation="softmax"))(x)

    x = Model(model.input, x)
    return x

def add_custom_fn_medium_shallow(model, class_labels):
    x = Flatten()(model.output)
    x = Dense(2048, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = (Dense(len(class_labels), activation="softmax"))(x)

    x = Model(model.input, x)
    return x

def add_custom_fn_small_shallow(model, class_labels):
    x = Flatten()(model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = (Dense(len(class_labels), activation="softmax"))(x)

    x = Model(model.input, x)
    return x
