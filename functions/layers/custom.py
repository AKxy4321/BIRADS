from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

def add_custom(model, class_labels):
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(len(class_labels), activation="softmax"))

    return model

def add_custom_fn(model, class_labels):
    x = Flatten()(model.output)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = (Dense(len(class_labels), activation="softmax"))(x)

    x = Model(model.input, x)
    return x