import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force TensorFlow to use GPU if available
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Set visible devices
    visible_gpus = gpus[:1]  # Choose the number of GPUs you want to use
    tf.config.experimental.set_visible_devices(visible_gpus, "GPU")

    print("GPU will be used.")
else:
    print("No GPU available. Using CPU.")


from functions.model.misc import (
    model_checkpt,
    summary,
    freeze_layers,
    get_unique_filename,
    early_stop
)
from functions.layers.custom import add_custom_fn, add_custom_fn_medium_shallow, add_custom_fn_medium_deep, add_custom_fn_small_shallow
from functions.generator.generators import (
    train_gen,
    validation_gen,
    ImageDG_no_processed,
)
from functions.learning_rate_scheduler.lr_scheduler import lr_custom


def main():
    name = "data"
    train_dir = os.path.join(".", "dataset", f"{name}_split", "train")
    val_dir = os.path.join(".", "dataset", f"{name}_split", "validation")
    batch_size = 16
    input_shape = (224, 224, 3)
    size = (224, 224)
    custom_epochs = 15
    monitor = "val_loss"
    patience = 5
    base_filename = f"{name}_densenet121.weights.h5"
    early_stop_model = (  #get_unique_filename
        os.path.join(".", "weights", base_filename)
    )

    print("Defining Model Checkpoint Callback")
    model_checkpoint = model_checkpt(early_stop_model, monitor)

    print("Defining Early Stop")
    early_stop_def = early_stop(monitor, patience)

    print("Applying Data Augmentation Techniques")

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_gen(train_dir, datagen, size, batch_size)

    train_size = len(train_generator.filenames)
    initial_learning_rate = 1e-3
    final_learning_rate = 1e-4
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (
        1 / custom_epochs
    )
    steps_per_epoch = int(train_size / batch_size)

    lr_scheduler_custom = lr_custom(
        initial_learning_rate, steps_per_epoch, learning_rate_decay_factor
    )

    class_labels = train_generator.class_indices
    print(class_labels)

    validation_generator = validation_gen(val_dir, datagen, size, batch_size)

    print("Loading the pre-trained model...")

    model = ResNet50(
        include_top=False,
    weights="imagenet",
    input_shape=input_shape,
    pooling="max"
    )
    summary(model, 1)

    model = freeze_layers(model, "all")

    print("Adding custom layers with regularization to the base model...")
    model = add_custom_fn(model=model, class_labels=class_labels)

    summary(model, 2)

    print("Training the custom layers...")

    model.compile(
        optimizer=Adam(learning_rate=lr_scheduler_custom),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # model_weights_path = os.path.join(
    #     ".", "weights", f"weights_{name}_categorical_1_1.h5"
    # )
    # if os.path.exists(model_weights_path):  # Manually choose which weights to load

    #     model.load_weights(model_weights_path)
    #     print("Weights loaded from previous training")

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=custom_epochs,
        callbacks=[
            model_checkpoint, early_stop_def
        ],
    )

if __name__ == "__main__":
    main()
