import tensorflow as tf
from tensorflow.keras.optimizers import Adam
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
    visible_gpus = gpus[:2]  # Choose the number of GPUs you want to use
    tf.config.experimental.set_visible_devices(visible_gpus, "GPU")

    print("GPU will be used.")
else:
    print("No GPU available. Using CPU.")


from functions.model.misc import (
    model_checkpt,
    summary,
    freeze_layers,
    get_unique_filename,
)
from functions.model.ResNet50 import ResNet50
from functions.layers.custom import add_custom_fn
from functions.generator.generators import (
    train_gen,
    validation_gen,
    ImageDG_no_processed,
)
from functions.learning_rate_scheduler.lr_scheduler import lr_custom


def main():
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        name = "data"
        train_dir = os.path.join(".", "dataset", f"{name}_split", "train")
        val_dir = os.path.join(".", "dataset", f"{name}_split", "validation")
        batch_size = 32
        input_shape = (224, 224, 3)
        size = (224, 224)
        custom_epochs = 1
        monitor = "loss"
        pretrained_weights = os.path.join('.', 'weights', 'resnet50_notop.h5')
        base_filename = f"{name}_resnet50.h5"
        early_stop_model = (  #get_unique_filename
            os.path.join(".", "weights", base_filename)
        )

        print("Defining Model Checkpoint Callback")
        model_checkpoint = model_checkpt(early_stop_model, monitor)

        print("Applying Data Augmentation Techniques")
        datagen = ImageDG_no_processed()

        train_generator = train_gen(train_dir, datagen, size, batch_size)

        train_size = len(train_generator.filenames)
        initial_learning_rate = 0.001
        final_learning_rate = 0.00001
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

        model = ResNet50(input_shape=input_shape, weights_path=pretrained_weights, choice="max")
        summary(model, 1)
        model = freeze_layers(model, "all")

        print("Adding custom layers with regularization to the base model...")
        model = add_custom_fn(model, class_labels=class_labels)

        summary(model, 2)

        print("Training the custom layers...")

        model.compile(
            optimizer=Adam(learning_rate=lr_scheduler_custom),
            loss="categorical_crossentropy",
            metrics="accuracy",
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
                model_checkpoint,
            ],
        )

        save_model = os.path.join(".", "weights")
        if not os.path.exists(save_model):
            os.makedirs(save_model)


if __name__ == "__main__":
    main()
