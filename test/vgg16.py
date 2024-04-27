import numpy as np
import sys
import os

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.metrics.f1_score_test import f1_score
from functions.plot.metrics import (
    save_metrics,
)
from functions.layers.vgg16 import build_model
from functions.generator.generators import test_gen


def test_saved_model(model, test_dir):

    batch_size = 16

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    data_exists = any(
        os.listdir(os.path.join(test_dir, class_dir))
        for class_dir in os.listdir(test_dir)
    )

    if data_exists:
        size = (224, 224)
        input_shape = (224, 224, 3)

        test_generator = test_gen(test_dir, datagen, size, batch_size)

        class_labels = test_generator.class_indices

        #Load Model
        model = build_model(input_shape=input_shape, class_labels=class_labels, weights_path=weights_path)

        # Evaluate the model on the testing data
        test_results = model.evaluate(test_generator)
        print(f"Test Loss:", test_results[0])

        # Make predictions on the testing data
        predictions = model.predict(test_generator)

        # Convert one-hot encoded labels back to categorical labels
        true_labels = test_generator.classes
        predicted_labels = np.argmax(predictions, axis=1)

        acc = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        print(f"Test Accuracy:", acc)
        print(f"Test F1 Score", f1)

        confusion_matrix_path = os.path.join(
            ".", "plots", "test", f"confusion_matrix.png"
        )

        report_file_path = os.path.join(
            ".", "plots", "test", f"classification_report.txt"
        )

        # Save classification report and plot confusion matrix for each batch
        save_metrics(
            true_labels,
            predicted_labels,
            confusion_matrix_path=confusion_matrix_path,
            report_file_path=report_file_path,
        )
    else:
        print("No images found in the data directory.")


if __name__ == "__main__":
    name = "data"
    test_dir = os.path.join(".", "dataset", f"{name}_split", "test")
    weights_path = os.path.join('.', 'weights', f'{name}_vgg16.weights.h5')

    # Create Test Plots folder if it doesn't exist
    save_test_plots = os.path.join(".", "plots", "test")
    if not os.path.exists(save_test_plots):
        os.makedirs(save_test_plots)

    # Wrap load_model call with custom_object_scope
    # with tf.keras.utils.custom_object_scope({"f1_score": f1_score}):
    test_saved_model(weights_path, test_dir)
