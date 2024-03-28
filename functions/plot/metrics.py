from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import pprint
import seaborn as sns
import numpy as np
import itertools


def save_metrics(
    true_labels, predicted_labels, confusion_matrix_path, report_file_path
):
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    cm = confusion_matrix(true_labels, predicted_labels)
    # print(cm)

    with open(report_file_path, "w") as file:
        pprint.pprint(report, indent=2, stream=file)

    # Plot confusion matrix for each batch
    plot_confusion_matrix(true_labels, predicted_labels, cm, confusion_matrix_path)


def plot_confusion_matrix(true_labels, predicted_labels, cm, confusion_matrix_path):
    """
    This function plots and saves the confusion matrix using Matplotlib.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Batch")
    plt.colorbar()

    classes = set(true_labels)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()
