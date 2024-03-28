import os
import matplotlib.pyplot as plt
from ..metrics.f1_score import f1_score


def overwrite_existing_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def save_plot(history, title, save_train_plots):
    plt.figure(figsize=(8, 6))

    # Plot training F1 score and loss
    plt.plot(history.history["f1_score"], label="Train F1 Score")
    plt.plot(history.history["loss"], label="Train Loss")

    # Check if 'val_f1_score' is present in the history object
    if "val_f1_score" in history.history:
        # Plot validation F1 score and loss using 'val_f1_score'
        plt.plot(history.history["val_f1_score"], label="Validation F1 Score")
        plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.legend()

    # Create 'plots' folder if it doesn't exist
    if not os.path.exists(save_train_plots):
        os.makedirs(save_train_plots)

    plot_path = os.path.join(
        save_train_plots, f'{title.lower().replace(" ", "_")}_metrics.png'
    )
    overwrite_existing_file(plot_path)
    plt.savefig(plot_path)
    plt.close()
