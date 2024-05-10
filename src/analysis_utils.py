from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def split_mitocheck_features(
    profile: pd.DataFrame,
    feature_type: Optional[list[str]] = ["CP", "DP"],
) -> Tuple[list[str], list[str]]:
    """splits metadata and feature columns

    Parameters
    ----------
    profile : str
        profile that contains the population of cells or wells
    metadata_tag : Optional[bool], optional
        indicating if the metadata has the 'METADATA_' tag applied,
        by default False
    feature_type : Optional[list[str]], optional
        types of feature to select, by default ["CP", "DP"]
    """

    # type checking
    allowed_feature_types = ["CP", "DP"]
    if not isinstance(profile, pd.DataFrame):
        raise TypeError("'profile' must be a dataframe")
    if not isinstance(feature_type, list):
        raise TypeError("'feature_type' must be a list containing strings")
    if not any([isinstance(item, str) for item in feature_type]):
        print("eww")
    if not all(entry in allowed_feature_types for entry in feature_type):
        raise TypeError("'feature_type' most only containg 'CP', 'DP' or both")

    meta_features = []
    features = []
    for colname in profile.columns.tolist():
        if colname.startswith(f"{feature_type[0]}__"):
            features.append(colname)
        elif colname.startswith(f"{feature_type[1]}__"):
            features.append(colname)
        else:
            meta_features.append(colname)

    return (meta_features, features)


def plot_history(history):
    """Plot training and validation loss, as well as training and validation
    accuracy over epochs.

    Parameters
    ----------
    history : keras.callbacks.History
        A Keras History object containing training and validation metrics.

    Returns
    -------
    None
        This function does not return anything. It plots the training and
        validation metrics using Matplotlib.
    """
    # Extracting training history
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(1, len(train_loss) + 1)

    # Plotting loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, "b", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "r", label="Validation accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
