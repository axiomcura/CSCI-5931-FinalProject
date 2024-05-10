import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras.models import load_model
from tensorflow.keras.models import Model


def load_profile(profile_path: str | pathlib.Path) -> pd.DataFrame:
    """Loads image-based profiles in memory as a pandas dataframe

    Parameters
    ----------
    profile_path : pd.DataFrame
        path to input file

    Returns
    -------
    pd.DataFrame
        loaded image-based profile

    Raises
    -------
    TypeError
        raised if 'profile_path' is not a string or pathlib.Path object
    FileNotFoundError
        raised if the path in the 'profile_path' does not exist
    """

    # type checking
    if not isinstance(profile_path, (str, pd.DataFrame)):
        raise TypeError("'profile_path' must be a string or pathlib.Path object ")
    if isinstance(profile_path, str):
        profile_path = pathlib.Path(profile_path)

    # check if the file exists
    # this will raise a FileNotFoundError if the file path does not exist
    abs_path = profile_path.resolve(strict=True)

    # use pandas to read in the data
    profile_df = pd.read_csv(abs_path)
    return profile_df


def predict_probabilities(model: Model, profile: pd.DataFrame) -> pd.DataFrame:
    """Predict probabilities of cellular states.

    Parameters
    ----------
    model : Model
        The TensorFlow model used for prediction.
    profile : pd.DataFrame
        DataFrame containing cell profiles.

    Returns
    -------
    pd.DataFrame
        DataFrame containing predicted probabilities for each cellular state.

    Raises
    ------
    TypeError
        If `model` is not an instance of tensorflow.keras.models.Model.
    TypeError
        If `profile` is not a pandas DataFrame.
    """
    # type checking
    if not isinstance(model, Model):
        raise TypeError("'model' must be a tensorflow model")
    if not isinstance(profile, pd.DataFrame):
        raise TypeError("'profile' must be a pd.DataFrame")

    # loading in cell_state code
    cell_state_encoders = pathlib.Path("./configs/cell_state_codes.json")
    with open(cell_state_encoders, "r") as f:
        cell_state_codes = json.load(f)

    # Perform prediction using the loaded model
    proba_df = pd.DataFrame(model.predict(profile))
    proba_df.columns = [
        cell_state_codes["decoder"][str(code)] for code in proba_df.columns.tolist()
    ]
    proba_df.insert(0, "predicted_state", proba_df.idxmax(axis=1))

    return proba_df


def plot_proportion(proba_df: pd.DataFrame, save_path: str | pathlib.Path) -> None:
    """plots the proportion of cell states within the given dataset as a pie chart.

    Parameters
    ----------
    proba_df : pd.DataFrame
        probabilities
    save_path : str | pathlib.Path
        where to save the file
    """

    # create counts of the predicted states
    label_counts = proba_df["predicted_state"].value_counts()

    # Plotting with Seaborn
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")
    sns.color_palette("pastel")
    plt.pie(label_counts, labels=label_counts.index, autopct="%1.1f%%", startangle=140)
    plt.title("Distribution of Predicted Labels", pad=20, fontweight="bold")
    plt.axis("equal")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def save_results(proba_df: pd.DataFrame, outname: str) -> None:
    """Save predicted states and population proportions.

    Parameters
    ----------
    proba_df : pd.DataFrame
        DataFrame containing predicted states for each cell.
    outname : str
        Name used for saving the results.

    Returns
    -------
    None
    """
    # create results directory
    results_dir = pathlib.Path("./results").resolve()
    results_dir.mkdir(exist_ok=True)

    # create save paths
    predicted_path = (results_dir / f"{outname}_predicted_states.csv").resolve()
    proportions_path = results_dir / f"{outname}_population_proportions.csv"

    # save predicted states per cell
    predicted_states = (
        proba_df["predicted_state"]
        .to_frame()
        .reset_index()
        .rename(columns={"index": "cell"})
    )
    predicted_states.to_csv(predicted_path, index=False)
    print(f"MESSAGE: Predicted cellular states: {predicted_path}")

    # save the proportion of cellular states of the population
    counts = proba_df["predicted_state"].value_counts()
    proportion = (
        round((counts / counts.sum()) * 100)
        .to_frame()
        .reset_index()
        .rename(columns={"count": "percentage"})
    )
    proportion.to_csv(proportions_path, index=False)
    print(
        f"MESSAGE: Proportion of cellular state within the population saved at: {proportions_path}"
    )


def main(image_profile: pd.DataFrame, outname: str) -> None:
    """_summary_

    Parameters
    ----------
    css_file : _type_
        _description_
    out_file : _type_
        _description_
    """
    # loaded pre-trained model
    model_path = pathlib.Path(
        "./notebooks/3.training_model/cell_state_identifier.keras"
    ).resolve(strict=True)
    model = load_model(model_path)

    # loading profile
    loaded_profile = load_profile(image_profile)

    # Perform prediction using the loaded model
    probabilities = predict_probabilities(model, loaded_profile)

    # plot the proportions of cell states
    plot_proportion(proba_df=probabilities, save_path=outname)

    # Save the results
    save_results(probabilities, outname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict cellular states probabilities from an image-based profile."
    )
    parser.add_argument(
        "-i", "--input", type=str, help="Inputs image based profile file"
    )
    parser.add_argument(
        "-o",
        "--outname",
        type=str,
        help="Output file name for saving the predicted probabilities",
    )
    args = parser.parse_args()

    main(args.input, args.outname)
