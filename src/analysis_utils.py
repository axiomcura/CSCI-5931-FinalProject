from typing import Optional, Tuple

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
    if any([isinstance(item, str) for item in feature_type]):
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
