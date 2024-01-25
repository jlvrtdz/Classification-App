import joblib
import torch

import pandas as pd
from preprocessing_utils import encode_features
from model import MLP, LGBM


def load_scaler(scaler_path):
    """
    Loads a pre-trained scaler from a joblib file.

    Args:
        scaler_path: The path to the joblib file containing the scaler.

    Returns:
        The loaded scaler.
    """
    return joblib.load(scaler_path, mmap_mode=None)

def predict_diagnosis_MLP(user_input_df, scaler, model=MLP):
    """
    Predicts the diagnosis for a given user input using a pre-trained model and scaler.

    Parameters: 
        user_input_df (pandas.DataFrame): A DataFrame containing the user input.
        scaler (object): The pre-trained scaler.
        model (torch.nn.Module, optional): The pre-trained model. Defaults to best_model.

    Returns:
        torch.Tensor: The predicted diagnosis tensor.
    """

    # Encode the categorical inputs using the 'encode_features' function
    encoded_user_input = encode_features(
        user_input_df,
        ordinal_features=[
            "Transparency",
            "Epithelial Cells",
            "Mucous Threads",
            "Amorphous Urates",
            "Bacteria",
            "Color",
            "Protein",
            "Glucose",
            "WBC",
            "RBC",
        ],
        nominal_features=["Gender"],
    )

    # Select the features to be scaled except Gender
    features_to_scale = encoded_user_input[
        ["Age", 
         "Color", 
         "Transparency", 
         "Glucose", 
         "Protein", 
         "pH", 
         "Specific Gravity", 
         "WBC", 
         "RBC", 
         "Epithelial Cells", 
         "Mucous Threads", 
         "Amorphous Urates", 
         "Bacteria"]
    ]

    # Apply the trained scaler to scale the selected features
    scaled_features = scaler.transform(features_to_scale)

    # Create a DataFrame with scaled features and Gender
    scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale.columns)

    # Concatenate the scaled features DataFrame with the "FEMALE" column, maintaining DataFrame structure
    combined_features = pd.concat(
        [
            scaled_df,
            encoded_user_input[["FEMALE"]],
        ],
        axis=1,
    )

    # Convert the combined features DataFrame to a NumPy array and then to a float32 tensor
    to_tensor = combined_features.values.astype('float32')

    # All features transformed to a tensor
    combined_features_tensor = torch.tensor(to_tensor)

    # Make a prediction using the pre-trained model
    pred = model(combined_features_tensor)

    return pred

def predict_diagnosis_LGBM(user_input_df, scaler, model=LGBM):
    """
    Predicts the diagnosis for a given user input using a pre-trained model and scaler.

    Parameters: 
        user_input_df (pandas.DataFrame): A DataFrame containing the user input.
        scaler (object): The pre-trained scaler.
        model (torch.nn.Module, optional): The pre-trained model. Defaults to best_model.

    Returns:
        torch.Tensor: The predicted diagnosis tensor.
    """

    # Encode the categorical inputs using the 'encode_features' function
    encoded_user_input = encode_features(
        user_input_df,
        ordinal_features=[
            "Transparency",
            "Epithelial Cells",
            "Mucous Threads",
            "Amorphous Urates",
            "Bacteria",
            "Color",
            "Protein",
            "Glucose",
            "WBC",
            "RBC",
        ],
        nominal_features=["Gender"],
    )

    # Select the features to be scaled except Gender
    features_to_scale = encoded_user_input[
        ["Age", 
         "Color", 
         "Transparency", 
         "Glucose", 
         "Protein", 
         "pH", 
         "Specific Gravity", 
         "WBC", 
         "RBC", 
         "Epithelial Cells", 
         "Mucous Threads", 
         "Amorphous Urates", 
         "Bacteria"]
    ]

    # Apply the trained scaler to scale the selected features
    scaled_features = scaler.transform(features_to_scale)

    # Create a DataFrame with scaled features and Gender
    scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale.columns)

    # Concatenate the scaled features DataFrame with the "FEMALE" column, maintaining DataFrame structure
    combined_features = pd.concat(
        [
            scaled_df,
            encoded_user_input[["FEMALE"]],
        ],
        axis=1,
    )

    # Make a prediction using the pre-trained model
    pred = model.predict_proba(combined_features)

    return pred


