

import joblib
import pandas as pd
from preprocessing_utils import encode_features

def load_model(model_path):
    return joblib.load(model_path, mmap_mode=None)

def predict_diagnosis(model, user_input_df):
    # Encode the categorical inputs using the 'encode_features' function
    encoded_user_input = encode_features(
        user_input_df,
        ordinal_features=["Transparency", "Epithelial_Cells", "Mucous_Threads", "Amorphous_Urates", "Bacteria",
                          "Color", "Protein", "Glucose", "WBC", "RBC"],
        nominal_features=["Gender"]
    )

    # Make a prediction using the pre-trained model
    pred = model.predict(encoded_user_input)

    return pred
