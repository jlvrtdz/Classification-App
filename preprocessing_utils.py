import numpy as np
import pandas as pd

def encode_features(dataframe_, ordinal_features, nominal_features):
    """
        Encodes the nominal and ordinal features of the dataset using map function for ordinal and One-Hot Encoding for nominal.

        Parameters
        ------------
        dataframe_ : pandas.core.frame.DataFrame
            the dataframe that contains the nominal and orginal features

        ordinal_features : list, tuple, set
            a list of strings that corresponds to the column name of the ordinal features

        nominal_features : list, tuple, set
            a list of strings that corresponds to the column name of the nominal features

        Returns
        ------------
        dataframe_ : pandas.core.frame.DataFrame
            returns the preprocessed dataframe with all the ordinal and nominal features encoded
    """
    
    
    MAPPING_REFERENCE = {"ABSENCE_REFERENCE": {"NONE SEEN":0, 
                                               "RARE":1,
                                               "FEW":2,
                                               "OCCASIONAL":3,
                                               "MODERATE":4,
                                               "LOADED":5,
                                               "PLENTY":6},
                        
                        "Color": {"LIGHT YELLOW":0,
                                  "STRAW":1,
                                  "AMBER":2,
                                  "BROWN":3,
                                  "DARK YELLOW":4,
                                  "YELLOW":5,
                                  "REDDISH YELLOW":6,
                                  "REDDISH":7,
                                  "LIGHT RED":8,
                                  "RED":9},
                         
                        "Transparency": {"CLEAR":0,
                                         "SLIGHTLY HAZY":1,
                                         "HAZY":2,
                                         "CLOUDY":3,
                                         "TURBID":4},

                        "Protein_and_Glocuse": {"NEGATIVE":0,
                                                "TRACE":1,
                                                "1+":2,
                                                "2+":3,
                                                "3+":4,
                                                "4+":5}}
    
    SORTED_RANGED_VALUES = ["0-0", "0-1", "0-2", "1-2", "0-3", "0-4", "1-3", "2-3", "1-4", "2-4", 
                            "1-5", "2-5", "3-4", "1-6", "3-5", "2-6", "3-6", "2-7", "4-5", "3-7",
                            "4-6", "4-7", "5-6", "5-7", "4-8", "3-10", "5-8", "6-8", "4-10","5-10",
                            "7-8", "7-9", "7-10", "8-10", "8-11", "6-14", "8-12", "9-11", "9-12",
                            "7-15", "10-12", "9-15", "11-13", "10-15", "11-14", "10-16", "11-15",
                            "12-14", "12-15", "10-18", "13-15", "12-17", "14-16", "15-17", "15-18",
                            "16-18", "15-20", "15-21", "17-20", "15-22", "18-20", "18-21", "18-22",
                            "20-22", "15-28", "18-25", "20-25", "22-24", "23-25", "25-30", "25-32",
                            "28-30", "30-32", "28-35", "30-35", "34-36", "36-38", "35-40", "38-40",
                            "45-50", ">50", "48-55", "50-55", "48-62", "55-58", "70-75", "79-85",
                            "85-87", ">100", "LOADED", "TNTC"]
    
    
    for ordinal in ordinal_features:
        if ordinal in ["Epithelial Cells", "Mucous Threads", "Amorphous Urates", "Bacteria"]:
            dataframe_[ordinal] = dataframe_[ordinal].map(MAPPING_REFERENCE["ABSENCE_REFERENCE"])

        elif ordinal in ["Protein", "Glucose"]:
            dataframe_[ordinal] = dataframe_[ordinal].map(MAPPING_REFERENCE["Protein_and_Glocuse"])

        elif ordinal in ["WBC", "RBC"]:
            num_of_splits = 13
            splitted_array = np.hsplit(np.array(SORTED_RANGED_VALUES), num_of_splits)

            list_of_bin = [bin_0, bin_1, bin_2, bin_3, bin_4, bin_5, bin_6, bin_7, bin_8, bin_9, bin_10, bin_11, bin_12] = splitted_array
        
            def mapping_function(value):
                map_ = {tuple(bin_):i for i, bin_ in enumerate(list_of_bin)}

                for key, reference in map_.items():
                    if value in key:
                        return reference

            dataframe_[ordinal] = dataframe_[ordinal].map(mapping_function)
        else:
            dataframe_[ordinal] = dataframe_[ordinal].map(MAPPING_REFERENCE[ordinal])
            
    for nominal in nominal_features:
        gender = dataframe_[nominal].values[0]
        if  gender == "MALE":
            dataframe_["FEMALE"] = [False]
            dataframe_ = dataframe_.drop(nominal, axis=1)
        else:
            dataframe_["FEMALE"] = [True]
            dataframe_ = dataframe_.drop(nominal,axis =1)
            
    return dataframe_
