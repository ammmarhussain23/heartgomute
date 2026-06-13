# this is the preprocessing script for the Heart Disease project

# importing necessary libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import OHE_COLUMNS, OHE_PREFIXES, CONTINUOUS_FEATURES, DROP_COLUMNS  


# function for target encoding:
def encode_target(df):
    df = df.copy()

    if 'Heart Disease' in df.columns and not pd.api.types.is_numeric_dtype(df['Heart Disease']):
        df['Heart Disease'] = df['Heart Disease'].map({'Absence':0, 'Presence':1}) # converting target values from String to Numeric

    return df


## now creating the preprocessing pipeline:
def preprocess_features(df, scaler=None, fit=False):

    '''
    What the parameters mean:
    scaler: StandardScaler - pass None during training, returns fitted scaler
    fit - True during training, False during inference
    '''

    df = df.copy()

    df = pd.get_dummies(
        df, columns = OHE_COLUMNS, prefix = OHE_PREFIXES, drop_first = False
    ) # creating dummy variables for features that need OHE

    
    # for scaling features
    if fit: # if fit == True - training 
        scaler = StandardScaler()
        df[CONTINUOUS_FEATURES] = scaler.fit_transform(df[CONTINUOUS_FEATURES]) # fit learns the mean and std from train rows and transform uses learned values to scale each column to mean = 0 and std = 1 
    else: # False for test set
        df[CONTINUOUS_FEATURES] = scaler.transform(df[CONTINUOUS_FEATURES]) # transform uses the already fitted scaler tot scale the new data without learning new stats

    feature_names = [c for c in df.columns if c not in DROP_COLUMNS]
    df[feature_names] = df[feature_names].astype(float) # converting features to float type
    X = df[feature_names].values # df of all feature values

    return X, feature_names, scaler


## preprocess func for single patient dict for API - input is the dict from API
def preprocess_single_patient(patient_dict, scaler, feature_names):
    df = pd.DataFrame([patient_dict]) # converting dict to DF
    X, actual_names, _ = preprocess_features(df, scaler = scaler, fit = False)

    # get_dummies on a single row only creates columns for values present in that row
    # e.g. Chest pain type=4 only creates cptype_4, not cptype_1/2/3
    # so we rebuild the array to match the full 22 features the model expects
    result = np.zeros((1, len(feature_names)))
    for i, name in enumerate(feature_names):
        if name in actual_names:
            j = actual_names.index(name)
            result[0, i] = X[0, j]

    return result




