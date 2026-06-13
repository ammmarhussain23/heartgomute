## now creating and training the model

# importing libs
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from preprocess import encode_target, preprocess_features
from config import XGB_PARAMS, N_FOLDS, RANDOM_STATE

DATA_PATH = 'train_set.csv'
MODEL_DIR = 'models'


# func to load data:
def load_data(path = DATA_PATH):
    df = pd.read_csv(path)
    df = encode_target(df)
    print(f"Dataset has {len(df):,} rows")
    print(f"Target distribution: {df['Heart Disease'].mean():.4f} positive")

    return df # df with target encoded 

# func for running 5 fold CV 3 times once per metric
def evaluate_model(model, X,y):
    auc_scores = cross_val_score(model, X, y, cv = N_FOLDS, scoring='roc_auc') # returns 5 AUC-ROC scores
    logloss_scores = cross_val_score(model, X, y, cv = N_FOLDS, scoring = 'neg_log_loss') # measures prob calibration
    acc_scores = cross_val_score(model, X, y, cv = N_FOLDS, scoring = 'accuracy') # measures % correct

    print(f"AUC-ROC: {auc_scores.mean():.4f} +/- {auc_scores.std():4f}")
    print(f"Log-Loss: {-logloss_scores.mean():.4f} +/- {logloss_scores.std():.4f}")
    print(f"Accuracy: {acc_scores.mean():.4f} +/- {acc_scores.std():.4f}")

    return auc_scores.mean()

''' 
Alternatively in above function we can use cross_validate instead of cross_val_score

Above func runs CV 3 times thats 15 total train/test cycles - 5 for each model: LR, RF, XGB
With cross_validate we could compute all 3 metrics in a single CV loop - thats 5 mins of training instead of 15mins
'''

## save artifacts the API needs to make predictions
def save_artifacts(model, scaler, feature_names, model_dir = MODEL_DIR):
    os.makedirs(model_dir, exist_ok = True) # creates models/ folder if not already present
    joblib.dump(model, os.path.join(model_dir, 'model.joblib')) 
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.joblib'))
    print(f"\nArtifacts saved to {model_dir}")

'''
##What above function does:

joblib.dump - freezes this object to disk for later reload

model.joblib - contains trained XGB model with all its trees, splits and weights, API needs this to call the model on new patients
scaler.joblib - contains fitted StandardScaler with the learned mean & stdDev from train set, API needs this to scale new patient info based on train stats
feature_names.joblib - list of 22 columns to ensure new data columns are in same order the model expects 

Without this func we'd have to retrain the model every time the API would be called

models/ folder is the packaged trained model
'''

# main execution block of train.py - executes the full training pipeline
if __name__ == '__main__': # to make sure code only runs when train.py is executed directly; if another file calls evaluate_model it will only import the function w/o running the whole pipeline
    df = load_data() # reading train_set data
    X, feature_names, scaler = preprocess_features(df, fit=True) # carrying out preprocessing
    y = df['Heart Disease'].values 

    print(f"\n Feature matrix: {X.shape}")
    print(f"Features: {feature_names}\n")

    model = XGBClassifier(**XGB_PARAMS) # creating XGB model object, also unpacks config dictionary hyperparams

    print("Evaluating XGBoost (5 fold CV)")
    evaluate_model(model, X, y) # running 5 fold CV - is the sanity check 

    print('Training final model on all data')
    model.fit(X, y) # training final model

    save_artifacts(model, scaler, feature_names) # saving the model, scaler and feat_names to models/



