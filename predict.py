## the prediction file: HeartDiseasePredictor class - loads saved artifacts, takes a patient dict and returns prob + risk level

# importing libs

import joblib
import numpy as np
import os
from preprocess import preprocess_single_patient

class HeartDiseasePredictor:

    # this class loads trained artifacts and predicts heart disease prob

    def __init__(self, model_dir = 'models'): # this loads all artifacts - the trained model, the standardScaler object, and feature_Names
        self.model = joblib.load(os.path.join(model_dir, 'model.joblib'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        self.feature_names = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))

    
    def predict(self, patient_dict):
        X = preprocess_single_patient(patient_dict, self.scaler, self.feature_names) # takes a raw dict, OHE's it, scales using train stats, returns a (1,22) numpy array

        probability = float(self.model.predict_proba(X)[0][1]) # predict_proba gives probabilities estimates for each class labels, [0] will give the 1st patient's prob as [0.04, 0.96] - that is .96 = Presence, 0.04 = Absence prob; [1] - will extrct just the 0.96 value 

        # to get a readable / actionable signal => creating risk ranges 
        if probability < 0.3:
            risk_level = 'low'
        elif probability < 0.6:
            risk_level = 'moderate'
        else:
            risk_level = 'high'

        return {
            "probability" : round(probability, 4), 
            "risk_level" : risk_level,
        } # sends back a float prob value to the API