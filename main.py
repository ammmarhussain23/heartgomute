## defining the main

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predict import HeartDiseasePredictor


## creates FastAPI application object 
app = FastAPI(
    title = "Heart Disease Prediction API",
    description = "Predicts heart disease probability from clinical features",
    version = '1.0.0',
)

predictor = HeartDiseasePredictor('models')


class PatientData(BaseModel): # input validation schema
    Age: int = Field(..., ge=0, le=120) # ... - says this is a required field; if someone sends missing Age - API returns 400 error; ge - greater than or equal to 0; le - less than or equal to 120
    Sex: int = Field(..., ge=0, le=1)
    Chest_pain_type: int = Field(..., ge=1, le=4, alias='Chest pain type')
    BP: int = Field(..., ge=50, le=300)
    Cholesterol: int = Field(..., ge=100, le=700)
    FBS_over_120: int = Field(..., ge=0, le=1, alias="FBS over 120")
    EKG_results: int = Field(..., ge=0, le=2, alias="EKG results")
    Max_HR: int = Field(..., ge=50, le=250, alias="Max HR")
    Exercise_angina: int = Field(..., ge=0, le=1, alias="Exercise angina")
    ST_depression: float = Field(..., ge=0, le=10, alias="ST depression")
    Slope_of_ST: int = Field(..., ge=1, le=3, alias="Slope of ST")
    Number_of_vessels_fluro: int = Field(..., ge=0, le=3, alias="Number of vessels fluro")
    Thallium: int = Field(...)

    ## 

    class Config: 
        populate_by_name = True # without this True the API only accepts the alias names


class PredictionResponse(BaseModel):
    probability: float
    risk_level: str

@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API", "status": "running"} ## API health check - checks if server is running


@app.get("/health") ## verifies the model loaded correctly
def health_check():
    return {
        "status": "healthy", # this is static string confirming the endpoint
        "model_loaded": predictor is not None, #  checks if the HeartDiseasePredictor object was created successfully at startup
        "n_features": len(predictor.feature_names), # returns the feature count
    }


@app.post("/predict", response_model=PredictionResponse) # Client sends POST with JSON
def predict(patient: PatientData): # FastAPI validates against PatientData schema - rejects bad input with 400
    try:  ## convert Python names to original column names
        patient_dict = {
            "Age": patient.Age,
            "Sex": patient.Sex,
            "Chest pain type": patient.Chest_pain_type,
            "BP": patient.BP,
            "Cholesterol": patient.Cholesterol,
            "FBS over 120": patient.FBS_over_120,
            "EKG results": patient.EKG_results,
            "Max HR": patient.Max_HR,
            "Exercise angina": patient.Exercise_angina,
            "ST depression": patient.ST_depression,
            "Slope of ST": patient.Slope_of_ST,
            "Number of vessels fluro": patient.Number_of_vessels_fluro,
            "Thallium": patient.Thallium,
        }
        result = predictor.predict(patient_dict) ## predictor.predict() → preprocess → predict_proba
        return PredictionResponse(**result) ## return as JSON => client receives probability and risk level
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))