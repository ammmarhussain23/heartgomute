# Heart Disease Prediction

Predicting heart disease probability from clinical diagnostic features using machine learning. Built on the [Kaggle Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2) dataset (630,000 patients, 13 clinical features).

## Objective

Build a model that predicts the **probability** of heart disease presence given a patient's clinical test results — blood pressure, cholesterol, EKG readings, stress test outcomes, and cardiac imaging.

## Dataset Overview

| Property | Value |
|----------|-------|
| Training samples | 630,000 |
| Test samples | 270,000 |
| Features | 13 clinical measurements |
| Target | Heart Disease (Absence / Presence) |
| Class balance | 55% Absence / 45% Presence |

## Feature Analysis

Each feature was analyzed for clinical significance and predictive power. Features were ranked into three tiers based on correlation analysis, disease rate breakdowns, and medical domain knowledge.

### High Impact Features

| Feature | Type | Correlation | Why It Matters |
|---------|------|-------------|----------------|
| **Thallium** | Categorical (3, 6, 7) | Strongest predictor | Nuclear imaging stress test — reversible defects (value 7) indicate ischemia; 81.5% of patients with value 7 have heart disease vs 19.8% for normal (value 3) |
| **Chest Pain Type** | Categorical (1-4) | Strong +ve | Type 4 (asymptomatic) patients are 3x more likely to have disease (69.7%) — referred for cardiac eval based on other warning signs, not chest pain |
| **Max HR** | Continuous | Strong -ve (r = -0.44) | Higher max heart rate during exercise indicates the heart can handle stress — lower values signal disease |
| **Exercise Angina** | Binary | Strong +ve | Chest pain during a controlled stress test is a direct signal of coronary artery disease |
| **Number of Vessels (Fluoroscopy)** | Ordinal (0-3) | Strong +ve | More blocked vessels visible on fluoroscopy = more structural heart damage |
| **ST Depression** | Continuous | Strong +ve (r = 0.43) | ST segment depression on EKG during exercise indicates the heart is struggling to recover between beats |
| **Slope of ST** | Categorical (1-3) | Strong +ve | Flat/downsloping ST waves indicate poor cardiac recovery; 72% disease rate for downsloping vs 26% for upsloping |

### Moderate Impact Features

| Feature | Correlation | Note |
|---------|-------------|------|
| Sex | Moderate +ve | Males more susceptible; stronger independent effect than expected after controlling for other features |
| EKG Results | Moderate +ve | Weakest categorical predictor — only 20 percentage point spread across categories |
| Age | Moderate +ve (r = 0.21) | Older patients at higher risk, but age alone is not diagnostic |

### Low Impact Features

| Feature | Correlation | Note |
|---------|-------------|------|
| Cholesterol | Weak (r = 0.08) | Total cholesterol alone is a poor predictor — an LDL/HDL split would add more value |
| FBS over 120 | Weak | Diabetes is a risk factor but this binary feature captures little signal |
| BP | Near zero (r = -0.005) | Blood pressure is a long-term risk factor, not a diagnostic marker |

## Feature Engineering

| Transformation | Features | Reasoning |
|----------------|----------|-----------|
| One-Hot Encoding | Chest Pain Type, Thallium, Slope of ST, EKG Results | Numeric codes have no ordinal meaning — disease rates jump non-linearly across categories |
| Standard Scaling | Age, Max HR, ST Depression, BP, Cholesterol | Different scales (ST depression in units, cholesterol in hundreds) would bias linear models |
| Kept as-is | Sex, Exercise Angina, FBS over 120 | Already binary (0/1) |
| Kept as-is | Number of Vessels Fluro | True ordinal — disease rate increases monotonically with vessel count |

Final feature matrix: **22 features** after encoding (no features dropped — all inter-feature correlations below 0.5).

## Correlation Analysis

Inter-feature correlation was analyzed to identify redundancy and multicollinearity risk.

**Notable correlations:**
- ST Depression ↔ Slope of ST (r = 0.44) — both measure the same EKG phenomenon during exercise
- Thallium ↔ Exercise Angina (r = 0.36) — both capture stress-test pathology
- Thallium ↔ Number of Vessels (r = 0.33) — both measure structural heart damage

Thallium correlates with 5 other features, confirming it as the most central diagnostic signal. No pair exceeds r = 0.5, so all features were retained.

## Model Comparison

Three models were evaluated using 5-fold cross-validation:

| Model | AUC-ROC | Log-Loss | Accuracy | Notes |
|-------|---------|----------|----------|-------|
| Logistic Regression | 0.9527 +/- 0.0005 | 0.2759 | 0.8853 | Baseline — strong performance on linear relationships |
| Random Forest | 0.9507 +/- 0.0005 | — | — | Scored lower than LR — confirms the data has predominantly linear patterns |
| **XGBoost** | **0.9552 +/- 0.0004** | **0.2685** | **0.8884** | Best performer — sequential error correction captures subtle non-linear patterns |

### Key Findings

- **XGBoost wins, but marginally** (+0.0025 AUC over LR). The small gap indicates the feature-target relationships are largely linear — a result of well-designed clinical features.
- **Random Forest underperformed LR**, which is common on structured clinical data with pre-engineered diagnostic features.
- **All three models agree on feature importance rankings** — Thallium and Chest Pain Type dominate; BP and FBS are noise. This consistency validates the domain analysis.
- **Very low standard deviations** across all models confirm stable, reliable scores with no data leakage.

### Coefficient Validation (Logistic Regression)

Model coefficients were cross-checked against clinical domain knowledge:

| Feature | Coefficient | Clinical Expectation | Match |
|---------|------------|---------------------|-------|
| thal_3 (normal) | -1.39 | Healthy signal → negative | Yes |
| cptype_4 (asymptomatic) | +1.24 | Highest risk type → positive | Yes |
| Exercise angina | +1.20 | Disease signal → positive | Yes |
| Max HR | -0.84 | Higher HR = healthier → negative | Yes |
| BP | +0.005 | Weak predictor → near zero | Yes |

All coefficients align with medical literature, confirming the preprocessing pipeline is correct.

## Results

- **Final model:** XGBoost (AUC-ROC: 0.9552)
- **Test set predictions:** 270,000 patients scored with calibrated probabilities
- **Calibration check:** Mean predicted probability (0.4498) closely matches training prevalence (0.4483), confirming well-calibrated predictions

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, xgboost, seaborn, matplotlib
- **Jupyter Notebook** — exploratory data analysis and model development
- **FastAPI** — REST API for model serving

## Project Structure

```
├── Heart Disease Challenge-May30.ipynb   # EDA, feature analysis, model training
├── config.py                             # Shared constants and hyperparameters
├── preprocess.py                         # Feature encoding and scaling pipeline
├── train.py                              # Model training and artifact generation
├── predict.py                            # Prediction module
├── main.py                               # FastAPI application
├── requirements.txt                      # Python dependencies
└── README.md
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (generates models/ directory)
python train.py

# Start the API
uvicorn main:app --port 8080

# Test: visit http://localhost:8080/docs
```
