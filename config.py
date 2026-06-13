# the config file for Heart Disease Prediction: shared constants (column names, params) and hyperparameters for the project

# for handling One Hot Encoding:

OHE_COLUMNS = ['Chest pain type', 'Thallium','Slope of ST', 'EKG results']
OHE_PREFIXES = ['cptype','thal','slopeST','ekg']

# for continuous features:
CONTINUOUS_FEATURES = ['Max HR','ST depression','Age','BP','Cholesterol'] # all features with numeric values

DROP_COLUMNS = ['id','Heart Disease'] # target 

# key hyperparameters for XGBoost Classifier:
XGB_PARAMS = {
    'n_estimators': 300, # number of boosting rounds
    'max_depth': 5, # maximum depth of each tree
    'learning_rate': 0.1, # how much the model learns from each tree
    'min_child_weight': 50, # minimum sum of instance weight needed in a child
    'subsample': 0.8, # fraction of samples used for training each tree
    'colsample_bytree': 0.8, # fraction of features used for training each tree
    'eval_metric': 'logloss', 
    'random_state': 42,
    'n_jobs': -1 
}

RANDOM_STATE = 42
N_FOLDS = 5

