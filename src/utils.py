import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            # Special handling for CatBoost
            if model_name == "CatBoosting Regressor":
                # For CatBoost, use its native parameter tuning or simple fit
                if para:
                    # Try different parameter combinations manually
                    best_score = -np.inf
                    best_params = None
                    
                    from itertools import product
                    
                    # Get all parameter combinations
                    keys = para.keys()
                    values = para.values()
                    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
                    
                    for params in param_combinations:
                        temp_model = type(model)(**params, verbose=False)
                        temp_model.fit(X_train, y_train)
                        y_test_pred = temp_model.predict(X_test)
                        test_score = r2_score(y_test, y_test_pred)
                        
                        if test_score > best_score:
                            best_score = test_score
                            best_params = params
                    
                    # Train final model with best params
                    model.set_params(**best_params)
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
            else:
                # For other models, use GridSearchCV
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            # Update the model in the dictionary with the trained version
            models[model_name] = model

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)