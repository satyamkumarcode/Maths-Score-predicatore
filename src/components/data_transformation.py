import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function creates and returns the preprocessing object
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function performs data transformation on train and test datasets
        """
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame shape: {train_df.shape}")
            logging.info(f"Test DataFrame shape: {test_df.shape}")

            target_column = "math_score"

            # Separate features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training dataframe")
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            
            logging.info("Applying preprocessing object on testing dataframe")
            X_test_arr = preprocessing_obj.transform(X_test)

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            logging.info("Data transformation completed successfully")
            
            # Return all 4 values explicitly
            return (
                X_train_arr,
                X_test_arr,
                np.array(y_train),
                np.array(y_test)
            )

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)