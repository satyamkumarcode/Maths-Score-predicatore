import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            logging.info("Starting training pipeline")

            # Data Ingestion
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            # Data Transformation
            data_transformation = DataTransformation()
            X_train_arr, X_test_arr, y_train, y_test = data_transformation.initiate_data_transformation(
                train_path,
                test_path
            )

            # Combine arrays for model trainer
            train_arr = np.c_[X_train_arr, np.array(y_train).reshape(-1, 1)]
            test_arr = np.c_[X_test_arr, np.array(y_test).reshape(-1, 1)]

            # Model Training
            model_trainer = ModelTrainer()
            r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f"Training pipeline completed with R² score: {r2_score}")
            print(f"Model trained successfully! R² Score: {r2_score}")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    TrainPipeline().run()
