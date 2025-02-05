import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Started')
        try:
            # Ensure correct path format
            dataset_path = os.path.join('Notebook', 'Datasets', 'pre_processed_data.csv')
            df = pd.read_csv(dataset_path)
            logging.info(f'Dataset loaded from {dataset_path}')

            # Ensure artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw data saved')

            # Train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Train-test split completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    try:
        # Data Ingestion
        obj = DataIngestion()
        train_data, test_data, raw_data = obj.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        # Model Training
        model_trainer = ModelTrainer()
        model_trainer.initial_model_trainer(train_arr, test_arr)

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise CustomException(e, sys)
