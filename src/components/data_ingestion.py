import pandas as pd
import numpy as np
import pymongo.mongo_client

from src.exception import CustomException
from src.logger import logging

## call the config file for Data Ingestion Configuration
from src.entity.config_entity import DataIngestionConfig
from src.entity.artificat_entity import DataIngestionArtifact

import os
import sys
import pymongo
from typing import List
from sklearn.model_selection import train_test_split

## load the data from mongodf
from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

## read the data
class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
            self.mongo_client =  pymongo.MongoClient(MONGO_DB_URL)

        except Exception as e:
            raise CustomException(e, sys)
    
    def export_collection_as_dataframe(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            collection=self.mongo_client[database_name][collection_name]
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df.drop(columns=['_id'], axis=1, inplace=True)
            df.replace({"na":np.nan}, inplace=True)
            return df 

        except Exception as e:
            raise CustomException(e, sys)


    def export_data_to_feature_store(self, dataframe:pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            ## recreating the folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise CustomException(e, sys)


    def split_data_as_train_test(self, dataframe:pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size= self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed Train Test Split on the Dataframe")
            logging.info('Exited split_data_as_train_test method of Data_Ingestion class')

            dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting Train and Test File Path")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info('Exported train and test  to file path')

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe=self.export_data_to_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path, test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info('Retreived the data training and testing artifact')
            return dataingestionartifact


        except Exception as e:
            raise CustomException(e, sys)
