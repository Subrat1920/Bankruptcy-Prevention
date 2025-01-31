import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## a dataclass in python decorator used to create class that primarily stores data. It automatically genereates special methods like __init__, __repr__, __eq__
@dataclass
class DataIngestionConfig:
    ## artifact folder is usually required to store the output of the data
    ### saving the train data in artifact
    train_data_path:str=os.path.join('artifact', 'train.csv')
    ### saving test data in artifact
    test_data_path:str=os.path.join('artifact', 'test.csv')
    ### we also need to save the whole data 
    raw_data_path:str=os.path.join('artifact', 'data.csv')

class DataIngestion:
    def __init__(self):
        ## calling the above class
        self.ingestion_config=DataIngestionConfig()
    
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Started/Entered')
        try:
            df = pd.read_csv('Notebook\Datasets\pre_processed_data.csv')
            logging.info('Dataset Imported to ')

            ## for training data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('train test split initiated')            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion and spliting is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )


        except Exception as e:
            raise CustomException(e, sys)


if __name__=='__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()