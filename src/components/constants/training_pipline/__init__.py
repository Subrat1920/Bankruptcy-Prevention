import os
import sys
import numpy as np
import pandas as pd


'''
Define common constant variable for training pipelin
'''
TARGET_COLUMN = "class"
PIPLINE_NAME = "Bankruptcy Security"
ARTIFACT_DIR = "Artifacts"
FILE_NAME = "pre_processed_data.csv"


'''
DATA INGESTION RELATED CONSTANT START WITH DATA_INGESTION VAR NAME
'''

DATA_INGESTION_COLLECTION_NAME:str='BankruptcyPrevention'
DATA_INGESTION_DATABASE_NAME:str='BankruptcyPreventionData'
DATA_INGESTION_DIR_NAME:str='data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR:str='feature_store'
DATA_INGESTION_INGESTED_DIR:str='ingested'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float=0.2
