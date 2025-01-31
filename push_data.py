import os
import sys
import json

import certifi

import pandas as pd
import numpy as np
import pymongo
from src import logger
from src import exception

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

## ca (Certificate Authorities)
ca = certifi.where()

class BankruptcyPrevention():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise exception(e, sys)
        
    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise exception(e, sys)
        
    def insert_data_into_mongodf(self, records, database, collection):
        ## here collection is a table that is present in mysql
        try:
            self.database = database
            self.records = records
            self.collection = collection

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise exception(e, sys)

if __name__=='__main__':
    FILE_PATH = 'Notebook\Datasets\pre_processed_data.csv'
    DATABASE = 'CleanedBankruptcyPrevention'
    Collection = 'BankruptcyData'
    bankjObj = BankruptcyPrevention()
    records = bankjObj.csv_to_json_converter(FILE_PATH)
    no_of_records = bankjObj.insert_data_into_mongodf(records, DATABASE, Collection)
    print(records)
    print(no_of_records)


