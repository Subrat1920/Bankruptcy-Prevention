import sys,os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                industrial_risk:float,
                management_risk:float,
                financial_flexibility:float,
                credibility:float,
                competitiveness:float,
                operating_risk:float
                 ):
        self.industrial_risk=industrial_risk
        self.management_risk= management_risk
        self.financial_flexibility = financial_flexibility
        self.credibility = credibility
        self.competitiveness = competitiveness
        self.operating_risk = operating_risk

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "industrial_risk":[self.industrial_risk],
                "management_risk":[self.management_risk],
                "financial_flexibility":[self.financial_flexibility],
                "credibility":[self.credibility],
                "competitiveness":[self.competitiveness],
                "operating_risk":[self.operating_risk]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
