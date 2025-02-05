from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
import pandas as pd
import numpy as np
import os, sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initial_model_trainer(self, training_arr, test_arr):
        try:
            logging.info('Splitting Training and Test Input Data')
            x_train, y_train = training_arr[:, :-1], training_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info('Splitting Training and Test Input Data Done')
            models = {
                'Logistic Regression' : LogisticRegression(),
                'Decision Tree' : DecisionTreeClassifier(),
                'Random Forest' : RandomForestClassifier(),
                'AdaBoost' : AdaBoostClassifier(),
                'Gradient Boosting' : GradientBoostingClassifier(),
                'CatBoost' : CatBoostClassifier(),
                'XGBoost' : XGBClassifier(),
                'Support Vector Classfier' : SVC(),
                'K Nearest Neighbors' : KNeighborsClassifier(),
            }

            model_report:dict=evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException('No Best Model Found')
            logging.info(f'Best model found to be {best_model_name}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(x_test)
            accuracy= accuracy_score(predicted, y_test)
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)

