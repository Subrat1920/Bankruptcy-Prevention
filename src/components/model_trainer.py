# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object, evaluate_model
# import pandas as pd
# import numpy as np
# import os, sys
# from dataclasses import dataclass
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from catboost import CatBoostClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.model_selection import RandomizedSearchCV

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join('artifacts', 'model.pkl')

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()
    
#     def initial_model_trainer(self, training_arr, test_arr):
#         try:
#             logging.info('Splitting Training and Test Input Data')
#             x_train, y_train = training_arr[:, :-1], training_arr[:, -1]
#             x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

#             logging.info('Splitting Training and Test Input Data Done')
#             models = {
#                 'Logistic Regression' : LogisticRegression(),
#                 'Decision Tree' : DecisionTreeClassifier(),
#                 'Random Forest' : RandomForestClassifier(),
#                 'AdaBoost' : AdaBoostClassifier(),
#                 'Gradient Boosting' : GradientBoostingClassifier(),
#                 'CatBoost' : CatBoostClassifier(),
#                 'XGBoost' : XGBClassifier(),
#                 'Support Vector Classfier' : SVC(),
#                 'K Nearest Neighbors' : KNeighborsClassifier(),
#             }

#             model_params = [
#                     ('Logistic Regression',
#                     LogisticRegression(),
#                     {
#                         'penalty': ['l1', 'l2', 'elasticnet'],
#                         'C': [0.1, 0.3, 0.5, 0.7, 0.9],
#                         'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
#                         'max_iter': [100, 300, 500, 700, 900]
#                     }),
#                     ('Decision Tree Classifier',
#                     DecisionTreeClassifier(),
#                     {
#                         'criterion': ['gini', 'entropy'],
#                         'splitter': ['best', 'random'],
#                         'max_depth': [x for x in range(1, 6)],  # Avoid 0
#                         'min_samples_split': [x for x in range(2, 10, 2)],
#                         'min_samples_leaf': [x for x in range(1, 10, 3)],
#                         'max_features': [x for x in range(1, 6)],  # Avoid 0
#                     }),
#                     ('Random Forest Classifier',
#                     RandomForestClassifier(),
#                     {
#                         'n_estimators': [x * 100 for x in range(1, 10, 2)],
#                         'criterion': ['gini', 'entropy', 'log_loss'],
#                         'max_depth': [x for x in range(1, 6)],  # Avoid 0
#                         'min_samples_split': [x for x in range(2, 10, 2)],
#                         'min_samples_leaf': [x for x in range(1, 10, 3)],
#                         'min_weight_fraction_leaf': [x / 10 for x in range(1, 10, 3)],
#                         'max_features': ['sqrt', 'log2', None],
#                     }),
#                     ('Gradient Boosting Classifier',
#                     GradientBoostingClassifier(),
#                     {
#                         'loss': ['log_loss', 'exponential'],
#                         'learning_rate': [x / 10 for x in range(1, 10, 3)],
#                         'n_estimators': [x * 100 for x in range(1, 10, 2)],
#                         'subsample': [x / 10 for x in range(1, 10, 3)],
#                         'criterion': ['squared_error'],  # Removed deprecated friedman_mse
#                         'min_samples_split': [x for x in range(2, 10, 2)],
#                         'min_samples_leaf': [x for x in range(1, 10, 3)],
#                         'min_weight_fraction_leaf': [x / 10 for x in range(1, 10, 3)],
#                         'max_depth': [x for x in range(3, 10, 2)],
#                         'max_features': ['sqrt', 'log2'],
#                     }),
#                     ('Support Vector Classifier',
#                     SVC(),
#                     {
#                         'C': [x / 10 for x in range(1, 10, 3)],
#                         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#                         'degree': [x for x in range(3, 10, 3)],
#                         'gamma': ['scale', 'auto'],
#                         'coef0': [x / 10 for x in range(1, 10, 3)],
#                         'tol': [1e-4, 1e-3],  
#                     }),
#                     ('K Nearest Neighbors Classifier',
#                     KNeighborsClassifier(),
#                     {
#                         'n_neighbors': [x for x in range(1, 10, 3)],
#                         'weights': ['uniform', 'distance'],
#                         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#                         'leaf_size': [x for x in range(30, 100, 20)],
#                         'p': [x for x in range(2, 10, 3)],
#                     }),
#                     ('Adaboost Classifier',
#                     AdaBoostClassifier(),
#                     {
#                         'n_estimators': [x * 100 for x in range(1, 10, 2)],
#                         'learning_rate': [x / 10 for x in range(1, 10, 2)],
#                         'algorithm': ['SAMME', 'SAMME.R'], 
#                     })
#                 ]


#             best_model_name = None
#             best_model_score = 0

#             for model_name, model, param_grid in model_params:
#                 logging.info(f"Training and tuning hyperparameters for {model_name}...")
#                 grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3, n_jobs=-1, verbose=2)
#                 grid_search.fit(x_train, y_train)

#                 logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
#                 logging.info(f"Best score for {model_name}: {grid_search.best_score_}")

#                 # Check if this model is the best one so far
#                 if grid_search.best_score_ > best_model_score:
#                     best_model_name = model_name
#                     best_model_score = grid_search.best_score_
#                     best_model = grid_search.best_estimator_
#             best_model = models[best_model_name]
#             if best_model_score < 0.6:
#                 raise CustomException('No Best Model Found')
#             logging.info(f'Best model found to be {best_model_name}')

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )
#             predicted = best_model.predict(x_test)
#             accuracy= accuracy_score(predicted, y_test)
#             return accuracy

#         except Exception as e:
#             raise CustomException(e, sys)


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
import os, sys
import numpy as np
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
from sklearn.model_selection import RandomizedSearchCV

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
            
            # Model and Hyperparameter Grid
            model_params = [
                ('Logistic Regression', LogisticRegression(),
                {
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'C': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
                    'max_iter': [100, 300, 500, 700, 900]
                }),
                ('Decision Tree', DecisionTreeClassifier(),
                {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [3, 5, 7, 9],
                    'min_samples_split': [2, 4, 6, 8],
                    'min_samples_leaf': [1, 3, 5, 7],
                    'max_features': ['sqrt', 'log2', None],
                }),
                ('Random Forest', RandomForestClassifier(),
                {
                    'n_estimators': [100, 300, 500, 700, 900],
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': [3, 5, 7, 9],
                    'min_samples_split': [2, 4, 6, 8],
                    'min_samples_leaf': [1, 3, 5, 7],
                    'max_features': ['sqrt', 'log2', None],
                }),
                ('Gradient Boosting', GradientBoostingClassifier(),
                {
                    'loss': ['log_loss', 'exponential'],
                    'learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'n_estimators': [100, 300, 500, 700],
                    'subsample': [0.5, 0.7, 0.9],
                    'min_samples_split': [2, 4, 6, 8],
                    'min_samples_leaf': [1, 3, 5, 7],
                    'max_depth': [3, 5, 7, 9],
                    'max_features': ['sqrt', 'log2'],
                }),
                ('Support Vector Classifier', SVC(),
                {
                    'C': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': [3, 5, 7],
                    'gamma': ['scale', 'auto'],
                    'coef0': [0.1, 0.3, 0.5],
                    'tol': [1e-4, 1e-3],  
                }),
                ('KNN', KNeighborsClassifier(),
                {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': [30, 50, 70, 90],
                    'p': [2, 3, 4],
                }),
                ('Adaboost', AdaBoostClassifier(),
                {
                    'n_estimators': [100, 300, 500, 700],
                    'learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'algorithm': ['SAMME', 'SAMME.R'], 
                })
            ]

            best_model_name = None
            best_model_score = 0
            best_model = None

            for model_name, model, param_grid in model_params:
                logging.info(f"Training {model_name}...")
                
                grid_search = RandomizedSearchCV(
                    estimator=model, 
                    param_distributions=param_grid, 
                    cv=3, 
                    n_jobs=-1, 
                    verbose=2
                )
                grid_search.fit(x_train, y_train)

                logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                logging.info(f"Best score for {model_name}: {grid_search.best_score_}")

                if grid_search.best_score_ > best_model_score:
                    best_model_name = model_name
                    best_model_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_

            if best_model_score < 0.6:
                logging.warning("No best model found with score > 0.6")
                return "No Best Model Found"

            logging.info(f"Best model selected: {best_model_name} with accuracy {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluate the model
            predicted = best_model.predict(x_test)
            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted, average='weighted')
            recall = recall_score(y_test, predicted, average='weighted')
            f1 = f1_score(y_test, predicted, average='weighted')

            logging.info(f"Model Performance - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
            
            return {
                "Model": best_model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1
            }

        except Exception as e:
            raise CustomException(e, sys)
