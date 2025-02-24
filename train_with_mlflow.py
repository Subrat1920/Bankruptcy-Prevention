import sys, os
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from src.exception import CustomException
from src.logger import logging
import warnings
warnings.filterwarnings('ignore')


# Function to evaluate model performance
def evaluate_models(actual, predicted):
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    return accuracy, precision, f1

def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError as e:
        logging.error(f'Pickle File not found at: {file_path}')
        raise e

def preprocess_data(csv_file_path, scaler_path, encoder_path):
    try:
        df = pd.read_csv(csv_file_path)
        logging.info(f'Loaded data from csv file: {csv_file_path}')
        if 'class' not in df.columns:
            raise ValueError("The 'class' column is not present in the dataset.")
        x = df.drop(columns='class', axis=1)
        y = df['class']
        logging.info('Separated Dependent and Independent Variables')

        scaler = load_pickle(scaler_path)
        encoder = load_pickle(encoder_path)
        logging.info('Pickle File loaded')

        return x, y, scaler, encoder
    except Exception as e:
        raise CustomException(e, sys)


def train_and_log_models(x_train_scaled, x_test_scaled, y_train_encoded, y_test_encoded, models_with_parameters):
    try:
        for model_name, model in models_with_parameters.items():
            try:
                logging.info(f'Training Model : {model_name}')
                model.fit(x_train_scaled, y_train_encoded)
                logging.info(f'Fitted with {model_name}')

                ## evaluate model
                logging.info(f'Evaluating {model_name} model')
                y_pred_train = model.predict(x_train_scaled)
                y_pred_test = model.predict(x_test_scaled)

                logging.info('evaluating training metrics')
                accuracy_train, precision_train, f1_train = evaluate_models(y_train_encoded, y_pred_train)

                logging.info('evaluating testing metrics')
                accuracy_test, precision_test, f1_test = evaluate_models(y_test_encoded, y_pred_test)

                try:
                    logging.info('MLFlow Loggers Initializing')
                    with mlflow.start_run(run_name=model_name):  # Set run name to model name
                        # Log model name as a tag
                        mlflow.set_tag("model_name", model_name)
                        
                        # Log model parameters
                        mlflow.log_params(model.get_params())
                        
                        ## logging training metrics
                        logging.info('Logging Training Metrics')
                        mlflow.log_metrics({
                            'training_accuracy': accuracy_train,
                            'training_precision': precision_train,
                            'training_f1': f1_train
                        })
                        
                        ## logging testing metrics
                        logging.info('Logging Testing Metrics')
                        mlflow.log_metrics({
                            'testing_accuracy': accuracy_test,
                            'testing_precision': precision_test,
                            'testing_f1': f1_test
                        })

                        ## infer and log model signature
                        signature = infer_signature(x_train_scaled, model.predict(x_train_scaled))
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path=f"model_{model_name}",  # Unique artifact path
                            signature=signature,
                            input_example=x_train_scaled[:5]
                        )
                        logging.info(f'Model {model_name} logged successfully')
                except Exception as e:
                    logging.error(f'Model {model_name} logging failed')
                    raise CustomException(e, sys)

            except Exception as e:
                raise CustomException(e, sys)
    except Exception as e:
        raise CustomException(e, sys)


def main():
    ## configuration
    mlflow.set_tracking_uri('http://localhost:5000')
    try:
        csv_file_path = os.path.join('Notebook', 'Datasets', 'pre_processed_data.csv')
        scaler_path = os.path.join('Notebook', 'pickle_files', 'scaler.pkl')
        encoder_path = os.path.join('Notebook', 'pickle_files', 'label_encoder.pkl')
        logging.info('Path Traced')
    except Exception as e:
        raise CustomException(e, sys)
    

    models_with_parameters = {
    'Logistic Regression': LogisticRegression(solver='saga', penalty='l2', max_iter=900, C=0.7),
    'Decision Tree Classifier': DecisionTreeClassifier(splitter='best', min_samples_split=2, min_samples_leaf=7, max_features=3, max_depth=3, criterion='gini'),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=300, min_weight_fraction_leaf=0.1, min_samples_split=6, min_samples_leaf=7, max_features='log2', max_depth=3, criterion='log_loss'),
    'Gradient Boosting Classifier': GradientBoostingClassifier(subsample=0.1, n_estimators=900, min_weight_fraction_leaf=0.1, min_samples_split=4, min_samples_leaf=4, max_features='log2', max_depth=5, loss='log_loss', learning_rate=0.1, criterion='squared_error'),
    'Extra Trees Classifier': ExtraTreesClassifier(n_estimators=900, min_weight_fraction_leaf=0.1, min_samples_split=2, min_samples_leaf=4, max_features='sqrt', max_depth=5, criterion='gini'),
    'Support Vector Classifier': SVC(tol=0.003, kernel='sigmoid', gamma='auto', degree=9, coef0=0.1, C=0.1),
    'K Nearest Neighbors Classifier': KNeighborsClassifier(weights='distance', p=2, n_neighbors=4, leaf_size=70, algorithm='brute'),
    'Adaboost Classifier': AdaBoostClassifier(n_estimators=100, learning_rate=0.1, algorithm='SAMME')
}

    try:
        x, y, scaler, encoder = preprocess_data(csv_file_path=csv_file_path, scaler_path=scaler_path, encoder_path=encoder_path)
        logging.info('Data Preprocessed')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        logging.info('Data Split')
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        y_train_encoded = encoder.fit_transform(y_train)
        y_test_encoded = encoder.transform(y_test)
        logging.info('Data Transformed')

        train_and_log_models(x_train_scaled, x_test_scaled, y_train_encoded, y_test_encoded, models_with_parameters)
        logging.info('Training and Logging Started')
    except Exception as e:
        raise CustomException(e, sys)
    

if __name__=='__main__':
    main()


