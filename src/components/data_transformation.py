import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass 
class DataTransformationConfig:
    preprocessor_ob_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            logging.info("Creating Data Transformation Pipelines...")

            # Define numerical features
            numerical_features = [
                'industrial_risk', 'management_risk', 'financial_flexibility',
                'credibility', 'competitiveness', 'operating_risk'
            ]

            # Numerical pipeline
            num_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])

            # Column transformer (only for input features)
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features)
            ])

            logging.info("Data Transformation Pipelines Created Successfully.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # def initiate_data_transformation(self, train_path, test_path):
    #     try:
    #         logging.info(f"Reading training data from: {train_path}")
    #         train_df = pd.read_csv(train_path)

    #         logging.info(f"Reading testing data from: {test_path}")
    #         test_df = pd.read_csv(test_path)

    #         # Define target column and numerical columns
    #         target_column_name = 'class'
    #         numerical_columns = [
    #             'industrial_risk', 'management_risk', 'financial_flexibility',
    #             'credibility', 'competitiveness', 'operating_risk'
    #         ]

    #         if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
    #             raise CustomException(f"Target column '{target_column_name}' not found in the dataset", sys)

    #         logging.info("Splitting input features and target variable...")

    #         input_feature_train_df = train_df[numerical_columns]
    #         target_feature_train_df = train_df[target_column_name]

    #         input_feature_test_df = test_df[numerical_columns]
    #         target_feature_test_df = test_df[target_column_name]

    #         logging.info("Applying preprocessing transformations...")

    #         # Get preprocessor
    #         preprocessor_obj = self.get_data_transformer_obj()

    #         input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
    #         input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

    #         # Convert target column to NumPy array
    #         target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
    #         target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

    #         # Combine features and target variable
    #         train_arr = np.hstack((input_feature_train_arr, target_feature_train_arr))
    #         test_arr = np.hstack((input_feature_test_arr, target_feature_test_arr))

    #         # Save preprocessor object
    #         save_object(
    #             file_path=self.data_transformation_config.preprocessor_ob_file_path,
    #             obj=preprocessor_obj
    #         )

    #         logging.info("Data Transformation Completed Successfully.")
    #         return train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file_path

    #     except Exception as e:
    #         raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info(f"Reading training data from: {train_path}")
            train_df = pd.read_csv(train_path)

            logging.info(f"Reading testing data from: {test_path}")
            test_df = pd.read_csv(test_path)

            # Define target column and numerical columns
            target_column_name = 'class'
            numerical_columns = [
                'industrial_risk', 'management_risk', 'financial_flexibility',
                'credibility', 'competitiveness', 'operating_risk'
            ]

            if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
                raise CustomException(f"Target column '{target_column_name}' not found in the dataset", sys)

            logging.info("Splitting input features and target variable...")

            input_feature_train_df = train_df[numerical_columns]
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df[numerical_columns]
            target_feature_test_df = test_df[target_column_name]

            # Encode the target variable from string to numeric (0, 1)
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)

            logging.info("Applying preprocessing transformations...")

            # Get preprocessor
            preprocessor_obj = self.get_data_transformer_obj()

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Convert target column to NumPy array
            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

            # Combine features and target variable
            train_arr = np.hstack((input_feature_train_arr, target_feature_train_arr))
            test_arr = np.hstack((input_feature_test_arr, target_feature_test_arr))

            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessor_obj
            )

            logging.info("Data Transformation Completed Successfully.")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file_path

        except Exception as e:
            raise CustomException(e, sys)