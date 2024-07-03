"""
Author: Livin Albert
Purpose: Define the static file names
Action: Update the file name based on requirement
"""
import pandas as pd
import os
from pathlib import Path 

ROOT = Path(os.getcwd()).parents[0]
INPUT_DATA = ROOT / "data/input_data/"
INTERIM_DATA = ROOT / "data/interim_data/"
MODELS = ROOT / "models"
MODEL_ID = "JDS01"
PROD_MODEL = MODELS / "light_gbm.joblib"
OUTPUT_DATA = ROOT / "output_data"

class LocationConfig:
    """

    """

    def __init__(self):

        # Input Files (SQL/CSV FILES ARE HERE LIKE '01_sql1.sql'/'01_sql1.csv')
        self.input_files = [
            'customer.csv'
        ]
        
        self.feature_consolidation_output_table = INTERIM_DATA / "train.csv"
        self.feature_transformation_output_table = INTERIM_DATA / "train.csv"
        self.transformed_features_oot_table_name = INTERIM_DATA / "test.csv"
        self.null_features_agg_value = INTERIM_DATA / "null_features_agg_value.csv"
        self.selected_features_table_name = INTERIM_DATA / "selected_features_table_name.csv"    
        
        # Output files
        self.feature_directionality = OUTPUT_DATA / "feature_directionality.csv"
        self.feature_importance = OUTPUT_DATA / "feature_importance.csv"
        self.model_accuracy_tracker = OUTPUT_DATA / "model_accuracy_tracker.csv"
        self.feature_coefficient = OUTPUT_DATA / "feature_coefficient.csv"
        self.feature_contribution = OUTPUT_DATA / "feature_contribution.csv"


class ModelConfig:

    def __init__(self,selected_model):
        self.value = selected_model

        # Model Joblib file
        self.joblib_file = MODELS/ f'{MODEL_ID}_{self.value}_model.joblib'



    

    
