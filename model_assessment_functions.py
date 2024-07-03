"""
Author: Livin Albert
Purpose: Support to the model development
"""

import json
from typing import Optional

from pathlib import Path
import pandas as pd
import numpy as np
import shap

from numpyencoder import NumpyEncoder
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
# from bayes_opt import BayesianOptimization
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

def corresponding_data(data: pd.DataFrame, reference_data: pd.DataFrame):
    """
    Get corresponding data for indexes from reference dataset

        Parameters:
            data (pd.DataFrame): _description_
            reference_data (pd.DataFrame): _description_

        Returns:
            _type_: _description_
    """

    return data.loc[reference_data.index.to_list()].to_numpy()

def preprocess_data(raw_data: pd.DataFrame, features_list: list, target_column: str):
    """
    Input Parameter
        raw_data - input dataframe, it contains all period data
        drop_features_list - list of feature required for the model
        target_column - Target variable column name

    Output:
        X_train, X_test,- independent dataframe
        y_train, y_test,  - target variable array
        feature_list - Number of features passing to the model

    """
    X = raw_data[features_list]
    y = raw_data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature List
    feature_list = list(X_train.columns)
    print("Number of features : ", len(feature_list))

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_list,
    )


def conf_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Prints confusion matrix

    Parameters:
        y_test (np.ndarray): _description_
        y_pred (np.ndarray): _description_
    """

    # Creating a confusion matrix
    con_mat = confusion_matrix(y_test, y_pred)
    con_mat = pd.DataFrame(con_mat, range(2), range(2))

    # Ploting the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.5)
    sns.heatmap(
        con_mat, annot=True, annot_kws={"size": 8}, fmt="g", cmap="Blues", cbar=False
    )
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")

    return None

def get_ytrue_and_yscore(y_train: np.ndarray, y_scores: np.ndarray):

    y_probability = list(map(lambda x: x[1], y_scores))
    y_true= list(np.where(y_train == 1, 1, 0))
    y_outcomes = pd.DataFrame(
        {"y_probability": y_probability, "y_true": y_true}
    )

    return y_outcomes

def prt_values(y_train: np.ndarray, y_scores: np.ndarray):
    """
    Get precision, recall and dollar recall values for various thresholds

        Parameters:
            y_train: Actual target value
            y_scores_train: Probability values
            
        Returns:
            precision: List of precision values
            recall: List of recall values
            thresholds: List of thresholds corresponding to above values
            roc_auc: ROC value for the model
    """
    y_outcomes = get_ytrue_and_yscore(y_train, y_scores)
    
    precision, recall, thresholds = precision_recall_curve(
        y_outcomes["y_true"], y_outcomes["y_probability"]
    )

    # Append final threshold value corresponding to max threshold
    thresholds = np.append(thresholds, 1)

    # Calculating ROC/AUC value
    roc_auc = roc_auc_score(y_outcomes["y_true"], y_outcomes["y_probability"])

    # Create dataframe from outputs
    df_accuracy_metrics = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
            "roc_auc": roc_auc,
        }
    )

    return df_accuracy_metrics


def model_tune_category(df_metrics: pd.DataFrame, threshold=None,data_set_type=None):
    
    if data_set_type == "train":

        # Check churn and cont threshold at dollar_recall using dynamic value
        df_filter = (
            df_metrics[df_metrics["recall"] >= threshold]
            .sort_values(by=["precision"], ascending=False)
            .reset_index(drop=True)
            .head(1)
        )
    elif data_set_type == "test":

        # get the accuracy value using train thresholds
        df_filter = (
            df_metrics[df_metrics["thresholds"].round(3) <= threshold]
            .sort_values(by=["precision"], ascending=False)
            .reset_index(drop=True)
            .head(1)
        )

    # Combining all the metrics
    df_acc_combined = df_filter

    return df_acc_combined



def feature_importance(pipeline, selected_model):
    """
    pipeline: train data model file
    X_train: Getting column name from X_train

    df: Dataframe of feature importance value
    """
    if selected_model == 'xgboost':

        # Calculating feature importance:
        feature_importance = pipeline.named_steps['xgb'].feature_importances_
        feature_name = pipeline.named_steps['xgb'].get_booster().feature_names

        # Sort the feature importances with attributes value
        sorted_indices = np.argsort(feature_importance)[::-1]
        x = []
        for idx in sorted_indices:
            x.append([feature_name[idx], feature_importance[idx]])

        df = pd.DataFrame(x, columns=["Feature_name", "value"])
        return df
    
    elif selected_model == 'LogisticRegression':
        feature_names = list(pipeline.steps[0][1].get_feature_names_out())
        # feature_names = [each.partition('__')[2] for each in feature_names]

        # Get the coefficient values in steps method
        coef_values = pd.DataFrame(pipeline.named_steps['classifier'].coef_.transpose(),columns = ['Expansion'])

        # Create a DataFrame with the coefficients and column names 
        coef_columns= pd.DataFrame(list(feature_names)).copy() 
        coef_table = pd.concat([coef_columns,coef_values],axis =1)
        coef_table = coef_table.rename(columns = { 0: 'Feature_name'})
        
        # Create 'Flag' column based on the values in 'Expansion'
        coef_table['Flag'] = np.where(coef_table['Expansion'] < 0, 'N', 'P') 
        
        return coef_table
    