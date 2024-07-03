"""
Author: Livin Albert
Purpose: Support to the feature selection process
"""

from pathlib import Path
from typing import List

import os
import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Function to perform RFE with a given model
def perform_rfe(model, X, y, n_features_to_select):
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    return selected_features

def select_rfe_model(model, X, y, n_features_to_select):
    # Logistic Regression
    if model == "LogisticRegression":
        lr = LogisticRegression(max_iter=1000)
        selected_features = perform_rfe(lr, X, y, n_features_to_select)
    # Random Forest
    elif model == "RandomForestClassifier":
        rf = RandomForestClassifier(n_estimators=100)
        selected_features = perform_rfe(rf, X, y, n_features_to_select)
    # Support Vector Machine
    elif model == "SVC":
        svm = SVC(kernel="linear")
        selected_features = perform_rfe(svm, X, y, n_features_to_select)
    else:
        print("Please select any one of the model from below list:")
        print("['LogisticRegression', 'RandomForestClassifier', 'SVC']")
        print("If cannot find any algorithm above, we are working to add to the list. Thank you")

    print("Selected features using RFE: ",selected_features)
    return selected_features

def hyperparameter_tuning(selected_model, X: pd.DataFrame, y: pd.DataFrame):
    """
    Perform hyperparameter tuning using BayesSearchCV to find the best parameters for modeling

        Parameters:
            X (pd.DataFrame): Independent Features
            y (pd.DataFrame): Dependent Features
            sample_weight (list, optional): _description_. Defaults to None.

        Returns:
            dict:best hyperparameters 
    """
    if selected_model == 'XGBClassifier':
        def xgb_cv(learning_rate, n_estimators, max_depth, subsample, colsample_bytree, min_child_weight, alpha, gamma):
            model = xgb.XGBClassifier(
                learning_rate=learning_rate,
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                gamma=gamma,
                alpha=alpha,
                objective="multi:softprob",  # Set the appropriate objective for multiclass
                num_class=num_classes,
                random_state=42
            )
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            return -np.mean(scores)

        pbounds = {
            "learning_rate": (0.01, 0.3),
            "n_estimators": (50, 300),  # Adjust the range based on your problem and dataset
            "max_depth": (3, 10),
            "subsample": (0.7, 1.0),
            "colsample_bytree": (0.7, 1.0),
            "min_child_weight": (1, 10),
            "alpha": (0, 10),  # Regularization parameter (L1)
            "gamma": (0, 10),  # Minimum loss reduction for tree splitting
            "num_classes": 4
        }

        # Create StratifiedKFold with sample_weights
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        num_classes = len(np.unique(y))

        optimizer = BayesianOptimization(
            f=xgb_cv,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )
        optimizer.maximize(n_iter=10)  # Number of optimization steps

        best_params = optimizer.max["params"]
        print("Best hyperparameters:", best_params)
        return best_params

    elif selected_model == 'LogisticRegression':
        # Define a function to optimize (cross-validation score of logistic regression)
        def optimize_logistic_regression(C):
            clf = LogisticRegression(C=C,random_state=42)
            scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
            return scores.mean()

        # Define the parameter space for optimization
        param_space = {
            'C': (0.01, 10.0),  # Regularization parameter
        }

        # Create the Bayesian optimization object
        optimizer = BayesianOptimization(
            f=optimize_logistic_regression,
            pbounds=param_space,
            random_state=42,
        )

        # Perform optimization
        optimizer.maximize(init_points=5, n_iter=10)

        # Get the best C value from the optimization
        best_C = optimizer.max['params']['C']
        print("Best hyperparameters:", best_C)
        return best_C
