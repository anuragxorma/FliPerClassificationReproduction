import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

def load_data(file_path):
    """
    Load the processed dataset.

    Parameters:
        file_path (str): Path to the dataset.

    Returns:
        DataFrame: Processed dataset.
    """
    return pd.read_csv(file_path)

def split_data(data, target_column="Star_Type", test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
        data (DataFrame): Input dataset.
        target_column (str): Column name for target labels.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed.

    Returns:
        tuple: X_train, X_test, y_train, y_test.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest model.

    Parameters:
        X_train (array): Training features.
        y_train (array): Training labels.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed.

    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.

    Parameters:
        model (RandomForestClassifier): Trained model.
        X_test (array): Testing features.
        y_test (array): Testing labels.

    Returns:
        dict: Evaluation metrics (accuracy, precision, recall, F1-score).
    """
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1-Score": f1_score(y_test, y_pred, average="weighted"),
        "Classification Report": classification_report(y_test, y_pred)
    }
    return metrics

def save_model(model, file_path):
    """
    Save the trained model.

    Parameters:
        model (RandomForestClassifier): Trained model.
        file_path (str): Path to save the model.

    Returns:
        None
    """
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")
