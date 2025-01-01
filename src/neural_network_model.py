import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def preprocess_data(data):
    """
    Preprocess the dataset for neural network training.

    Parameters:
        data (pd.DataFrame): Input data with features and labels.

    Returns:
        tuple: Processed feature array (X) and encoded labels (y).
    """
    # Extract features and labels
    X = data[["Teff", "logg", "Lum", "FliPer"]].values
    y = LabelEncoder().fit_transform(data["Star_Type"])

    # Apply feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def build_neural_network(input_dim, output_shape):
    """
    Build and compile a simple neural network.

    Parameters:
        input_dim (int): Number of input features.

    Returns:
        keras.Sequential: Compiled neural network model.
    """
    model = Sequential([
        Dense(128, activation="relu", input_dim=input_dim),
        Dense(128, activation="relu"),
        Dense(output_shape, activation="softmax")  # 10 output classes
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def train_and_evaluate_nn(X_train, y_train, X_test, y_test):
    """
    Train and evaluate the neural network model.

    Parameters:
        X_train (array): Training feature array.
        y_train (array): Training labels.
        X_test (array): Testing feature array.
        y_test (array): Testing labels.

    Returns:
        keras.Sequential: Trained model.
    """
    model = build_neural_network(X_train.shape[1], y_train.shape[1])
    model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0.2, verbose=1)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return model
