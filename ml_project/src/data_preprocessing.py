
# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def load_data():
    iris = load_iris(as_frame=True)
    return iris.frame

def preprocess_data(df: pd.DataFrame, target_column: str = 'target'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
