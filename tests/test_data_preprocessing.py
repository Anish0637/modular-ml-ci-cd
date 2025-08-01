
# tests/test_data_preprocessing.py
import pytest
import pandas as pd
from src.data_preprocessing import load_data, preprocess_data, split_data

def test_load_data():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_preprocess_data():
    df = load_data()
    X, y = preprocess_data(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert 'target' not in X.columns

def test_split_data():
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) + len(X_test) == len(X)
