
# tests/test_model_training.py
import pytest
import os
from src.model_training import train_model, save_model, load_model
from src.data_preprocessing import load_data, preprocess_data, split_data

@pytest.fixture
def trained_model_and_data():
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    return model, X_test, y_test

def test_train_model(trained_model_and_data):
    model, _, _ = trained_model_and_data
    assert hasattr(model, 'predict')

def test_save_and_load_model(trained_model_and_data, tmp_path):
    model, _, _ = trained_model_and_data
    model_path = tmp_path / "test_model.pkl"
    save_model(model, str(model_path))
    assert os.path.exists(model_path)
    
    loaded_model = load_model(str(model_path))
    assert isinstance(loaded_model, type(model))
