import os

def create_file(path, content=""):
    """Helper function to create a file with given content."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"Created file: {path}")

def create_directory(path):
    """Helper function to create a directory."""
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")

def main():
    project_name = "ml_project"

    # Check if the directory already exists to prevent accidental overwrites
    if os.path.exists(project_name):
        response = input(f"The directory '{project_name}' already exists. Do you want to continue and potentially overwrite files? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return

    # --- Create Main Directories ---
    create_directory(f"{project_name}/.github/workflows")
    create_directory(f"{project_name}/data/raw")
    create_directory(f"{project_name}/data/processed")
    create_directory(f"{project_name}/notebooks")
    create_directory(f"{project_name}/src")
    create_directory(f"{project_name}/models")
    create_directory(f"{project_name}/tests")


# --- Create Core Files ---
    create_file(f"{project_name}/.gitignore", """
# Virtual environment
venv/
.venv/

# Byte-compiled files
__pycache__/
*.pyc

# IDE files
.vscode/

# Local data files
data/processed/
models/

# Logs and generated data
*.log
    """)

    create_file(f"{project_name}/README.md", f"# {project_name}\n\nThis is a modular machine learning project repository with a CI/CD pipeline using GitHub Actions.")

    create_file(f"{project_name}/requirements.txt", """
pandas
scikit-learn
pytest
flake8
joblib
""")

    # --- Create Python Source Files ---
    create_file(f"{project_name}/src/__init__.py")
    create_file(f"{project_name}/src/data_preprocessing.py", """
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
""")

    create_file(f"{project_name}/src/model_training.py", """
# src/model_training.py
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

def load_model(model_path):
    return joblib.load(model_path)
""")

    create_file(f"{project_name}/src/model_evaluation.py", """
# src/model_evaluation.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }
    return metrics
""")

    create_file(f"{project_name}/src/prediction.py", """
# src/prediction.py
def make_predictions(model, new_data):
    return model.predict(new_data)
""")

    create_file(f"{project_name}/main.py", """
# main.py
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model_training import train_model, save_model
from src.model_evaluation import evaluate_model, print_metrics
import pandas as pd

def main():
    print("--- Starting ML Pipeline ---")
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_model(X_train, y_train)
    save_model(model, "models/iris_classifier.pkl")
    
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\\n--- Model Evaluation Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()
""")

    # --- Create Test Files ---
    create_file(f"{project_name}/tests/__init__.py")
    create_file(f"{project_name}/tests/test_data_preprocessing.py", """
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
""")

    create_file(f"{project_name}/tests/test_model_training.py", """
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
""")

    # --- Create GitHub Actions Workflow File ---
    create_file(f"{project_name}/.github/workflows/ci_cd_pipeline.yml", """
name: Python ML CI/CD Pipeline

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  build_and_test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Linting with Flake8
        run: |
          pip install flake8
          flake8 src/ tests/ main.py --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 src/ tests/ main.py --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

      - name: Run Pytest Unit Tests
        run: |
          pytest tests/

  train_and_evaluate:
    needs: build_and_test
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run ML Training and Evaluation
        run: |
          python main.py

      - name: Upload Trained Model (Artifact)
        uses: actions/upload-artifact@v4
        with:
          name: trained-model-${{ github.sha }}
          path: models/iris_classifier.pkl
          retention-days: 5
""")

    print(f"\nML project structure '{project_name}' created successfully!")
    print("Next steps:")
    print(f"1. Navigate into the folder: cd {project_name}")
    print(f"2. Initialize a git repository: git init")
    print(f"3. Install dependencies in a virtual environment: pip install -r requirements.txt")
    print("4. Add and commit all files: git add . && git commit -m 'Initial project setup'")
    print("5. Push to GitHub to trigger your first CI/CD pipeline!")

if __name__ == "__main__":
    main()






