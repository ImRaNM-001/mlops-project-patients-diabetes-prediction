import os
import logging
import sys
import json
import joblib as jb
import numpy as np
import pandas as pd
import mlflow as mfl
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).parent.parent.parent))           # appends system root path as string
from src.data.data_loader import get_root_path, log_message, load_data

# calling the logging function
logger: logging.Logger = log_message('evaluate_model', 'evaluate_model.log')

def load_model(file_path: Path) -> RandomForestClassifier:
    """
    Load the trained model from the saved location
    """
    try:
        with open(file_path, 'rb') as file:
            model: RandomForestClassifier = jb.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model

    except FileNotFoundError as error:
        logger.error('Model file not found: %s', error)
        raise

    except Exception as exception:
        logger.error('Unexpected exception occurred while trying to load the model: %s', exception)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float | np.ndarray]:
    """
    Make predictions on the model and return the evaluation metrics
    """
    try:
        y_pred = model.predict(X_test)

        accuracy: float = accuracy_score(y_test, y_pred)
        precision: float = precision_score(y_test, y_pred)
        conf_matrix: np.ndarray = confusion_matrix(y_test, y_pred)

        metrics_dict: dict[str, float | np.ndarray] = {
            'accuracy': accuracy,
            'precision': precision,
            'confusion_matrix': conf_matrix,
        }

        logger.debug('Model evaluation metrics calculated')

        # Convert to Serializable metrics dictonary of lists from NumPy arrays
        """
            while dealing with nested dictionaries or lists, all NumPy arrays must be converted to lists for JSON serialization to avoid unexpected exception 
        """
        metrics_dict = {key: value.tolist() if isinstance(value, np.ndarray) 
                        else value for key, value in metrics_dict.items()}
        
        return metrics_dict
    
    except Exception as error:
        logger.error('Error during model evaluation: %s', error)
        raise

def save_metrics(metrics: dict[str, float | np.ndarray], file_path: Path) -> None:
    """
    Save the evaluation metrics to a JSON file
    """
    try:
        # Check if the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok = True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent = 2)
        logger.debug('Metrics saved to %s', file_path)

    except Exception as error:
        logger.error('Error occurred while saving the metrics: %s', error)
        raise

def main() -> None:
    try:
        # Load the model to perform predictions
        model = load_model(get_root_path() / 'models/rfclf_model.joblib')   
     
        test_data = load_data(get_root_path() / 'data/processed/test_df_processed.csv')   
        X_test: np.ndarray = test_data.iloc[:, :-1].values

        # Binning 'Outcome' column from "y_test" into two categories: 0 and 1
        y_test: pd.arrays.Categorical = pd.cut(test_data.iloc[:, -1].values, 
                         bins = [-float('inf'), 0.5, float('inf')], 
                                 labels = [0, 1], 
                                 right = False)

        metrics = evaluate_model(model, X_test, y_test)
        with mfl.start_run(run_name = 'best_model') as best_model:       
            mfl.log_metric(metrics)

        save_metrics(metrics, get_root_path() / 'reports/metrics.json')

    except Exception as exception:
        logger.error('Model evaluation failed due to: %s', exception)
        print(f'Exception: {exception}')

if __name__ == '__main__':
    main()

