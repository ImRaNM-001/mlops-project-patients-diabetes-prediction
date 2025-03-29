import os
import logging
import sys
import json
import joblib as jb
import numpy as np
import pandas as pd
import mlflow as mfl
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).parent.parent.parent))           # appends system root path as string
from src.data.data_loader import get_root_path, log_message, load_data
from src.utils.mlflow_utils import setup_mlflow

# calling the logging function
logger: logging.Logger = log_message('evaluate_model', 'evaluate_model.log')

# Authenticating with dagshub and reusing same mlflow experiment
experiment_name: str = setup_mlflow()

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

def save_model_info(run_id: str, model_path: str, file_path: Path) -> None:
    """
    Save the model run ID and path to a JSON file
    """
    try:
        model_info = {
            'run_id': run_id, 
            'model_path': model_path
        }
        
        with open(file_path, 'w') as json_file:
            json.dump(model_info, json_file, indent = 2)
        logger.debug('Model info saved to %s', file_path)

    except Exception as exception:
        logger.error('Error occurred while saving the model info: %s', exception)
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

        train_data: pd.Dataframe = load_data(get_root_path() / 'data/processed/train_df_processed.csv')            
        X_train: np.ndarray = train_data.iloc[:, :-1].values          

        # get the model signature
        signature = mfl.models.infer_signature(model_input = X_train,
                                              model_output = model.predict(X_test))

        # fetching the run id for existing run_name = 'best_model'
        best_model_run_id_path = get_root_path() / 'src/models/best_model_run_id.txt'
        best_model_run_id = None
        
        if best_model_run_id_path.exists():
            with open(best_model_run_id_path, 'r') as txt_file:
                best_model_run_id = txt_file.read().strip()

        if best_model_run_id and experiment_name:
            with mfl.start_run(run_id = best_model_run_id): 
                logger.info(f'Loaded run ID as {best_model_run_id}')
                mfl.log_metric('accuracy_score', metrics['accuracy'])
                mfl.log_metric('precision_score', metrics['precision'])

                with open('confusion_matrix.joblib', 'wb') as conf_matrix_file:
                    jb.dump(metrics['confusion_matrix'], conf_matrix_file)

                # Log the confusion matrix as an artifact
                mfl.log_artifact('confusion_matrix.joblib')         

                # Log the model with the signature
                mlflow.sklearn.log_model(sk_model = model,
                          artifact_path = 'final_model',
                          signature = signature)

        save_metrics(metrics, get_root_path() / 'reports/metrics.json')
        save_model_info(best_model_run_id, 
                        'final_model', 
                        get_root_path() / 'reports/experiment_info.json')

    except Exception as exception:
        logger.error('Model evaluation failed due to: %s', exception)
        raise

    finally:
        try:
            os.remove(best_model_run_id_path)
            logger.info(f'Deleted run Id file at {best_model_run_id_path}')

        except Exception as exception:
            logger.warning(f'Failed to delete run ID file: {exception}')

if __name__ == '__main__':
    main()
