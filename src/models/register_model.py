import logging
import sys
import json
import mlflow as mfl
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))           # appends system root path as string
from src.data.data_loader import get_root_path, log_message
from src.utils.mlflow_utils import setup_mlflow

# calling the logging function
logger: logging.Logger = log_message('register_model', 'register_model.log')

# Authenticating with dagshub and reusing same mlflow experiment
experiment_name: str = setup_mlflow()

def load_model_info(file_path: Path) -> dict[str, str]:
    """
    Load the saved model info from a JSON file
    """
    try:
        with open(file_path, 'r') as file:
            model_info: dict[str, str] = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info

    except FileNotFoundError as error:
        logger.error('Json file not found: %s', error)
        raise

    except Exception as exception:
        logger.error('Unexpected exception occurred while trying to load the model info: %s', exception)
        raise

def register_model(model_name: str, model_info: dict[str, str]):
    """
    Register the model to the MLflow Model Registry
    """
    try:
        model_uri = f'runs:/{
            model_info['run_id']
            }/{
                model_info['model_path']
            }'
        
        # Register the model
        model_version = mfl.register_model(model_uri, model_name)
        
        # Assign the model as 'Staging'
        client = mfl.tracking.MlflowClient()  
        client.set_registered_model_alias(
            name = model_name,
            alias = 'staging',
            version = model_version.version
        )     

        # Add tags to the model versions - Version-specific tag (only applies to this specific version)
        client.set_model_version_tag(
            name = model_name,
            version = model_version.version,
            key = 'created_by',
            value = 'mlops-project-diabetes-prediction_pipeline'
        )

        # Model-level tag (applies to ALL versions)
        client.set_registered_model_tag(
            name = model_name,
            key = 'experiment_team',
            value = 'mlops-projects'
        )
        logger.debug(f'Model: {model_name} with version {model_version.version} is now registered and promoted to Staging')

    except Exception as exception:
        logger.error('Error during model registration: %s', exception)
        raise

def main() -> None:
    try:
        if experiment_name:
            model_info = load_model_info(get_root_path() / 'reports/experiment_info.json')
            register_model('RandomForestClassifier_model', model_info)

    except Exception as exception:
        logger.error('Failed to complete the model registration process: %s', exception)
        raise

if __name__ == '__main__':
    main()

