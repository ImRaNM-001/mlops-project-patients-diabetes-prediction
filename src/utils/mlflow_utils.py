import os
import sys
import mlflow as mfl
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))           # appends system root path as string
from src.data.data_loader import get_root_path, load_params

# Localhost execution: 
# from dotenv import load_dotenv
# load_dotenv()         

def setup_mlflow() -> str:
    # Load the dagshub configurations
    dagshub_configs = load_params(get_root_path() / 'config/dagshub_config.yaml')
    
    repo_owner = dagshub_configs['DAGSHUB_REPO_OWNER']
    repo_name = dagshub_configs['DAGSHUB_REPO_NAME']
    
    # Retrieve the token
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    if not dagshub_token:
        raise EnvironmentError('DAGSHUB_TOKEN environment variable is not set')
    
    # Set multiple environment variables with the same value
    for env_var in ['MLFLOW_TRACKING_USERNAME', 'MLFLOW_TRACKING_PASSWORD']:
        os.environ[env_var] = dagshub_token
    
    mfl.set_tracking_uri(f'https://dagshub.com/{repo_owner}/{repo_name}.mlflow')
    mfl.set_experiment(dagshub_configs['EXPERIMENT_NAME'])

    return dagshub_configs['EXPERIMENT_NAME']
