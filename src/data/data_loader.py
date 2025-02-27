import os
import logging
import pandas as pd
from pathlib import Path
from yaml import safe_load, YAMLError

def get_root_path(current_file_path: Path = Path(__file__)) -> Path:
  """
  Returns the root path by going up 3 levels from the current file's directory
  """
  return current_file_path.parent.parent.parent 

def log_message(logger_name: str, log_file_name: str) -> logging.Logger:

    # Check if the "logs" directory exists
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok = True)

    # logging configuration
    logger: logging.Logger = logging.getLogger(logger_name)
    logger.setLevel('DEBUG')

    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    log_file_path: str = f'{get_root_path()}/{log_dir}/{log_file_name}'

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel('DEBUG')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

# calling the logging function
logger: logging.Logger = log_message('Loading configurations and data', 'data_config.log')

def load_params(params_path: Path) -> dict[str, object]:
    """
    Load parameters from an YAML file
    """
    try:
        with open(params_path, 'r') as file:
            params: dict[str, object] = safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise

    except YAMLError as error:
        logger.error('YAML error: %s', error)
        raise

    except Exception as exception:
        logger.error('Unexpected error: %s', exception)
        raise

def load_data(file_path: Path, column_names: list[str] = None) -> pd.DataFrame:
    """
    Load data from a CSV file
    """
    try:
        if column_names is None:
            df: pd.DataFrame = pd.read_csv(file_path)
        else:
            df: pd.DataFrame = pd.read_csv(file_path,
                                    header = None,
                                    names = column_names)
        logger.debug('Data loaded from: %s', file_path)
        return df
    
    except pd.errors.ParserError as error:
        logger.error('Failed to parse the CSV file: %s', error)
        raise
    
    except Exception as exception:
        logger.error('Unexpected exception occurred while loading the data: %s', exception)
        raise

