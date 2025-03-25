import os
import logging
import pandas as pd
from data_loader import get_root_path, log_message, load_params, load_data
from pathlib import Path
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split

# calling the logging function
logger: logging.Logger = log_message('make_dataset', 'make_dataset.log')

# save the train & test data to target folder
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: Path) -> None:
    """
    Save the train and test datasets
    """
    try:
        target_data_path: str = os.path.join(data_path, 'interim')
        os.makedirs(target_data_path, exist_ok = True)
        train_data.to_csv(os.path.join(target_data_path, 'train_data.csv'), index = False)
        test_data.to_csv(os.path.join(target_data_path, 'test_data.csv'), index = False)
        logger.debug('Train and Test data saved to: %s', target_data_path)

    except Exception as exception:
        logger.error('Unexpected error occurred while saving the data: %s', exception)
        raise


def main() -> None:
    # Load the parameters
    params: dict[str, object] = load_params(get_root_path() / 'params.yaml') 
    url = params['make_dataset']['data_source_url']

    # Check if data exists in folder if not download the data from an external source 
    # and save to a .csv file
    file_path: Path = get_root_path() / 'data/external/pima-indians-diabetes.csv'
    
    if file_path.exists():
        logger.info(f'CSV file already exists at {file_path}, hence skipped download')
    else:
        # Create the directory if it doesn't exist
        os.makedirs(file_path.parent, exist_ok = True)
        logger.info(f'Downloading data from {url}..')
        try:
            df_patient_data = pd.read_csv(url, header = None)       # the data file do not have a header
            df_patient_data.to_csv(file_path, 
                                    index = False,
                                    header = False)                  # enforcing pandas not to write 1st row as headers
        
        except Exception as exception:
            logger.error(f'Failed to download data: {exception}')
            raise

    try:
        # define column names
        column_names: list[str] = params['make_dataset']['column_names'] 
        test_size = params['make_dataset']['test_size']
        random_state = params['make_dataset']['random_state']

        # Load the data from the .csv file
        df_patient_data: pd.DataFrame = load_data(file_path, column_names)

        # df_patient_data_input: DataFrame = df_patient_data.drop('Outcome', axis = 1)

        train_df: pd.DataFrame
        test_df: pd.DataFrame
        train_df, test_df = train_test_split(df_patient_data, 
                                            test_size = test_size, 
                                            random_state = random_state)
        
        save_data(train_df, test_df, data_path = get_root_path() / 'data')
    
    except Exception as exception:
        logger.error('Failed to complete the data ingestion process: %s', exception)
        raise

if __name__ == '__main__':
    main()
