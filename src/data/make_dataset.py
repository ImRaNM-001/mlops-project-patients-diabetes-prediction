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
    try:
        # Load the parameters
        params: dict[str, object] = load_params(get_root_path() / 'params.yaml') 
        test_size = params['make_dataset']['test_size']
        random_state = params['make_dataset']['random_state']

        # define column names
        column_names: list[str] = params['make_dataset']['column_names'] 

        # Load the data from the .csv file
        df_patient_data: pd.DataFrame = load_data(get_root_path() / 'data/external/pima-indians-diabetes.csv',
                                                column_names)

        # df_patient_data_input: DataFrame = df_patient_data.drop('Outcome', axis = 1)

        train_df: pd.DataFrame
        test_df: pd.DataFrame
        train_df, test_df = train_test_split(df_patient_data, 
                                             test_size = test_size, 
                                             random_state = random_state)
        
        save_data(train_df, test_df, data_path = get_root_path() / 'data')
    
    except Exception as exception:
        logger.error('Failed to complete the data ingestion process: %s', exception)
        print(f'Exception: {exception}')

if __name__ == '__main__':
    main()
