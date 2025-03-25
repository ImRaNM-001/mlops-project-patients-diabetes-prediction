import logging
import pandas as pd
from data_loader import get_root_path, log_message, load_params, load_data

# calling the logging function
logger: logging.Logger = log_message('preprocess_dataset', 'preprocess_dataset.log')

# display patients with Insulin levels as 0 (i.e, not measured or missing value) - such columns would need Imputation: Replace missing values with a statistical measure such as the mean, median, or mode of the non-missing values.
def preprocess_data(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Preprocess the data
    """
    try:
        df_patient_zero = df[df[column_name] == 0]
        logger.debug('Data preprocessing completed')
        return df_patient_zero.head(10)
    
    except KeyError as error:
        logger.error('Missing column in the dataframe: %s', error)
        raise

    except Exception as exception:
        logger.error('Unexpected error during preprocessing: %s', exception)
        raise

def main() -> None:
    try:
        # Load the parameters
        params: dict[str, object] = load_params(get_root_path() / 'params.yaml')
        column_name = params['process_dataset']['column_name'][0]

        # get data of patients with Insulin levels as 0
        df_patient_zero: pd.DataFrame = load_data(get_root_path() / 'data/interim/train_data.csv')        

        logger.info('Columns with patients with 0 Insulin levels: %s',
                    preprocess_data(df_patient_zero, column_name))                                 
    
    except Exception as exception:
        logger.error('Failed to complete the data preprocessing: %s', exception)
        raise


if __name__ == '__main__':
    main()
