import os
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent.parent))           # appends system root path as string
from src.data.data_loader import get_root_path, log_message, load_params, load_data

# calling the logging function
logger: logging.Logger = log_message('build_features', 'build_features.log')

def save_data(df: pd.DataFrame, file_path: Path) -> None:
    """
    Save the dataframe to a CSV file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        df.to_csv(file_path, index = False)
        logger.debug('Data saved to %s', file_path)

    except Exception as exception:
        logger.error('Unexpected error occurred while saving the data: %s', exception)
        raise

# Optional: Scale the input (features data - both train & test data) for better model performance
def apply_scaling(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply Standard Scaler to the data
    """
    try:
        scaler = StandardScaler()
        X_train: np.ndarray = scaler.fit_transform(train_data)
        X_test: np.ndarray = scaler.transform(test_data)
        logger.debug('Standard Scaler applied...')

        # Convert the NumPy arrays to pandas DataFrames
        X_train_df: pd.DataFrame = pd.DataFrame(X_train, columns = train_data.columns)
        X_test_df: pd.DataFrame = pd.DataFrame(X_test, columns = test_data.columns)

        return X_train_df, X_test_df
    
    except Exception as exception:
        logger.error('Error during Bag of Words transformation: %s', exception)
        raise

def main() -> None:
    try:
        # Load the parameters
        params: dict[str, object] = load_params(get_root_path() / 'params.yaml') 

        # Replace zero's with "NaN" in columns where zero values do not make sense
        cols_with_nonsenical_zeros: list[str] = params['make_dataset']['column_names'][1:6]

        train_data: pd.DataFrame = load_data(get_root_path() / 'data/interim/train_data.csv')        
        test_data: pd.DataFrame = load_data(get_root_path() / 'data/interim/test_data.csv')        

        train_data[cols_with_nonsenical_zeros] = train_data[cols_with_nonsenical_zeros].replace(0, np.nan)
        test_data[cols_with_nonsenical_zeros] = test_data[cols_with_nonsenical_zeros].replace(0, np.nan)

        # Impute the NaN's with mean of respective column
        train_data.fillna(train_data.mean(), inplace = True)
        test_data.fillna(test_data.mean(), inplace = True)

        # now checking again if any meaningful columns has zero aka missing values or NaN's
        logger.info((train_data == 0).any())
        logger.info((test_data == 0).any())

        train_df_processed, test_df_processed = apply_scaling(train_data, test_data)
       
        save_data(train_df_processed, get_root_path() / 'data/processed/train_df_processed.csv')
        save_data(test_df_processed, get_root_path() / 'data/processed/test_df_processed.csv')

    except Exception as exception:
        logger.error('Failed to complete the feature engineering process: %s', exception)
        raise

if __name__ == '__main__':
    main()

