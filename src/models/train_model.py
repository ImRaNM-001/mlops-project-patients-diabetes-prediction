import os
import logging
import sys
import joblib as jb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).parent.parent.parent))           # appends system root path as string
from src.data.data_loader import get_root_path, log_message, load_params, load_data

# calling the logging function
logger: logging.Logger = log_message('train_model', 'train_model.log')

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict[str, int]) -> RandomForestClassifier:
    """
    Train the RandomForest classifier model.
    
    :param X_train: Training feature sets
    :param y_train: Training target variables
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier model
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be same")
        
        logger.debug('Initializing RandomForest model with parameters: %s', params)
        rf_model: RandomForestClassifier = RandomForestClassifier(**params)
        
        logger.debug('Model training started with %d samples', X_train.shape[0])
        rf_model.fit(X_train, y_train)
        logger.debug('Model training completed')
        
        return rf_model

    except ValueError as error:
        logger.error('ValueError during model training: %s', error)
        raise

    except Exception as exception:
        logger.error('Error during model training: %s', exception)
        raise

def save_model(model, file_path: Path) -> None:
    """
    Save the trained model as a joblib file
    :param model: Trained model object
    :param file_path: Path where the model file is saved
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        with open(file_path, 'wb') as file:
            jb.dump(value = model,
                    filename = file)
        logger.debug('Model saved to %s', file_path)

    except FileNotFoundError as error:
        logger.error('File path not found: %s', error)
        raise

    except Exception as exception:
        logger.error('Error occurred while saving the model: %s', exception)
        raise

def main() -> None:
    try:
        # Load the parameters
        params: dict[str, str | int] = load_params(get_root_path() / 'params.yaml') 

        # set the model training parameters 
        best_params: dict[str, int] = {
                            'n_estimators': params['train_model']['n_estimators'],
                            'max_features': params['train_model']['max_features'],
                            'max_depth': params['train_model']['max_depth'],
                            'max_samples': params['train_model']['max_samples'],
                            'verbose': params['train_model']['verbose'],
                            'random_state': params['train_model']['random_state'],
                            'n_jobs': params['train_model']['n_jobs'],
                            'min_samples_split': params['train_model']['min_samples_split'],
                            'min_samples_leaf': params['train_model']['min_samples_leaf'],
                        }
     
        train_data: pd.Dataframe = load_data(get_root_path() / 'data/processed/train_df_processed.csv')    
        
        X_train: np.ndarray = train_data.iloc[:, :-1].values        

        # Binning 'Outcome' column from "y_train" into two categories: 0 and 1
        y_train: pd.arrays.Categorical = pd.cut(train_data.iloc[:, -1].values, 
                         bins = [-float('inf'), 0.5, float('inf')], 
                                 labels = [0, 1], 
                                 right = False)

        rf_model: RandomForestClassifier = train_model(X_train, y_train, best_params)
        save_model(model = rf_model,
                   file_path = get_root_path() / 'models/rfclf_model.joblib')

    except Exception as exception:
        logger.error('Failed to train the classifier model: %s', exception)
        print(f'Exception: {exception}')

if __name__ == '__main__':
    main()



"""
    Fix for ERROR - ValueError during model training:                   <Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.>
    train_model - ERROR - Failed to train the classifier model:         <Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.>
    Exception:                                                          <Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.>

    (context): For converting a continuous target variable (like "y_train") into a binary classification problem.
    Must transform the "y_train" variable by binning the last column of the train_data DataFrame into two categories using the "pd.cut" function. Here's a detailed breakdown:

    1. Extracting the Target Variable / Extract Features and Target:

        X_train contains all columns except the last one from train_data. These are the features.
        train_data.iloc[:, :-1].values selects all rows and all columns except the last one.

        train_data.iloc[:, -1].values selects all rows of the last column. This is the target variable.
        train_data.iloc[:, -1].values extracts the values of the last column of the train_data DataFrame, which is assumed to be the target variable.


    2. Binning the Target Variable:

        pd.cut is used to bin the continuous values of the target variable into two categories. In other simple words, pd.cut is used to categorize the target variable into two bins
            --> bins = [-float('inf'), 0.5, float('inf')] defines the bin edges. Values less than or equal to 0.5 fall into the first bin, and values greater than 0.5 fall into the second bin.
            --> i.e, bins = [-float('inf'), 0.5, float('inf')] creates two bins: one for values less than or equal to 0.5, and one for values greater than 0.5.
    
        labels = [0, 1] assigns labels to the bins. The first bin is labeled as 0, and the second bin is labeled as 1.
        i.e, labels = [0, 1] assigns the label 0 to the first bin and 1 to the second bin.

    
        right = False specifies that the bins are left-inclusive (i.e., the left edge is included in the bin).
        i.e, right = False means the bins are left-inclusive (the left edge is included in the bin).


    3. Result:

        y_train is now a categorical array with values 0 or 1, representing the binned categories of the original target variable.

"""