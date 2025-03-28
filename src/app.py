import os
import sys
import logging
import werkzeug
import numpy as np
import pandas as pd
import mlflow as mfl
import joblib as jb
from pathlib import Path
from flask import Flask, request, render_template, jsonify

sys.path.append(str(Path(__file__).parent.parent))           # appends system root path as string      
from src.data.data_loader import get_root_path, log_message, load_params
from src.utils.mlflow_utils import setup_mlflow

# calling the logging function
logger: logging.Logger = log_message('flask_app', 'app.log')

# Authenticating with dagshub and reusing same mlflow experiment
setup_mlflow()

# Initialize the Flask app
app: Flask = Flask(__name__)

# Load the trained model
def load_model(model_name):
    try:
        # First trying to load model from MLflow model registry
        client = mfl.tracking.MlflowClient()       
        
        # Get the version with the "staging" alias
        try:
            model_version = client.get_model_version_by_alias(model_name, 'staging')
            model_uri = f'models:/{model_name}/{model_version.version}'         
            logger.info(f'Loading model from MLflow: {model_uri}')
            return mfl.pyfunc.load_model(model_uri)
        
        except Exception as mlflow_exception:
            # Log the MLflow error but not raised yet - try local file next
            logger.warning('Could not load model from MLflow model registry: %s', mlflow_exception)

            # If model registry loading fails, trying to load model from the local directory as fallback
            model_path = get_root_path() / 'models/rfclf_model.joblib'
            if model_path.exists():
                return jb.load(model_path)
            else:
                logger.error(f'Error loading model locally: {exception}')
                raise FileNotFoundError(f'Model could not be loaded from Mlflow registry {model_name} '
                                        f'and model not found at local path'
                )
            
    except FileNotFoundError:
        raise

    except Exception as exception:
        logger.error('Unexpected error while loading model: %s', exception)
        raise

 # Load the parameters and the related input feature names
params: dict[str, object] = load_params(get_root_path() / 'params.yaml') 
FEATURE_NAMES: list[str] = params['make_dataset']['column_names'][:-1] 

# Define the application routes
@app.route('/')
def home():
    return render_template('index.html', feature_names = FEATURE_NAMES)

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        # If JSON data is sent
        if request.is_json:
            data = request.get_json()
            input_features = [float(data.get(feature, 0)) for feature in FEATURE_NAMES]
        # If form data is sent
        else:
            # Get features from form data
            try:
                input_features = [float(request.form.get(feature, 0)) for feature in FEATURE_NAMES]

            except ValueError:
                # Invalid input data
                return jsonify({'error': 'Invalid input values. Numbers expected..'}), 400
        
        # Create a DataFrame with the input features
        input_df = pd.DataFrame([input_features], columns = FEATURE_NAMES)
        
        # Load the model and make prediction
        model = load_model('RandomForestClassifier_model')            # 'RandomForestClassifier_model' is our registered model name
        prediction = model.predict(input_df)[0]
        
        # Interpret the prediction
        result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'
        probability = None
        
        # If the model can provide probabilities
        if hasattr(model, 'predict_proba'):
            try:
                # Get the raw probability value
                proba_result = model.predict_proba(input_df)
                
                # Handle different return formats (for test compatibility)
                if isinstance(proba_result, list) or isinstance(proba_result, np.ndarray):
                    if len(proba_result) > 0 and isinstance(proba_result[0], (list, np.ndarray)) and len(proba_result[0]) > 1:
                        raw_probability = proba_result[0][1]        # Standard structure [[0.2, 0.8]]
                    else:
                        raw_probability = float(proba_result[0])
                else:
                    raw_probability = float(proba_result)
                
                # Use round() function instead of .round() method for compatibility
                probability = round(float(raw_probability), 4) * 100
                
            except Exception as exception:
                logger.warning(f'Error calculating probability: {str(exception)}')
                probability = None
            
        response = {
            'prediction': int(prediction),
            'result': result
        }
        
        if probability is not None:
            response['probability'] = float(probability)
        
        # Return JSON if the request was JSON, otherwise render result template
        if request.is_json:
            return jsonify(response)
        else:
            return render_template('result.html', 
                                  prediction = response['result'], 
                                  probability = response.get('probability'))
    
    except Exception as exception:
        # Log the full exception for developers/admins
        logger.error(f'Prediction error: {str(exception)}', 
                     exc_info = True)
        
        # Return generic error message to users
        return jsonify({'Error': 'An error occurred processing the request. The system administrator has been notified..'}), 500


@app.route('/batch_predict', methods = ['POST'])
def batch_predict():
    try:
        # Wrap the file checking in try/except to catch RequestEntityTooLarge
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
        
            file = request.files['file']
        
        except werkzeug.exceptions.RequestEntityTooLarge:
            logger.warning('Request entity too large - file exceeds size limit')
            return jsonify({'error': 'The uploaded file is too large. Maximum size is 1MB'}), 413

         # Security check for filename
        filename = file.filename
        if filename and (os.path.sep in filename or '..' in filename):
            return jsonify({'error': 'Invalid filename'}), 400
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Ensure the dataframe has the expected columns
        missing_columns = [col for col in FEATURE_NAMES if col not in df.columns]
        if missing_columns:
            return jsonify({
                'error': f'Missing columns in uploaded file: {', '.join(missing_columns)}'
            }), 400
        
        # Select only the needed features
        input_df = df[FEATURE_NAMES]
        
        # Load model and make predictions
        model = load_model('RandomForestClassifier_model')
        predictions = model.predict(input_df)

         # Calculate probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)[:, 1] * 100
            df['Probability'] = probabilities.round(2)
        
        # Add predictions to the dataframe
        df['Prediction'] = predictions
        df['Result'] = df['Prediction'].apply(lambda x: 'Diabetic' if x == 1 else 'Non-Diabetic')

        # Create summary statistics
        total_records = len(df)
        diabetic_count = int(df['Prediction'].sum())
        percent_diabetic = (diabetic_count / total_records * 100) if total_records > 0 else 0
        
        summary = {
            'total_records': total_records,
            'diabetic_count': diabetic_count,
            'non_diabetic_count': total_records - diabetic_count,
            'percent_diabetic': round(percent_diabetic, 2)
        }
        
        # Provide option to download as CSV
        csv_data = df.to_csv(index=False)
        
        # Return either JSON or render template based on request type
        if request.is_json:
            return jsonify({
                'predictions': df[['Prediction', 'Result']].to_dict('records'),
                'summary': summary
            })
        else:
            # Pass the dataframe to the template
            return render_template('batch_results.html', 
                                 df = df.head(100),  # Limit to first 100 rows for display
                                 summary = summary,
                                 csv_data = csv_data,
                                 total_rows = len(df),
                                 displayed_rows = min(100, len(df)))

    except FileNotFoundError:
        logger.error('Model file not found', 
                     exc_info = True)
        return jsonify({'error': 'The prediction model could not be found. Please contact support..'}), 500
        
    except ValueError as exception:
        # Only if it's safe to show this message
        logger.error(f'Invalid input data: {str(exception)}', 
                     exc_info = True)
        return jsonify({'error': 'The input data format is invalid. Please check your file and try again..'}), 400
        
    except Exception as exception:
        # Log full details for debugging
        logger.error(f'Batch prediction error: {str(exception)}', 
                     exc_info = True)
        
        # Return generic message to users
        return jsonify({'error': 'An error occurred processing your batch prediction. The system administrator has been notified..'}), 500

if __name__ == '__main__':
    # app.run(debug = True, port = 3100)            # for localhost default (127.0.0.1) run
    
    # Flask app listens to 0.0.0.0 host running inside a Docker container
    app.run(host = '0.0.0.0',
            port = int(os.environ.get('PORT', 3100)), 
            debug = True)    
    