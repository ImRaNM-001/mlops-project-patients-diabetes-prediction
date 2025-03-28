import sys
import time
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO
from unittest.mock import patch, MagicMock

# Add src to path so we can import app
sys.path.append(str(Path(__file__).parent.parent.resolve()))            
from flask_app.app import app, FEATURE_NAMES

# Constants for test security
MAX_TEST_TIMEOUT = 5  # seconds
MAX_FILE_SIZE = 1024 ** 2  # 1MB limit for test file uploads

@pytest.fixture(scope = 'module')
def client():
    """
    Create a test client for the app
    """
    # Ensure we're in a testing environment
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    
    with app.test_client() as client:
        yield client

@pytest.fixture(scope = 'module')
def mock_model():
    """
    Create a mock model for testing
    """
    mock = MagicMock()
    mock.predict.return_value = [0]  # Non-diabetic
    mock.predict_proba.return_value = [[0.8, 0.2]]  # 20% chance of diabetes
    return mock

@pytest.fixture(autouse = True)
def timeout_check():
    """
    Ensure tests don't hang beyond reasonable time
    """
    start_time = time.time()
    yield
    execution_time = time.time() - start_time
    assert execution_time < MAX_TEST_TIMEOUT, f'Test took too long: {execution_time}s'

@patch('flask_app.app.load_model')
def test_home_page(mock_load_model, client):
    """
    Test that the home page loads correctly
    """
    response = client.get('/')
    assert response.status_code == 200
    assert b'html' in response.data

@patch('flask_app.app.load_model')
def test_predict_form_data(mock_load_model, client, mock_model):
    """
    Test prediction with form data
    """
    # Setup mock to return proper structure
    mock_model.predict.return_value = [1]  # Single prediction of class 1
    mock_model.predict_proba.return_value = [[0.8, 0.2]]        # 20% probability for class 1
    mock_load_model.return_value = mock_model
    
    # Create form data with all features
    form_data = {feature: '1.0' for feature in FEATURE_NAMES}
    
    response = client.post('/predict', data = form_data)
    assert response.status_code == 200
    
    # Check that the template was rendered with correct data
    assert b'Diabetic' in response.data
    assert b'20.0' in response.data         # 20% probability

@patch('flask_app.app.load_model')
def test_predict_form_data_invalid_input(mock_load_model, client, mock_model):
    """
    Test prediction with invalid form data for security
    """
    mock_load_model.return_value = mock_model

    # Create form data with script injection attempt
    form_data = {feature: '1.0' for feature in FEATURE_NAMES}
    form_data[FEATURE_NAMES[0]] = '<script>alert("xss")</script>'
    
    response = client.post('/predict', data = form_data)
    assert response.status_code == 400
    
    # Another test with SQL injection attempt
    form_data = {feature: '1.0' for feature in FEATURE_NAMES}
    form_data[FEATURE_NAMES[0]] = '1.0; DROP TABLE users;'
    
    response = client.post('/predict', data = form_data)
    assert response.status_code == 400

@patch('flask_app.app.load_model')
def test_predict_json_data(mock_load_model, client, mock_model):
    """
    Test prediction with JSON data
    """
    # CHANGE: Set mock to return class 0 explicitly
    mock_model.predict.return_value = [0]  # Non-diabetic
    mock_load_model.return_value = mock_model
    
    # Create JSON data with all features
    json_data = {feature: 1.0 for feature in FEATURE_NAMES}
    
    response = client.post('/predict', 
                          json = json_data,
                          content_type = 'application/json')
    assert response.status_code == 200
    
    # Parse the response JSON
    data = json.loads(response.data)
    assert data['prediction'] == 0
    assert data['result'] == 'Non-Diabetic'
    assert data['probability'] == 20.0

@patch('flask_app.app.load_model')
def test_predict_error_handling(mock_load_model, client):
    """
    Test error handling in predict endpoint
    """
    mock_load_model.side_effect = Exception('Test error')
    
    # Create JSON data with all features
    json_data = {feature: 1.0 for feature in FEATURE_NAMES}
    
    response = client.post('/predict', 
                          json = json_data,
                          content_type = 'application/json')
    assert response.status_code == 500
    assert b'error' in response.data.lower()
    
    # Don't expose detailed error messages
    assert b'Test error' not in response.data

@patch('flask_app.app.load_model')
def test_batch_predict(mock_load_model, client, mock_model):
    """
    Test batch prediction with CSV file
    """
    mock_model.predict.return_value = [1, 0, 1]
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    mock_load_model.return_value = mock_model
    
    # Create a test DataFrame
    df = pd.DataFrame({feature: [1.0, 2.0, 3.0] for feature in FEATURE_NAMES})
    
    # Convert DataFrame to CSV in memory
    csv_data = BytesIO()
    df.to_csv(csv_data, index = False)
    csv_data.seek(0)
    
    # Send the CSV file
    response = client.post('/batch_predict',
                          data = {'file': (csv_data, 'test.csv')},
                          content_type = 'multipart/form-data')
    
    assert response.status_code == 200
    assert b'Diabetic' in response.data or b'Non-Diabetic' in response.data

@patch('flask_app.app.load_model')
def test_batch_predict_file_size_limit(mock_load_model, client, mock_model):
    """
    Test batch prediction with oversized file
    """

    mock_load_model.return_value = mock_model
    
    # Create a large test DataFrame that exceeds limit
    large_df = pd.DataFrame({
        feature: [float(i) for i in range(100000)] 
        for feature in FEATURE_NAMES
    })
    
    # Convert DataFrame to CSV in memory
    csv_data = BytesIO()
    large_df.to_csv(csv_data, index = False)
    csv_data.seek(0)
    
    # Should be rejected due to file size
    # with pytest.raises(Exception):
    response = client.post('/batch_predict',
                        data = {'file': (csv_data, 'large_test.csv')},
                        content_type = 'multipart/form-data')
    
    assert response.status_code == 413  # Request Entity Too Large
    
@patch('flask_app.app.load_model')
def test_batch_predict_missing_file(mock_load_model, client):
    """
    Test batch prediction with missing file
    """
    response = client.post('/batch_predict')
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data
    assert 'No file uploaded' in data['error']

@patch('flask_app.app.load_model')
def test_batch_predict_missing_columns(mock_load_model, client):
    """
    Test batch prediction with missing columns
    """
    # Create a DataFrame with missing columns
    df = pd.DataFrame({'some_column': [1.0, 2.0]})
    
    # Convert DataFrame to CSV in memory
    csv_data = BytesIO()
    df.to_csv(csv_data, index = False)
    csv_data.seek(0)
    
    # Send the CSV file
    response = client.post('/batch_predict',
                          data = {'file': (csv_data, 'test.csv')},
                          content_type = 'multipart/form-data')
    
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Missing columns' in data['error']

@patch('flask_app.app.load_model')
def test_batch_predict_malicious_file(mock_load_model, client):
    """
    Test batch prediction with potentially malicious file
    """
    # Create a malicious filename with path traversal attempt
    malicious_filename = '../../../etc/passwd'
    
    # Create a simple DataFrame
    df = pd.DataFrame({feature: [1.0] for feature in FEATURE_NAMES})
    
    # Convert DataFrame to CSV in memory
    csv_data = BytesIO()
    df.to_csv(csv_data, index = False)
    csv_data.seek(0)
    
    # Send the CSV file with malicious filename
    response = client.post('/batch_predict',
                          data = {'file': (csv_data, malicious_filename)},
                          content_type = 'multipart/form-data')
    
    # Application should reject this
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data
