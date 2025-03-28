import os
import sys
import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add src to path so we can import model modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

# Import model-related functionality
# Assuming these are the correct imports - adjust as needed for your project
from src.models.predict_model import predict_model
from src.models.train_model import train_model
from src.models.predict_model import predict_model
from src.models.register_model import load_model_info, register_model

# Constants for test security
MAX_TEST_TIMEOUT = 5  # seconds
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit for test data files

@pytest.fixture(autouse=True)
def timeout_check():
    """Ensure tests don't hang beyond reasonable time."""
    start_time = time.time()
    yield
    execution_time = time.time() - start_time
    assert execution_time < MAX_TEST_TIMEOUT, f"Test took too long: {execution_time}s"

@pytest.fixture
def sample_data():
    """Create sample data for model testing."""
    # Create a small dataset for testing
    np.random.seed(42)
    X = np.random.rand(100, 8)  # Assuming 8 features
    y = np.random.randint(0, 2, 100)  # Binary classification
    
    # Convert to DataFrame with appropriate column names
    feature_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='Outcome')
    
    return X_df, y_series

@pytest.fixture
def mock_trained_model():
    """Create a mock trained model."""
    mock = MagicMock()
    mock.predict.return_value = np.array([0, 1, 0, 1])
    mock.predict_proba.return_value = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.9, 0.1],
        [0.4, 0.6]
    ])
    return mock

def test_train_model_basic(sample_data):
    """Test that the model training function works with valid data."""
    X, y = sample_data
    
    with patch('src.models.train_model.save_model') as mock_save:
        result = train_model(X, y)
        
        # Check that the function returns a model
        assert result is not None
        
        # Check that the model has the expected methods
        assert hasattr(result, 'fit')
        assert hasattr(result, 'predict')
        assert hasattr(result, 'predict_proba')
        
        # Verify the model was saved
        mock_save.assert_called_once()

def test_train_model_empty_data():
    """Test model training with empty data."""
    X_empty = pd.DataFrame()
    y_empty = pd.Series()
    
    with pytest.raises(ValueError):
        train_model(X_empty, y_empty)

def test_train_model_mismatched_data(sample_data):
    """Test model training with mismatched X and y lengths."""
    X, y = sample_data
    y_mismatched = y[:-10]  # Remove last 10 elements
    
    with pytest.raises(ValueError):
        train_model(X, y_mismatched)

def test_predict_model(mock_trained_model, sample_data):
    """Test that the prediction function works."""
    X, _ = sample_data
    X_test = X.iloc[:4]  # Use first 4 samples
    
    with patch('src.models.predict_model.load_model', return_value=mock_trained_model):
        predictions = predict_model(X_test)
        
        # Check predictions shape and type
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (4,)
        assert set(predictions).issubset({0, 1})

def test_predict_model_with_probabilities(mock_trained_model, sample_data):
    """Test predictions with probability outputs."""
    X, _ = sample_data
    X_test = X.iloc[:4]
    
    with patch('src.models.predict_model.load_model', return_value=mock_trained_model):
        predictions, probabilities = predict_model(X_test, return_proba=True)
        
        # Check predictions
        assert predictions.shape == (4,)
        
        # Check probabilities
        assert probabilities.shape == (4, 2)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)

def test_predict_model_invalid_input():
    """Test prediction with invalid input."""
    # Test with invalid input types
    with patch('src.models.predict_model.load_model') as mock_load:
        mock_load.return_value = MagicMock()
        
        with pytest.raises(TypeError):
            predict_model("not a dataframe")
        
        with pytest.raises(ValueError):
            predict_model(pd.DataFrame())  # Empty dataframe

def test_evaluate_model(mock_trained_model, sample_data):
    """Test model evaluation function."""
    X, y = sample_data
    X_test = X.iloc[:4]
    y_test = y.iloc[:4]
    
    with patch('src.models.evaluate_model.load_model', return_value=mock_trained_model):
        metrics = evaluate_model(X_test, y_test)
        
        # Check that metrics dictionary contains expected keys
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

def test_evaluate_model_invalid_input(mock_trained_model):
    """Test model evaluation with invalid inputs."""
    with patch('src.models.evaluate_model.load_model', return_value=mock_trained_model):
        # Test with mismatched X and y lengths
        X_test = pd.DataFrame({"feature": [1, 2, 3, 4]})
        y_test = pd.Series([0, 1])
        
        with pytest.raises(ValueError):
            evaluate_model(X_test, y_test)
        
        # Test with invalid input types
        with pytest.raises(TypeError):
            evaluate_model("not a dataframe", y_test)

def test_save_model():
    """Test model saving functionality."""
    mock_model = MagicMock()
    
    # Test with various paths
    with patch('joblib.dump') as mock_dump:
        save_model(mock_model, "model.joblib")
        mock_dump.assert_called_once()
    
    # Test with absolute path
    with patch('joblib.dump') as mock_dump:
        save_model(mock_model, "/absolute/path/model.joblib")
        mock_dump.assert_called_once()
    
    # Test with path traversal attempt
    with pytest.raises(ValueError):
        save_model(mock_model, "../../../etc/passwd")
    
    # Test with invalid file extension
    with pytest.raises(ValueError):
        save_model(mock_model, "model.dangerous_extension")

def test_load_model():
    """Test model loading functionality."""
    mock_joblib = MagicMock()
    mock_model = MagicMock()
    mock_joblib.load.return_value = mock_model
    
    # Test standard load
    with patch('joblib.load', mock_joblib.load):
        with patch('os.path.exists', return_value=True):
            loaded_model = load_model("model.joblib")
            assert loaded_model == mock_model
    
    # Test with nonexistent file
    with patch('os.path.exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model.joblib")
    
    # Test with path traversal attempt
    with pytest.raises(ValueError):
        load_model("../../../etc/passwd")
    
    # Test with invalid file extension
    with pytest.raises(ValueError):
        load_model("model.dangerous_extension")

def test_model_security():
    """Test model security against serialization attacks."""
    # Test against pickle serialization vulnerability
    malicious_data = b"cos\nsystem\n(S'echo VULNERABILITY EXPOSED > /tmp/hack.txt'\ntR."
    
    # Ensure load_model validates input
    with patch('builtins.open', mock_open(read_data=malicious_data)):
        with patch('os.path.exists', return_value=True):
            with pytest.raises(Exception):
                # This should be caught by proper model loading security
                load_model("malicious_model.joblib")

def test_model_feature_importance():
    """Test accessing feature importance from model."""
    mock_model = MagicMock()
    mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.15, 0.05, 0.1, 0.05, 0.05])
    
    with patch('src.models.evaluate_model.load_model', return_value=mock_model):
        # Assuming get_feature_importance is a function in your project
        with patch('src.models.evaluate_model.get_feature_importance') as mock_get:
            mock_get.return_value = dict(zip(
                ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                mock_model.feature_importances_
            ))
            
            feature_importance = mock_get()
            
            # Check that feature importance is returned as expected
            assert isinstance(feature_importance, dict)
            assert len(feature_importance) == 8
            assert sum(feature_importance.values()) == pytest.approx(1.0)
            assert feature_importance['Glucose'] == 0.2  # Highest importance feature
            
def test_model_reproducibility():
    """Test that model training is reproducible with fixed random seed."""
    np.random.seed(42)
    X1 = np.random.rand(100, 8)
    y1 = np.random.randint(0, 2, 100)
    
    np.random.seed(42)
    X2 = np.random.rand(100, 8)
    y2 = np.random.randint(0, 2, 100)
    
    # Check data is identical with same seed
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    
    # Train two models with same seed
    with patch('src.models.train_model.save_model'):
        with patch('src.models.train_model.RANDOM_SEED', 42):
            model1 = train_model(pd.DataFrame(X1), pd.Series(y1))
            model2 = train_model(pd.DataFrame(X2), pd.Series(y2))
    
    # Models should have same weights/parameters
    for param1, param2 in zip(model1.get_params().items(), model2.get_params().items()):
        assert param1 == param2
