#!/usr/bin/env python3
"""
Test script for Deep Learning Insurance Prediction

This script tests the main functionality of the insurance prediction model.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import accuracy_score

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from insurance_prediction import InsurancePredictor
from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG


class TestInsurancePredictor(unittest.TestCase):
    """Test cases for InsurancePredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = InsurancePredictor()
        
    def test_data_loading(self):
        """Test data loading functionality."""
        data = self.predictor.load_data()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape[1], 3)  # age, affordability, bought_insurance
        self.assertGreater(data.shape[0], 0)
        
    def test_data_preparation(self):
        """Test data preparation functionality."""
        self.predictor.load_data()
        self.predictor.prepare_data()
        
        # Check that data is split
        self.assertIsNotNone(self.predictor.X_train)
        self.assertIsNotNone(self.predictor.X_test)
        self.assertIsNotNone(self.predictor.y_train)
        self.assertIsNotNone(self.predictor.y_test)
        
        # Check shapes
        self.assertEqual(self.predictor.X_train.shape[1], 2)  # 2 features
        self.assertEqual(self.predictor.X_test.shape[1], 2)
        
    def test_model_building(self):
        """Test model building functionality."""
        self.predictor.load_data()
        self.predictor.prepare_data()
        self.predictor.build_model()
        
        self.assertIsNotNone(self.predictor.model)
        self.assertEqual(self.predictor.model.input_shape[1], 2)  # 2 input features
        self.assertEqual(self.predictor.model.output_shape[1], 1)  # 1 output (binary)
        
    def test_model_training(self):
        """Test model training functionality."""
        self.predictor.load_data()
        self.predictor.prepare_data()
        self.predictor.build_model()
        
        # Train with fewer epochs for testing
        history = self.predictor.train_model(epochs=5)
        
        self.assertIsNotNone(history)
        self.assertIn('accuracy', history.history)
        self.assertIn('loss', history.history)
        
    def test_model_evaluation(self):
        """Test model evaluation functionality."""
        self.predictor.load_data()
        self.predictor.prepare_data()
        self.predictor.build_model()
        self.predictor.train_model(epochs=5)
        
        accuracy, y_pred, y_pred_proba = self.predictor.evaluate_model()
        
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        self.assertEqual(len(y_pred), len(self.predictor.y_test))
        
    def test_prediction_functionality(self):
        """Test prediction functionality."""
        self.predictor.load_data()
        self.predictor.prepare_data()
        self.predictor.build_model()
        self.predictor.train_model(epochs=5)
        
        # Test prediction for a new customer
        age = 30
        affordability = 1
        prediction, probability = self.predictor.predict_new_customer(age, affordability)
        
        self.assertIn(prediction, [0, 1])
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        
    def test_model_saving_loading(self):
        """Test model saving and loading functionality."""
        self.predictor.load_data()
        self.predictor.prepare_data()
        self.predictor.build_model()
        self.predictor.train_model(epochs=5)
        
        # Save model
        model_path = 'test_model.h5'
        self.predictor.save_model(model_path)
        self.assertTrue(os.path.exists(model_path))
        
        # Load model
        new_predictor = InsurancePredictor()
        new_predictor.load_model(model_path)
        self.assertIsNotNone(new_predictor.model)
        
        # Clean up
        os.remove(model_path)
        
    def test_data_validation(self):
        """Test data validation."""
        self.predictor.load_data()
        
        # Check for required columns
        required_columns = ['age', 'affordibility', 'bought_insurance']
        for col in required_columns:
            self.assertIn(col, self.predictor.data.columns)
            
        # Check data types
        self.assertTrue(self.predictor.data['age'].dtype in [np.int64, np.float64])
        self.assertTrue(self.predictor.data['affordibility'].dtype in [np.int64, np.float64])
        self.assertTrue(self.predictor.data['bought_insurance'].dtype in [np.int64, np.float64])
        
        # Check for missing values
        self.assertEqual(self.predictor.data.isnull().sum().sum(), 0)
        
    def test_configuration_loading(self):
        """Test configuration loading."""
        self.assertIsNotNone(DATA_CONFIG)
        self.assertIsNotNone(MODEL_CONFIG)
        self.assertIsNotNone(TRAINING_CONFIG)
        
        # Check required config keys
        self.assertIn('data_path', DATA_CONFIG)
        self.assertIn('features', DATA_CONFIG)
        self.assertIn('target', DATA_CONFIG)


def run_quick_test():
    """Run a quick end-to-end test."""
    print("Running quick end-to-end test...")
    
    try:
        predictor = InsurancePredictor()
        
        # Load data
        data = predictor.load_data()
        print(f"✓ Data loaded successfully: {data.shape}")
        
        # Prepare data
        predictor.prepare_data()
        print(f"✓ Data prepared: {predictor.X_train.shape[0]} training, {predictor.X_test.shape[0]} test samples")
        
        # Build and train model
        predictor.build_model()
        history = predictor.train_model(epochs=10)  # Quick training
        print("✓ Model trained successfully")
        
        # Evaluate model
        accuracy, _, _ = predictor.evaluate_model()
        print(f"✓ Model evaluated: {accuracy:.3f} accuracy")
        
        # Test prediction
        prediction, probability = predictor.predict_new_customer(35, 1)
        print(f"✓ Prediction test: Age 35, Affordability 1 -> {prediction} (prob: {probability:.3f})")
        
        print("\n🎉 All tests passed! The project is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    # Run quick test first
    if run_quick_test():
        # Run full unit tests
        print("\nRunning full unit tests...")
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        print("Quick test failed. Please check your setup.")
        sys.exit(1) 