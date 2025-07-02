#!/usr/bin/env python3
"""
Deep Learning Insurance Prediction Script

This script implements a neural network model to predict insurance purchase
likelihood based on customer age and affordability status.

Author: Deep Learning Insurance Project
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class InsurancePredictor:
    """
    A class to handle insurance prediction using deep learning.
    """
    
    def __init__(self, data_path='simple_insurance_data.csv'):
        """
        Initialize the InsurancePredictor.
        
        Args:
            data_path (str): Path to the CSV data file
        """
        self.data_path = data_path
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and prepare the dataset."""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.data.shape[0]} records, {self.data.shape[1]} features")
        print(f"Features: {list(self.data.columns)}")
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n=== Data Exploration ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"\nFirst 5 rows:")
        print(self.data.head())
        
        print(f"\nData types:")
        print(self.data.dtypes)
        
        print(f"\nSummary statistics:")
        print(self.data.describe())
        
        print(f"\nTarget variable distribution:")
        print(self.data['bought_insurance'].value_counts())
        
        # Create visualizations
        self._create_visualizations()
    
    def _create_visualizations(self):
        """Create data visualization plots."""
        plt.figure(figsize=(15, 10))
        
        # Age distribution
        plt.subplot(2, 3, 1)
        plt.hist(self.data['age'], bins=10, alpha=0.7, color='skyblue')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        
        # Affordability distribution
        plt.subplot(2, 3, 2)
        self.data['affordibility'].value_counts().plot(kind='bar', color='lightgreen')
        plt.title('Affordability Distribution')
        plt.xlabel('Affordability')
        plt.ylabel('Count')
        
        # Insurance purchase distribution
        plt.subplot(2, 3, 3)
        self.data['bought_insurance'].value_counts().plot(kind='bar', color='lightcoral')
        plt.title('Insurance Purchase Distribution')
        plt.xlabel('Bought Insurance')
        plt.ylabel('Count')
        
        # Age vs Insurance purchase
        plt.subplot(2, 3, 4)
        plt.scatter(self.data['age'], self.data['bought_insurance'], alpha=0.6)
        plt.title('Age vs Insurance Purchase')
        plt.xlabel('Age')
        plt.ylabel('Bought Insurance')
        
        # Correlation heatmap
        plt.subplot(2, 3, 5)
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        # Age distribution by insurance purchase
        plt.subplot(2, 3, 6)
        self.data.boxplot(column='age', by='bought_insurance')
        plt.title('Age Distribution by Insurance Purchase')
        plt.suptitle('')  # Remove default title
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare data for training by splitting and scaling.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        print("\n=== Data Preparation ===")
        
        # Separate features and target
        X = self.data[['age', 'affordibility']].values
        y = self.data['bought_insurance'].values
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Features scaled using StandardScaler")
    
    def build_model(self, input_dim=2):
        """
        Build the neural network model.
        
        Args:
            input_dim (int): Number of input features
        """
        print("\n=== Building Model ===")
        
        self.model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Model architecture:")
        self.model.summary()
    
    def train_model(self, epochs=100, batch_size=8, validation_split=0.2):
        """
        Train the neural network model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Proportion of training data for validation
        """
        print("\n=== Training Model ===")
        
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Plot training history
        self._plot_training_history(history)
        
        return history
    
    def _plot_training_history(self, history):
        """Plot training history."""
        plt.figure(figsize=(12, 4))
        
        # Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Precision
        plt.subplot(1, 3, 3)
        plt.plot(history.history['precision'], label='Training Precision')
        plt.plot(history.history['val_precision'], label='Validation Precision')
        plt.title('Model Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self):
        """Evaluate the trained model."""
        print("\n=== Model Evaluation ===")
        
        # Make predictions
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        self._plot_confusion_matrix(cm)
        
        return accuracy, y_pred, y_pred_proba
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Insurance', 'Insurance'],
                   yticklabels=['No Insurance', 'Insurance'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_new_customer(self, age, affordability):
        """
        Make prediction for a new customer.
        
        Args:
            age (int): Customer age
            affordability (int): Affordability status (0 or 1)
        
        Returns:
            tuple: (prediction, probability)
        """
        # Prepare input
        customer_data = np.array([[age, affordability]])
        customer_data_scaled = self.scaler.transform(customer_data)
        
        # Make prediction
        probability = self.model.predict(customer_data_scaled)[0][0]
        prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability
    
    def save_model(self, model_path='insurance_model.h5'):
        """Save the trained model."""
        self.model.save(model_path)
        print(f"\nModel saved to: {model_path}")
    
    def load_model(self, model_path='insurance_model.h5'):
        """Load a trained model."""
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")


def main():
    """Main function to run the insurance prediction pipeline."""
    print("=== Deep Learning Insurance Prediction ===\n")
    
    # Initialize predictor
    predictor = InsurancePredictor()
    
    # Load data
    predictor.load_data()
    
    # Explore data
    predictor.explore_data()
    
    # Prepare data
    predictor.prepare_data()
    
    # Build model
    predictor.build_model()
    
    # Train model
    history = predictor.train_model(epochs=100)
    
    # Evaluate model
    accuracy, y_pred, y_pred_proba = predictor.evaluate_model()
    
    # Save model
    predictor.save_model()
    
    # Example predictions
    print("\n=== Example Predictions ===")
    test_customers = [
        (25, 1),  # Young customer with affordability
        (45, 0),  # Middle-aged customer without affordability
        (60, 1),  # Older customer with affordability
        (30, 0),  # Young customer without affordability
    ]
    
    for age, affordability in test_customers:
        prediction, probability = predictor.predict_new_customer(age, affordability)
        status = "Likely to buy" if prediction == 1 else "Unlikely to buy"
        print(f"Age: {age}, Affordability: {affordability} -> {status} (Probability: {probability:.3f})")
    
    print("\n=== Pipeline Complete ===")
    print("Generated files:")
    print("- data_analysis.png: Data exploration visualizations")
    print("- training_history.png: Model training history")
    print("- confusion_matrix.png: Model evaluation results")
    print("- insurance_model.h5: Trained model file")


if __name__ == "__main__":
    main() 