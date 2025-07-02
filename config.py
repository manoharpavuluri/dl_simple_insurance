"""
Configuration file for Deep Learning Insurance Prediction Project

This file contains all the configurable parameters and settings for the project.
"""

# Data Configuration
DATA_CONFIG = {
    'data_path': 'simple_insurance_data.csv',
    'features': ['age', 'affordibility'],
    'target': 'bought_insurance',
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True
}

# Model Configuration
MODEL_CONFIG = {
    'input_dim': 2,
    'layers': [
        {'units': 64, 'activation': 'relu', 'dropout': 0.3},
        {'units': 32, 'activation': 'relu', 'dropout': 0.2},
        {'units': 16, 'activation': 'relu', 'dropout': 0.0},
        {'units': 1, 'activation': 'sigmoid', 'dropout': 0.0}
    ],
    'optimizer': 'adam',
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy', 'precision', 'recall']
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 8,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'figure_size': (15, 10),
    'dpi': 300,
    'save_format': 'png',
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis'
}

# File Paths
PATHS = {
    'model_save_path': 'insurance_model.h5',
    'scaler_save_path': 'scaler.pkl',
    'data_analysis_plot': 'data_analysis.png',
    'training_history_plot': 'training_history.png',
    'confusion_matrix_plot': 'confusion_matrix.png',
    'feature_importance_plot': 'feature_importance.png'
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'threshold': 0.5,
    'cv_folds': 5,
    'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1']
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'insurance_prediction.log'
}

# Example customers for testing
EXAMPLE_CUSTOMERS = [
    {'age': 25, 'affordibility': 1, 'description': 'Young customer with affordability'},
    {'age': 45, 'affordibility': 0, 'description': 'Middle-aged customer without affordability'},
    {'age': 60, 'affordibility': 1, 'description': 'Older customer with affordability'},
    {'age': 30, 'affordibility': 0, 'description': 'Young customer without affordability'},
    {'age': 50, 'affordibility': 1, 'description': 'Middle-aged customer with affordability'},
    {'age': 35, 'affordibility': 1, 'description': 'Young adult with affordability'}
]

# Feature engineering options
FEATURE_ENGINEERING = {
    'create_age_groups': True,
    'age_group_bins': [18, 30, 45, 60, 100],
    'age_group_labels': ['Young', 'Young Adult', 'Middle-aged', 'Senior'],
    'create_interaction_features': True,
    'normalize_features': True
}

# Model comparison options
MODEL_COMPARISON = {
    'compare_models': True,
    'models_to_compare': ['neural_network', 'logistic_regression', 'random_forest', 'xgboost'],
    'cross_validation_folds': 5
} 