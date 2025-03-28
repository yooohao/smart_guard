# modules/model_training.py

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

class ModelTrainer:
    """Class to train different machine learning models for attack detection"""
    
    def __init__(self, output_dir='models'):
        """Initialize the model trainer
        
        Args:
            output_dir: Directory to save trained models
        """
        self.output_dir = output_dir
        
    def train_models(self, X_train, y_train, hyperparameter_tuning=False, multi_class=False):
        """Train multiple machine learning models
        
        Args:
            X_train: Training features
            y_train: Training labels
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            multi_class: Whether to perform multi-class classification
            
        Returns:
            Dictionary of trained models
        """
        print("\nTraining machine learning models...")
        
        # Define models with parameters
        models = {}
        
        # Set multi_class solver for logistic regression if multi_class=True
        if multi_class:
            # For multi-class classification
            models['logistic_regression'] = LogisticRegression(
                max_iter=1000, 
                multi_class='multinomial',
                solver='lbfgs',
                random_state=42
            )
            
            # Random Forest
            models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            # SVM
            models['svm'] = SVC(
                probability=True,
                decision_function_shape='ovo',  # one-vs-one for multi-class
                random_state=42
            )
            
            # Decision Tree
            models['decision_tree'] = DecisionTreeClassifier(
                random_state=42
            )
        else:
            # For binary classification
            models['logistic_regression'] = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
            
            # Random Forest
            models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            # SVM
            models['svm'] = SVC(
                probability=True,
                random_state=42
            )
            
            # Decision Tree
            models['decision_tree'] = DecisionTreeClassifier(
                random_state=42
            )
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            print("Performing hyperparameter tuning (this may take a while)...")
            models = self._tune_hyperparameters(models, X_train, y_train, multi_class)
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Save the model
            model_path = os.path.join(self.output_dir, f"{name}.pkl")
            joblib.dump(model, model_path)
            
            trained_models[name] = model
        
        print("Model training completed!")
        return trained_models
    
    def _tune_hyperparameters(self, models, X_train, y_train, multi_class=False):
        """Tune hyperparameters for each model
        
        Args:
            models: Dictionary of models
            X_train: Training features
            y_train: Training labels
            multi_class: Whether to perform multi-class classification
            
        Returns:
            Dictionary of tuned models
        """
        # Define parameter grids
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf']
            },
            'decision_tree': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        }
        
        # Adjust param grid for multi-class if needed
        if multi_class:
            param_grids['logistic_regression']['multi_class'] = ['multinomial']
            param_grids['logistic_regression']['solver'] = ['lbfgs']  # Only lbfgs works with multinomial
        
        # Perform grid search for each model
        tuned_models = {}
        for name, model in models.items():
            print(f"Tuning hyperparameters for {name}...")
            
            # Create a smaller grid for SVM to save time
            if name == 'svm' and len(X_train) > 10000:
                # Use a subset of data for SVM tuning
                subset_size = min(10000, len(X_train))
                indices = np.random.choice(len(X_train), subset_size, replace=False)
                X_subset, y_subset = X_train[indices], y_train[indices]
                
                grid_search = GridSearchCV(
                    model, param_grids[name], cv=3, n_jobs=-1, verbose=1
                )
                grid_search.fit(X_subset, y_subset)
            else:
                grid_search = GridSearchCV(
                    model, param_grids[name], cv=3, n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train, y_train)
            
            # Get the best model
            tuned_models[name] = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
        
        return tuned_models