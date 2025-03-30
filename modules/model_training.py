# modules/model_training.py

import os
import time
import joblib
import numpy as np
import sys
import threading
from sklearn.ensemble import RandomForestClassifier
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
            
            # Decision Tree
            models['decision_tree'] = DecisionTreeClassifier(
                random_state=42
            )
        
        # Track total training time
        total_start_time = time.time()
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            print("Performing hyperparameter tuning (this may take a while)...")
            tuning_start_time = time.time()
            models = self._tune_hyperparameters(models, X_train, y_train, multi_class)
            tuning_time = time.time() - tuning_start_time
            print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds ({tuning_time/60:.2f} minutes)")
        
        # Train models
        trained_models = {}
        model_times = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model_start_time = time.time()
            
            # Set up the timer display
            stop_timer = threading.Event()
            
            def update_timer():
                while not stop_timer.is_set():
                    elapsed = time.time() - model_start_time
                    # Use a fixed model name in the timer display
                    sys.stdout.write(f"\rTraining {name}: Elapsed time: {elapsed:.2f} seconds")
                    sys.stdout.flush()
                    time.sleep(0.5)  # Update twice per second
            
            # Start the timer thread
            timer_thread = threading.Thread(target=update_timer)
            timer_thread.daemon = True
            timer_thread.start()
            
            try:
                # Train the model
                model.fit(X_train, y_train)
            finally:
                # Ensure we always stop the timer
                stop_timer.set()
                timer_thread.join(timeout=1.0)  # Wait for thread to finish but not indefinitely
            
            # Calculate training time
            model_training_time = time.time() - model_start_time
            model_times[name] = model_training_time
            
            # Print final time (overwrites the running timer)
            sys.stdout.write("\r" + " " * 70 + "\r")  # Clear the line
            print(f"Training {name} completed in {model_training_time:.2f} seconds ({model_training_time/60:.2f} minutes)")
            
            # Save the model
            model_path = os.path.join(self.output_dir, f"{name}.pkl")
            joblib.dump(model, model_path)
            
            trained_models[name] = model
        
        total_time = time.time() - total_start_time
        print(f"\nModel training completed! Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        # Display training time summary
        print("\nTraining Time Summary:")
        print("-" * 60)
        print(f"{'Model':<20} {'Time (seconds)':<15} {'Time (minutes)':<15}")
        print("-" * 60)
        for name, training_time in model_times.items():
            print(f"{name:<20} {training_time:.2f}s{'':<10} {training_time/60:.2f}m{'':<10}")
        if hyperparameter_tuning:
            print(f"{'Hyperparameter Tuning':<20} {tuning_time:.2f}s{'':<10} {tuning_time/60:.2f}m{'':<10}")
        print(f"{'Total':<20} {total_time:.2f}s{'':<10} {total_time/60:.2f}m{'':<10}")
        print("-" * 60)
        
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
        model_tuning_times = {}
        
        for name, model in models.items():
            print(f"\nTuning hyperparameters for {name}...")
            model_tune_start = time.time()
            
            # Set up the timer display
            stop_timer = threading.Event()
            
            def update_timer():
                while not stop_timer.is_set():
                    elapsed = time.time() - model_tune_start
                    # Use fixed model name in the timer display
                    sys.stdout.write(f"\rTuning {name}: Elapsed time: {elapsed:.2f} seconds")
                    sys.stdout.flush()
                    time.sleep(0.5)  # Update twice per second
            
            # Start the timer thread
            timer_thread = threading.Thread(target=update_timer)
            timer_thread.daemon = True
            timer_thread.start()
            
            try:
                # Perform grid search
                grid_search = GridSearchCV(
                    model, param_grids[name], cv=3, n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train, y_train)
            finally:
                # Ensure we always stop the timer
                stop_timer.set()
                timer_thread.join(timeout=1.0)  # Wait for thread to finish but not indefinitely
            
            # Get the best model
            tuned_models[name] = grid_search.best_estimator_
            model_tuning_time = time.time() - model_tune_start
            model_tuning_times[name] = model_tuning_time
            
            # Clear the line and print the final result
            sys.stdout.write("\r" + " " * 70 + "\r")
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Hyperparameter tuning for {name} completed in {model_tuning_time:.2f} seconds ({model_tuning_time/60:.2f} minutes)")
        
        # Display tuning time summary
        print("\nHyperparameter Tuning Time Summary:")
        print("-" * 60)
        print(f"{'Model':<20} {'Time (seconds)':<15} {'Time (minutes)':<15}")
        print("-" * 60)
        for name, tuning_time in model_tuning_times.items():
            print(f"{name:<20} {tuning_time:.2f}s{'':<10} {tuning_time/60:.2f}m{'':<10}")
        total_tuning_time = sum(model_tuning_times.values())
        print(f"{'Total':<20} {total_tuning_time:.2f}s{'':<10} {total_tuning_time/60:.2f}m{'':<10}")
        print("-" * 60)
        
        return tuned_models