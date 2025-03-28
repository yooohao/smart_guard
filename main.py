"""
SmartGuard: AI-Driven Security for Smart Homes
Main script to run the attack detection pipeline
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from modules.data_preprocessing import DataProcessor
from modules.model_training import ModelTrainer
from modules.model_evaluator import ModelEvaluator
from modules.visualizer import Visualizer
from modules.utils import create_directory

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SmartGuard: AI-Driven Attack Detection for Smart Homes')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset CSV file')
    
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs (models, figures, reports)')
    
    parser.add_argument('--tune_hyperparams', action='store_true',
                        help='Perform hyperparameter tuning')
    
    parser.add_argument('--test_size', type=float, default=0.25,
                        help='Proportion of data to use for testing (default: 0.25)')
    
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to load (to prevent memory issues)')
    
    parser.add_argument('--multi_class', action='store_true',
                        help='Perform multi-class classification instead of binary')
    
    parser.add_argument('--simulate_data', action='store_true',
                        help='Use simulated data instead of loading from file')
    
    return parser.parse_args()

def run_pipeline(args):
    """Run the full SmartGuard pipeline"""
    print("\n==== Running SmartGuard Attack Detection Pipeline ====\n")
    
    # Create output directories
    create_directory(args.output_dir)
    models_dir = os.path.join(args.output_dir, 'models')
    figures_dir = os.path.join(args.output_dir, 'figures')
    reports_dir = os.path.join(args.output_dir, 'reports')
    
    create_directory(models_dir)
    create_directory(figures_dir)
    create_directory(reports_dir)
    
    # Step 1---------------------: Load and preprocess data
    if args.simulate_data:
        from modules.utils import simulate_multi_attack_traffic
        
        # Generate simulated data
        print("\nGenerating simulated multi-attack traffic data...")
        data = simulate_multi_attack_traffic(n_samples=50000 if args.max_samples is None else args.max_samples)
        
        # Get features, labels, and attack types
        X = data.drop(['attack_type'], axis=1)
        y = data['attack_type'].astype('category').cat.codes
        feature_names = X.columns.tolist()
        
        # Create attack types mapping
        attack_categories = data['attack_type'].astype('category')
        attack_types = {i: category for i, category in enumerate(attack_categories.cat.categories)}
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Save scaler for demo
        import joblib
        joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
        
        # Calculate attack distribution
        attack_distribution = data['attack_type'].value_counts()
        multi_class = True
        
    else:
        # Load and process real data
        data_processor = DataProcessor(args.data_path)
        X_train, X_test, y_train, y_test, feature_names = data_processor.process_data(
            test_size=args.test_size,
            multi_class=args.multi_class
        )
        
        # Get attack types for multi-class classification
        if args.multi_class:
            attack_types = data_processor.get_attack_types()
            multi_class = True
        else:
            attack_types = None
            multi_class = False
    
    # Step 2---------------------: Train models
    model_trainer = ModelTrainer(output_dir=models_dir)
    models = model_trainer.train_models(
        X_train, y_train, 
        hyperparameter_tuning=args.tune_hyperparams,
        multi_class=multi_class
    )
    
    # Step 3---------------------: Evaluate models
    model_evaluator = ModelEvaluator(output_dir=reports_dir)
    metrics, results = model_evaluator.evaluate_models(
        models, X_test, y_test, 
        attack_types=attack_types,
        multi_class=multi_class
    )
    
    # Step 4---------------------: Visualize results
    visualizer = Visualizer(output_dir=figures_dir)
    
    # Plot feature importance (works for both binary and multi-class)
    visualizer.plot_feature_importance(models['random_forest'], feature_names)
    
    # Plot confusion matrices
    visualizer.plot_confusion_matrices(metrics, attack_types, multi_class)
    
    # Plot ROC curves
    visualizer.plot_roc_curves(metrics, multi_class, results, attack_types)
    
    # Plot performance comparison
    visualizer.plot_performance_comparison(metrics, multi_class)
    
    # For multi-class, plot additional visualizations
    if multi_class:
        # Plot attack distribution
        if args.simulate_data:
            visualizer.plot_attack_distribution(attack_distribution, attack_types)
        else:
            # Calculate class distribution from y_train and y_test
            all_y = np.concatenate([y_train, y_test])
            unique_values, counts = np.unique(all_y, return_counts=True)
            attack_dist = {attack_types[val]: count for val, count in zip(unique_values, counts)}
            visualizer.plot_attack_distribution(attack_dist)
        
        # Plot performance by attack type
        visualizer.plot_comparison_by_attack_type(results, attack_types)
    
    # Step 5---------------------: Generate summary report
    dataset_info = {
        "total_samples": len(X_train) + len(X_test),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
    }
    
    if multi_class:
        # Add class distribution information
        if args.simulate_data:
            dataset_info["class_distribution"] = attack_distribution.to_dict()
        else:
            all_y = np.concatenate([y_train, y_test])
            unique_values, counts = np.unique(all_y, return_counts=True)
            dataset_info["class_distribution"] = {attack_types[val]: int(count) for val, count in zip(unique_values, counts)}
    else:
        # Add attack ratio for binary classification
        dataset_info["attack_ratio"] = y_test.mean()
    
    model_evaluator.generate_summary_report(
        metrics=metrics,
        dataset_info=dataset_info,
        report_path=os.path.join(reports_dir, 'summary_report.txt'),
        multi_class=multi_class,
        attack_types=attack_types
    )
    
    print("\n==== SmartGuard Attack Detection Pipeline Completed ====")
    print(f"All outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)