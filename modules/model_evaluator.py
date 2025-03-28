"""
Model evaluation module for SmartGuard
"""

import os
import time
import pickle
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

class ModelEvaluator:
    """
    Class for evaluating machine learning models for attack detection
    """
    
    def __init__(self, output_dir):
        """
        Initialize the model evaluator.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save evaluation results
        """
        self.output_dir = output_dir
    
    def evaluate_models(self, models, X_test, y_test, attack_types=None, multi_class=False):
        """
        Evaluate the trained models on test data.
        
        Parameters:
        -----------
        models : dict
            Dictionary of trained models
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        attack_types : dict, optional
            Mapping from encoded labels to attack type names (for multi-class)
        multi_class : bool
            Whether evaluation is for multi-class classification
            
        Returns:
        --------
        tuple
            Evaluation metrics and prediction results
        """
        print("\n--- Evaluating Models ---")
        
        metrics = {}
        results = {}
        
        # Metrics to evaluate
        metrics_funcs = {
            'accuracy': accuracy_score
        }
        
        # Add additional metrics based on classification type
        if multi_class:
            # For multi-class, use different averaging methods
            metrics_funcs.update({
                'precision_macro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
                'recall_macro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
                'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
                'precision_weighted': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
                'recall_weighted': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
                'f1_weighted': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
            })
        else:
            # For binary classification, use standard metrics
            metrics_funcs.update({
                'precision': precision_score,
                'recall': recall_score,
                'f1': f1_score
            })
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name.replace('_', ' ').title()}:")
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - start_time
            
            # Initialize metrics dictionary for this model
            metrics[model_name] = {'inference_time': inference_time}
            
            # Calculate basic metrics
            for metric_name, metric_func in metrics_funcs.items():
                score = metric_func(y_test, y_pred)
                metrics[model_name][metric_name] = score
                print(f"{metric_name.capitalize()}: {score:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics[model_name]['confusion_matrix'] = cm
            
            # ROC curve and AUC for binary classification
            if not multi_class and hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                metrics[model_name]['fpr'] = fpr
                metrics[model_name]['tpr'] = tpr
                metrics[model_name]['auc'] = roc_auc
                
                print(f"AUC: {roc_auc:.4f}")
                
                # Store probabilities
                results[model_name] = {
                    'y_pred': y_pred,
                    'y_true': y_test,
                    'y_proba': y_proba
                }
            else:
                # For multi-class or models without predict_proba
                results[model_name] = {
                    'y_pred': y_pred,
                    'y_true': y_test
                }
                
                if hasattr(model, "predict_proba"):
                    results[model_name]['y_proba'] = model.predict_proba(X_test)
            
            # Classification report
            if multi_class and attack_types is not None:
                # Create custom target names for readability
                target_names = [attack_types[i] for i in range(len(attack_types))]
                report = classification_report(y_test, y_pred, target_names=target_names)
            else:
                report = classification_report(y_test, y_pred)
                
            print("\nClassification Report:")
            print(report)
            
            print(f"Inference time: {inference_time:.4f} seconds")
        
        # Save evaluation metrics
        self._save_metrics(metrics, results)
        
        return metrics, results
    
    def _save_metrics(self, metrics, results):
        """
        Save evaluation metrics to disk.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of evaluation metrics
        results : dict
            Dictionary of prediction results
        """
        # Save metrics
        metrics_path = os.path.join(self.output_dir, "evaluation_metrics.pkl")
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        # Save prediction results
        results_path = os.path.join(self.output_dir, "prediction_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Evaluation metrics and results saved to {self.output_dir}")
    
    def generate_summary_report(self, metrics, dataset_info, report_path, multi_class=False, attack_types=None):
        """
        Generate a comprehensive summary report of the detection system.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of evaluation metrics
        dataset_info : dict
            Information about the dataset
        report_path : str
            Path to save the report
        multi_class : bool
            Whether the report is for multi-class classification
        attack_types : dict
            Mapping from encoded labels to attack type names (for multi-class)
            
        Returns:
        --------
        dict
            Summary data
        """
        print("\n--- Generating Summary Report ---")
        
        # Prepare summary data
        summary = {
            "Dataset": {
                "Total Samples": dataset_info["total_samples"],
                "Training Samples": dataset_info["training_samples"],
                "Testing Samples": dataset_info["testing_samples"],
            },
            "Models": {}
        }
        
        # Add attack distribution information
        if multi_class and "class_distribution" in dataset_info:
            summary["Attack Distribution"] = dataset_info["class_distribution"]
        elif not multi_class and "attack_ratio" in dataset_info:
            summary["Dataset"]["Attack Samples (%)"] = f"{dataset_info['attack_ratio'] * 100:.2f}%"
        
        # Add model metrics
        for model_name in metrics:
            model_metrics = {}
            
            # Common metrics
            model_metrics["Accuracy"] = f"{metrics[model_name].get('accuracy', 0):.4f}"
            model_metrics["Inference Time (s)"] = f"{metrics[model_name].get('inference_time', 0):.4f}"
            
            if multi_class:
                # Multi-class metrics
                model_metrics["Precision (Macro)"] = f"{metrics[model_name].get('precision_macro', 0):.4f}"
                model_metrics["Recall (Macro)"] = f"{metrics[model_name].get('recall_macro', 0):.4f}"
                model_metrics["F1 Score (Macro)"] = f"{metrics[model_name].get('f1_macro', 0):.4f}"
                model_metrics["Precision (Weighted)"] = f"{metrics[model_name].get('precision_weighted', 0):.4f}"
                model_metrics["Recall (Weighted)"] = f"{metrics[model_name].get('recall_weighted', 0):.4f}"
                model_metrics["F1 Score (Weighted)"] = f"{metrics[model_name].get('f1_weighted', 0):.4f}"
            else:
                # Binary classification metrics
                model_metrics["Precision"] = f"{metrics[model_name].get('precision', 0):.4f}"
                model_metrics["Recall"] = f"{metrics[model_name].get('recall', 0):.4f}"
                model_metrics["F1 Score"] = f"{metrics[model_name].get('f1', 0):.4f}"
                model_metrics["AUC"] = f"{metrics[model_name].get('auc', 0):.4f}"
            
            summary["Models"][model_name.replace('_', ' ').title()] = model_metrics
        
        # Generate report as text file
        with open(report_path, 'w') as f:
            f.write("======================================================================\n")
            f.write("                SmartGuard: AI-Driven Security for Smart Homes         \n")
            
            if multi_class:
                f.write("              Multi-Class Attack Detection - Summary Report         \n")
            else:
                f.write("                  DDoS Attack Detection - Summary Report           \n")
                
            f.write("======================================================================\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("------------------\n")
            for key, value in summary["Dataset"].items():
                f.write(f"{key}: {value}\n")
            
            if multi_class and "Attack Distribution" in summary:
                f.write("\nATTACK DISTRIBUTION\n")
                f.write("------------------\n")
                for attack_type, count in summary["Attack Distribution"].items():
                    if attack_types and isinstance(attack_type, int):
                        attack_name = attack_types.get(attack_type, f"Type {attack_type}")
                        f.write(f"{attack_name}: {count}\n")
                    else:
                        f.write(f"{attack_type}: {count}\n")
            
            f.write("\n")
            f.write("MODEL PERFORMANCE\n")
            f.write("----------------\n")
            
            for model_name, metrics in summary["Models"].items():
                f.write(f"\n{model_name}:\n")
                for metric_name, metric_value in metrics.items():
                    f.write(f"  {metric_name}: {metric_value}\n")
            
            f.write("\n")
            f.write("CONCLUSION\n")
            f.write("----------\n")
            
            # Determine which model performed better
            if len(metrics) > 1:
                if multi_class:
                    # Use F1 macro for multi-class
                    model_f1_scores = {model: float(metrics["F1 Score (Macro)"]) 
                                     for model, metrics in summary["Models"].items()}
                else:
                    # Use F1 for binary classification
                    model_f1_scores = {model: float(metrics["F1 Score"]) 
                                     for model, metrics in summary["Models"].items()}
                    
                best_model = max(model_f1_scores, key=model_f1_scores.get)
                
                if multi_class:
                    f.write(f"Based on F1 score (macro), the {best_model} model performed better for attack classification.\n")
                    f.write(f"This model achieved an accuracy of {summary['Models'][best_model]['Accuracy']} and a macro F1 score of {summary['Models'][best_model]['F1 Score (Macro)']}.\n\n")
                else:
                    f.write(f"Based on F1 score, the {best_model} model performed better for DDoS attack detection.\n")
                    f.write(f"This model achieved an accuracy of {summary['Models'][best_model]['Accuracy']} and an F1 score of {summary['Models'][best_model]['F1 Score']}.\n\n")
            
            if multi_class:
                f.write("The results demonstrate the effectiveness of machine learning techniques for classifying different types of network attacks in smart home environments.\n")
                f.write("This approach could be integrated into smart home security systems to provide comprehensive protection against a variety of network-based threats.\n")
            else:
                f.write("The results demonstrate the effectiveness of machine learning techniques for detecting DDoS attacks in smart home networks.\n")
                f.write("This approach could be integrated into smart home security systems to provide real-time protection against network-based threats.\n")
        
        print(f"Summary report generated and saved to {report_path}")
        
        return summary