"""
Visualization module for SmartGuard
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

class Visualizer:
    """
    Class for visualizing attack detection results
    """
    
    def __init__(self, output_dir):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        """
        self.output_dir = output_dir
    
    def plot_feature_importance(self, model, feature_names, top_n=20):
        """
        Plot feature importance from the model.
        
        Parameters:
        -----------
        model : trained model
            Model with feature_importances_ attribute
        feature_names : list
            Names of features
        top_n : int
            Number of top features to display
        """
        print("\nAnalyzing and plotting feature importance...")
        
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute. Skipping feature importance plot.")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        
        # Get top N features (or all if fewer than N)
        top_n = min(top_n, len(feature_names))
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title("Top Features for Attack Detection", fontsize=16)
        plt.barh(range(top_n), top_importances, align="center", color='darkblue')
        plt.yticks(range(top_n), top_features)
        plt.xlabel("Feature Importance", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "feature_importance.png"), dpi=300)
        plt.close()
        
        print(f"Feature importance plot saved to {os.path.join(self.output_dir, 'feature_importance.png')}")
        
        # Return top features and their importance scores
        top_features_dict = {feature: importance for feature, importance in zip(top_features, top_importances)}
        
        print("\nTop 10 Features for Attack Detection:")
        for i, (feature, importance) in enumerate(list(top_features_dict.items())[:10], 1):
            print(f"{i}. {feature}: {importance:.4f}")
        
        return top_features_dict
    
    def plot_roc_curves(self, metrics, multi_class=False, results=None, attack_types=None):
        """
        Plot ROC curves for all models.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary containing evaluation metrics for each model
        multi_class : bool
            Whether to plot ROC curves for multi-class classification
        results : dict
            Dictionary containing prediction results (needed for multi-class)
        attack_types : dict
            Mapping from encoded labels to attack type names (for multi-class)
        """
        print("\nPlotting ROC curves...")
        
        if multi_class and results is not None:
            self._plot_multiclass_roc(results, attack_types)
        else:
            # Binary classification ROC
            plt.figure(figsize=(10, 8))
            
            for model_name, model_metrics in metrics.items():
                if 'fpr' in model_metrics and 'tpr' in model_metrics:
                    fpr = model_metrics['fpr']
                    tpr = model_metrics['tpr']
                    auc_score = model_metrics['auc']
                    
                    label = f"{model_name.replace('_', ' ').title()} (AUC = {auc_score:.3f})"
                    color = 'blue' if model_name == 'random_forest' else 'red'
                    plt.plot(fpr, tpr, color=color, lw=2, label=label)
            
            plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title('ROC Curves for Attack Detection', fontsize=16)
            plt.legend(loc="lower right")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(self.output_dir, "roc_curves.png"), dpi=300)
            plt.close()
            
            print(f"ROC curves saved to {os.path.join(self.output_dir, 'roc_curves.png')}")
    
    def _plot_multiclass_roc(self, results, attack_types):
        """
        Plot ROC curves for multi-class classification.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing prediction results
        attack_types : dict
            Mapping from encoded labels to attack type names
        """
        for model_name, result in results.items():
            if 'y_proba' not in result:
                continue
                
            y_true = result['y_true']
            y_proba = result['y_proba']
            
            # Get number of classes
            n_classes = y_proba.shape[1]
            
            # Binarize the labels for ROC calculation
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            # Calculate ROC curve and ROC area for each class
            fpr = {}
            tpr = {}
            roc_auc = {}
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot ROC curves
            plt.figure(figsize=(12, 10))
            
            # Use different colors for each class
            colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
            
            # Only plot top classes (if there are many)
            max_classes_to_plot = 10
            classes_to_plot = sorted(range(n_classes), key=lambda i: roc_auc[i], reverse=True)[:max_classes_to_plot]
            
            for i, color in zip(classes_to_plot, colors):
                attack_name = attack_types[i] if attack_types and i in attack_types else f"Class {i}"
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{attack_name} (AUC = {roc_auc[i]:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title(f'Multi-class ROC Curves - {model_name.replace("_", " ").title()}', fontsize=16)
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save the plot
            output_path = os.path.join(self.output_dir, f"multiclass_roc_{model_name}.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            print(f"Multi-class ROC curves for {model_name} saved to {output_path}")
    
    def plot_confusion_matrices(self, metrics, attack_types=None, multi_class=False):
        """
        Plot confusion matrices for all models.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary containing evaluation metrics for each model
        attack_types : dict
            Mapping of encoded labels to attack type names (for multi-class)
        multi_class : bool
            Whether plotting for multi-class classification
        """
        print("\nPlotting confusion matrices...")
        
        for model_name, model_metrics in metrics.items():
            if 'confusion_matrix' in model_metrics:
                cm = model_metrics['confusion_matrix']
                
                # For multi-class, we'll use a different approach for better visualization
                if multi_class and attack_types is not None:
                    plt.figure(figsize=(12, 10))
                    
                    # Get attack names for labels
                    labels = [attack_types[i] for i in range(len(attack_types))]
                    
                    # Normalize confusion matrix for better visualization
                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    
                    # Plot using seaborn for better appearance
                    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', 
                                xticklabels=labels, yticklabels=labels, cbar=False)
                    
                    # Rotate labels for better readability
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    plt.yticks(fontsize=10)
                else:
                    # Binary classification - simpler matrix
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                
                plt.xlabel('Predicted Label', fontsize=12)
                plt.ylabel('True Label', fontsize=12)
                plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"confusion_matrix_{model_name}.png"), dpi=300)
                plt.close()
                
                print(f"Confusion matrix for {model_name} saved to {os.path.join(self.output_dir, f'confusion_matrix_{model_name}.png')}")
    
    def plot_performance_comparison(self, metrics, multi_class=False):
        """
        Plot performance comparison bar chart for all models.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary containing evaluation metrics for each model
        multi_class : bool
            Whether the comparison is for multi-class classification
        """
        print("\nPlotting performance comparison...")
        
        if multi_class:
            metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            metric_labels = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)']
        else:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
            metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Prepare data for bar chart
        model_names = list(metrics.keys())
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        plt.figure(figsize=(12, 8))
        
        for i, model_name in enumerate(model_names):
            values = [metrics[model_name].get(metric, 0) for metric in metrics_to_plot]
            offset = width * (i - 0.5 * (len(model_names) - 1))
            plt.bar(x + offset, values, width, label=model_name.replace('_', ' ').title())
        
        plt.xlabel('Metrics', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.title('Performance Comparison of Models', fontsize=16)
        plt.xticks(x, metric_labels)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.output_dir, "performance_comparison.png"), dpi=300)
        plt.close()
        
        print(f"Performance comparison chart saved to {os.path.join(self.output_dir, 'performance_comparison.png')}")
    
    def plot_attack_distribution(self, attack_distribution, attack_types=None):
        """
        Plot the distribution of attack types in the dataset.
        
        Parameters:
        -----------
        attack_distribution : dict or Series
            Distribution of attacks in the dataset
        attack_types : dict, optional
            Mapping from encoded labels to attack type names
        """
        print("\nPlotting attack distribution...")
        
        plt.figure(figsize=(14, 8))
        
        # Convert distribution to proper format
        if isinstance(attack_distribution, dict):
            labels = []
            values = []
            
            for attack_type, count in attack_distribution.items():
                if attack_types and isinstance(attack_type, int):
                    label = attack_types.get(attack_type, f"Type {attack_type}")
                else:
                    label = str(attack_type)
                
                labels.append(label)
                values.append(count)
        else:
            # Assume it's a pandas Series
            values = attack_distribution.values
            labels = attack_distribution.index
            
            # Convert numeric indices to names if attack_types provided
            if attack_types:
                labels = [attack_types.get(i, f"Type {i}") if isinstance(i, int) else str(i) 
                         for i in labels]
        
        # Sort by count for better visualization
        sorted_indices = np.argsort(values)[::-1]
        sorted_values = [values[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]
        
        # Limit to top categories if there are many
        max_categories = 15
        if len(sorted_labels) > max_categories:
            other_count = sum(sorted_values[max_categories:])
            sorted_values = sorted_values[:max_categories] + [other_count]
            sorted_labels = sorted_labels[:max_categories] + ["Other"]
        
        # Create bar chart
        bars = plt.bar(range(len(sorted_values)), sorted_values, color='darkblue')
        
        # Add percentages
        total = sum(sorted_values)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = height / total * 100
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Attack Type', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title('Distribution of Attack Types', fontsize=16)
        plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "attack_distribution.png"), dpi=300)
        plt.close()
        
        print(f"Attack distribution chart saved to {os.path.join(self.output_dir, 'attack_distribution.png')}")
    
    def plot_comparison_by_attack_type(self, results, attack_types):
        """
        Plot model performance comparison for each attack type.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing prediction results
        attack_types : dict
            Mapping of encoded labels to attack type names
        """
        print("\nPlotting performance by attack type...")
        
        model_names = list(results.keys())
        
        # Calculate F1 score per class for each model
        class_f1_scores = {}
        
        for model_name, result in results.items():
            y_true = result['y_true']
            y_pred = result['y_pred']
            
            # Initialize dictionary for this model
            model_scores = {}
            
            # Get unique classes
            classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
            
            # Calculate F1 score for each class
            from sklearn.metrics import f1_score
            for cls in classes:
                # Create binary indicators for this class
                true_bin = (y_true == cls)
                pred_bin = (y_pred == cls)
                
                if true_bin.sum() > 0:  # Skip if no true instances
                    f1 = f1_score(true_bin, pred_bin)
                    
                    # Convert class ID to name if available
                    class_name = attack_types.get(cls, f"Type {cls}") if attack_types else f"Class {cls}"
                    model_scores[class_name] = f1
            
            class_f1_scores[model_name] = model_scores
        
        # Get all unique attack types across all models
        all_attack_types = set()
        for model_scores in class_f1_scores.values():
            all_attack_types.update(model_scores.keys())
        
        # Sort attack types by average F1 score
        attack_type_avg_f1 = {}
        for attack_type in all_attack_types:
            scores = [scores.get(attack_type, 0) for scores in class_f1_scores.values()]
            attack_type_avg_f1[attack_type] = np.mean(scores) if scores else 0
        
        sorted_attack_types = sorted(attack_type_avg_f1.keys(), key=lambda x: attack_type_avg_f1[x], reverse=True)
        
        # Limit to top categories if there are many
        max_categories = 12
        if len(sorted_attack_types) > max_categories:
            sorted_attack_types = sorted_attack_types[:max_categories]
        
        # Plot comparison
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(sorted_attack_types))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            model_scores = class_f1_scores[model_name]
            values = [model_scores.get(attack_type, 0) for attack_type in sorted_attack_types]
            offset = width * (i - 0.5 * (len(model_names) - 1))
            
            bars = plt.bar(x + offset, values, width, label=model_name.replace('_', ' ').title())
        
        plt.xlabel('Attack Type', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.title('Model Performance by Attack Type (F1 Score)', fontsize=16)
        plt.xticks(x, sorted_attack_types, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "performance_by_attack_type.png"), dpi=300)
        plt.close()
        
        print(f"Performance by attack type chart saved to {os.path.join(self.output_dir, 'performance_by_attack_type.png')}")