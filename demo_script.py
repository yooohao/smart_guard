"""
SmartGuard: AI-Driven Security for Smart Homes
Demo Script for Project Presentation

This script demonstrates the capabilities of the SmartGuard attack detection system
using simulated network traffic data.
"""

import os
import time
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from modules.utils import load_model, live_detection_demo, create_directory
from modules.visualizer import Visualizer

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SmartGuard Demo Script')
    
    parser.add_argument('--model_path', type=str, default='output/models/random_forest.pkl',
                        help='Path to the trained model file')
    
    parser.add_argument('--scaler_path', type=str, default='output/models/scaler.pkl',
                        help='Path to the fitted scaler file')
    
    parser.add_argument('--output_dir', type=str, default='output/demo',
                        help='Directory to save demo outputs')
    
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of simulated network traffic samples')
    
    parser.add_argument('--multi_class', action='store_true',
                        help='Run demo for multi-class attack detection')
    
    parser.add_argument('--attack_types_path', type=str, default=None,
                        help='Path to the attack types mapping file (for multi-class)')
    
    return parser.parse_args()

def generate_binary_demo_plot(results, threshold=0.5, output_dir='output/demo'):
    """Generate a visualization of the demo results for binary classification"""
    # Sort by probability for better visualization
    sorted_results = results.sort_values('attack_probability')
    
    # Prepare data for plotting
    probs = sorted_results['attack_probability'].values
    true_labels = sorted_results['is_ddos'].values
    pred_labels = sorted_results['predicted_attack'].values
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Create probability plot
    plt.subplot(2, 1, 1)
    colors = ['green' if y == 0 else 'red' for y in true_labels]
    plt.bar(range(len(probs)), probs, color=colors, alpha=0.7)
    plt.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
    plt.ylabel('DDoS Attack Probability', fontsize=12)
    plt.title('SmartGuard DDoS Detection - Live Demo', fontsize=16)
    
    # Add legend for probability plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Normal Traffic'),
        Patch(facecolor='red', alpha=0.7, label='DDoS Attack'),
        Patch(facecolor='black', label='Detection Threshold')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    # Create detection accuracy plot
    plt.subplot(2, 1, 2)
    correct_mask = true_labels == pred_labels
    plt.scatter(range(len(probs)), correct_mask, c=['green' if x else 'red' for x in correct_mask], alpha=0.7)
    plt.yticks([0, 1], ['Incorrect', 'Correct'])
    plt.xlabel('Network Flow Sample (sorted by probability)', fontsize=12)
    plt.ylabel('Detection Accuracy', fontsize=12)
    
    # Calculate and display accuracy statistics
    accuracy = correct_mask.mean()
    true_positive = ((pred_labels == 1) & (true_labels == 1)).sum()
    false_positive = ((pred_labels == 1) & (true_labels == 0)).sum()
    true_negative = ((pred_labels == 0) & (true_labels == 0)).sum()
    false_negative = ((pred_labels == 0) & (true_labels == 1)).sum()
    
    plt.figtext(0.5, 0.01, 
                f"Accuracy: {accuracy:.2f} | True Positives: {true_positive} | False Positives: {false_positive} | "
                f"True Negatives: {true_negative} | False Negatives: {false_negative}",
                ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # Save and show plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'binary_detection_demo.png'), dpi=300)
    plt.close()
    
    print(f"Demo visualization saved to {os.path.join(output_dir, 'binary_detection_demo.png')}")

def generate_multiclass_demo_plot(results, attack_types=None, output_dir='output/demo'):
    """Generate a visualization of the demo results for multi-class classification"""
    # 准备数据 - 使用数字标签和名称标签
    if 'attack_type_num' in results.columns:
        # 使用数字标签进行排序和处理
        true_numeric_labels = results['attack_type_num'].values
        pred_numeric_labels = results['predicted_attack'].values
        correct_mask = results['correctly_classified'].values
        
        # 获取字符串标签用于显示
        if 'predicted_attack_name' in results.columns:
            # 使用已映射的名称
            true_string_labels = results['attack_type'].values
            pred_string_labels = results['predicted_attack_name'].values
        else:
            # 如果没有映射的名称，使用数字标签
            true_string_labels = [str(label) for label in true_numeric_labels]
            pred_string_labels = [str(label) for label in pred_numeric_labels]
    else:
        # 旧版本的结果可能没有数字标签列
        true_labels = results['attack_type'].values
        pred_labels = results['predicted_attack'].values
        
        # 确保标签类型一致 (全部转为字符串)
        true_string_labels = np.array([str(label) for label in true_labels])
        pred_string_labels = np.array([str(label) for label in pred_labels])
        true_numeric_labels = np.array([float(i) for i in range(len(true_labels))])
        pred_numeric_labels = np.array([float(i) for i in range(len(pred_labels))])
        correct_mask = results['correctly_classified'].values
    
    # 按攻击概率排序
    sorted_indices = np.argsort(results['attack_probability'].values)[::-1]  # 倒序排列
    
    true_string_labels = np.array(true_string_labels)[sorted_indices]
    pred_string_labels = np.array(pred_string_labels)[sorted_indices]
    true_numeric_labels = np.array(true_numeric_labels)[sorted_indices]
    pred_numeric_labels = np.array(pred_numeric_labels)[sorted_indices]
    correct_mask = np.array(correct_mask)[sorted_indices]
    probs = results['attack_probability'].values[sorted_indices]
    
    # 创建图形
    plt.figure(figsize=(15, 12))
    
    # 创建概率图
    plt.subplot(3, 1, 1)
    plt.bar(range(len(probs)), probs, alpha=0.7)
    plt.ylabel('Attack Probability', fontsize=12)
    plt.title('SmartGuard Multi-Attack Detection - Live Demo', fontsize=16)
    
    # 创建真实与预测攻击类型图
    plt.subplot(3, 1, 2)
    
    # 获取唯一攻击类型
    if attack_types is not None:
        unique_attacks = list(attack_types.values())
    else:
        # 安全地获取所有唯一的标签字符串
        # 修复此处的类型混合问题
        # all_string_labels = np.concatenate([true_string_labels, pred_string_labels])
        # unique_attacks = np.unique(all_string_labels).tolist()
        
        all_string_labels = np.concatenate([
        np.array(true_string_labels, dtype=str), 
        np.array(pred_string_labels, dtype=str)
        ])
        unique_attacks = np.unique(all_string_labels).tolist()
    
    # 创建从攻击类型到数字值的映射
    attack_to_num = {attack: i for i, attack in enumerate(unique_attacks)}
    
    # 将标签转换为数字
    try:
        true_nums = np.array([attack_to_num.get(label, -1) for label in true_string_labels])
        pred_nums = np.array([attack_to_num.get(label, -1) for label in pred_string_labels])
    except:
        # 备用方案
        true_nums = np.array(range(len(true_string_labels)))
        pred_nums = np.array(range(len(pred_string_labels)))
    
    # 绘图
    plt.scatter(range(len(true_nums)), true_nums, marker='o', color='blue', label='True Attack Type')
    plt.scatter(range(len(pred_nums)), pred_nums, marker='x', color='red', label='Predicted Attack Type')
    
    plt.yticks(range(len(unique_attacks)), unique_attacks)
    plt.xlabel('Network Flow Sample', fontsize=12)
    plt.ylabel('Attack Type', fontsize=12)
    plt.legend(loc='upper right')
    
    # 创建检测准确率图
    plt.subplot(3, 1, 3)
    plt.scatter(range(len(correct_mask)), correct_mask, c=['green' if x else 'red' for x in correct_mask], alpha=0.7)
    plt.yticks([0, 1], ['Incorrect', 'Correct'])
    plt.xlabel('Network Flow Sample', fontsize=12)
    plt.ylabel('Detection Accuracy', fontsize=12)
    
    # 计算并显示准确率统计
    accuracy = correct_mask.mean()
    
    # 使用数字标签计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    try:
        cm = confusion_matrix(true_numeric_labels, pred_numeric_labels)
    except:
        # 备用方案 - 创建空矩阵
        n_classes = len(unique_attacks)
        cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # 显示整体准确率
    plt.figtext(0.5, 0.01, 
                f"Overall Accuracy: {accuracy:.2f}",
                ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # 保存图
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'multiclass_detection_demo.png'), dpi=300)
    plt.close()
    
    # 生成混淆矩阵热图
    plt.figure(figsize=(12, 10))
    import seaborn as sns
    
    # 创建归一化混淆矩阵以便更好的可视化
    with np.errstate(divide='ignore', invalid='ignore'):  # 忽略除零警告
        cm_normalized = np.true_divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis])
        cm_normalized[~np.isfinite(cm_normalized)] = 0  # 替换无穷值为0
    
    # 使用seaborn获得更好的外观
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=unique_attacks, yticklabels=unique_attacks)
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Attack Type', fontsize=14)
    plt.ylabel('True Attack Type', fontsize=14)
    
    # 旋转标签以提高可读性
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multiclass_confusion_matrix.png'), dpi=300)
    plt.close()
    
    print(f"Demo visualizations saved to:")
    print(f"  - {os.path.join(output_dir, 'multiclass_detection_demo.png')}")
    print(f"  - {os.path.join(output_dir, 'multiclass_confusion_matrix.png')}")

def load_attack_types(file_path):
    """Load attack types mapping from a file"""
    if file_path is None:
        return None
    
    try:
        with open(file_path, 'rb') as f:
            attack_types = pickle.load(f)
        print(f"Attack types loaded from {file_path}")
        return attack_types
    except Exception as e:
        print(f"Error loading attack types from {file_path}: {e}")
        
        # Create a simple mapping for demo
        print("Creating default attack types mapping")
        attack_types = {
            0: 'BENIGN',
            1: 'DDoS',
            2: 'DoS',
            3: 'MITM',
            4: 'Recon',
            5: 'Mirai',
            6: 'Brute_Force'
        }
        return attack_types

def run_demo(args):
    """Run the SmartGuard demo"""
    print("\n==== Running SmartGuard Attack Detection Demo ====\n")
    
    # Create output directory
    create_directory(args.output_dir)
    
    # Load model
    model = load_model(args.model_path)
    if model is None:
        print(f"Error: Could not load model from {args.model_path}")
        return
    
    # Load scaler
    try:
        scaler = joblib.load(args.scaler_path)
        print(f"Scaler loaded from {args.scaler_path}")
    except:
        print(f"Warning: Could not load scaler from {args.scaler_path}. Using None.")
        scaler = None
    
    # For multi-class, load attack types
    attack_types = None
    if args.multi_class:
        attack_types = load_attack_types(args.attack_types_path)
    
    # 获取模型期望的特征数量
    n_features_expected = 46  # 从错误消息中得知模型期望的特征数量
    
    # 创建虚拟特征名称列表
    dummy_feature_names = [f'feature_{i}' for i in range(n_features_expected)]
    
    # Run live detection demo
    results = live_detection_demo(
        model, 
        scaler,
        feature_names=dummy_feature_names,  # 传递虚拟特征名称
        n_samples=args.samples,
        multi_class=args.multi_class
    )
    
    # Generate demo visualization
    if args.multi_class:
        generate_multiclass_demo_plot(results, attack_types, args.output_dir)
    else:
        generate_binary_demo_plot(results, output_dir=args.output_dir)
    
    # Additional visualizations
    visualizer = Visualizer(args.output_dir)
    
    # Feature importance if using Random Forest
    if hasattr(model, 'feature_importances_'):
        if args.multi_class:
            # For multi-class
            feature_cols = [col for col in results.columns if col not in ['attack_type', 'predicted_attack', 'attack_probability', 'correctly_classified']]
        else:
            # For binary classification
            feature_cols = [col for col in results.columns if col not in ['is_ddos', 'predicted_attack', 'attack_probability', 'correctly_classified']]
        
        visualizer.plot_feature_importance(model, feature_cols)
    
    print("\n==== SmartGuard Demo Completed ====")

if __name__ == "__main__":
    args = parse_arguments()
    run_demo(args)