"""
Utility functions for SmartGuard
"""

import os
import time
import pickle
import numpy as np
import pandas as pd

def create_directory(directory_path):
    """
    Create a directory if it doesn't exist.
    
    Parameters:
    -----------
    directory_path : str
        Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

# def load_model(model_path):
#     """
#     Load a trained model from disk.
    
#     Parameters:
#     -----------
#     model_path : str
#         Path to the saved model
        
#     Returns:
#     --------
#     object
#         Loaded model
#     """
#     try:
#         with open(model_path, 'rb') as f:
#             model = pickle.load(f)
#         print(f"Model loaded from {model_path}")
#         return model
#     except Exception as e:
#         print(f"Error loading model from {model_path}: {e}")
#         return None

def load_model(model_path):
    try:
        import joblib
        model = joblib.load(model_path)  # 使用joblib加载
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def simulate_ddos_traffic(n_samples=500, attack_ratio=0.3):
    """
    Simulate smart home network traffic data for DDoS detection.
    
    Parameters:
    -----------
    n_samples : int
        Number of network traffic samples to generate
    attack_ratio : float
        Proportion of samples that are DDoS attacks
        
    Returns:
    --------
    DataFrame with simulated network traffic features and labels
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate normal traffic samples
    n_normal = int(n_samples * (1 - attack_ratio))
    normal_data = {
        'packet_count': np.random.normal(20, 5, n_normal).astype(int),
        'flow_duration': np.random.normal(500, 100, n_normal),
        'avg_packet_size': np.random.normal(250, 50, n_normal),
        'flow_packets_per_sec': np.random.normal(0.5, 0.2, n_normal),
        'flow_bytes_per_sec': np.random.normal(125, 30, n_normal),
        'rst_count': np.random.normal(0.2, 0.4, n_normal),
        'urg_count': np.random.normal(0.1, 0.3, n_normal),
        'is_ddos': np.zeros(n_normal)
    }
    
    # Generate DDoS attack samples
    n_attack = n_samples - n_normal
    attack_data = {
        'packet_count': np.random.normal(120, 30, n_attack).astype(int),
        'flow_duration': np.random.normal(300, 80, n_attack),
        'avg_packet_size': np.random.normal(100, 20, n_attack),
        'flow_packets_per_sec': np.random.normal(8.0, 2.0, n_attack),
        'flow_bytes_per_sec': np.random.normal(800, 200, n_attack),
        'rst_count': np.random.normal(10.0, 3.0, n_attack),
        'urg_count': np.random.normal(8.0, 2.0, n_attack),
        'is_ddos': np.ones(n_attack)
    }
    
    # Combine normal and attack data
    combined_data = {}
    for key in normal_data.keys():
        combined_data[key] = np.concatenate([normal_data[key], attack_data[key]])
    
    # Convert to DataFrame
    df = pd.DataFrame(combined_data)
    
    # Add protocols (categorical feature)
    protocols = np.random.choice(['TCP', 'UDP', 'ICMP', 'HTTP'], size=n_samples, 
                                p=[0.5, 0.3, 0.1, 0.1])
    df['protocol'] = protocols
    
    # Add a few additional features for better modeling
    df['Number'] = np.random.normal(50, 15, n_samples)
    df['Weight'] = np.random.normal(20, 5, n_samples) 
    df['IAT'] = np.random.normal(100, 30, n_samples)
    df['Duration'] = np.random.exponential(100, n_samples)
    df['Variance'] = np.random.gamma(2, 2, n_samples)
    df['Radius'] = np.random.uniform(1, 10, n_samples)
    df['Header_Length'] = np.random.normal(40, 10, n_samples).astype(int)
    
    # One-hot encode protocol
    df = pd.get_dummies(df, columns=['protocol'])
    
    # Shuffle data
    return df.sample(frac=1).reset_index(drop=True)

def simulate_multi_attack_traffic(n_samples=5000):
    """
    Simulate smart home network traffic data with multiple attack types.
    
    Parameters:
    -----------
    n_samples : int
        Number of network traffic samples to generate
        
    Returns:
    --------
    DataFrame with simulated network traffic features and attack type labels
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Define attack types and their proportions(设定每种的攻击比例)
    attack_types = {
        'BENIGN': 0.35,
        'DDoS': 0.25,
        'DoS': 0.15,
        'MITM': 0.10,
        'Recon': 0.05,
        'Mirai': 0.05,
        'Brute_Force': 0.05
    }
    
    # Calculate sample counts
    attack_counts = {attack: int(n_samples * ratio) for attack, ratio in attack_types.items()}
    # Ensure we have exactly n_samples (account for rounding)
    remaining = n_samples - sum(attack_counts.values())
    attack_counts['BENIGN'] += remaining
    
    # Create feature sets for each attack type
    all_data = []
    
    # BENIGN traffic
    n_benign = attack_counts['BENIGN']
    benign_data = pd.DataFrame({
        'packet_count': np.random.normal(20, 5, n_benign).astype(int),
        'flow_duration': np.random.normal(500, 100, n_benign),
        'avg_packet_size': np.random.normal(250, 50, n_benign),
        'flow_packets_per_sec': np.random.normal(0.5, 0.2, n_benign),
        'flow_bytes_per_sec': np.random.normal(125, 30, n_benign),
        'rst_count': np.random.normal(0.2, 0.4, n_benign),
        'urg_count': np.random.normal(0.1, 0.3, n_benign),
        'fin_count': np.random.normal(0.5, 0.3, n_benign),
        'syn_count': np.random.normal(1.0, 0.4, n_benign),
        'psh_count': np.random.normal(0.8, 0.3, n_benign),
        'ack_count': np.random.normal(15.0, 5.0, n_benign),
        'arp_count': np.random.normal(0.2, 0.4, n_benign),
        'icmp_count': np.random.normal(0.1, 0.2, n_benign),
        'dns_query_count': np.random.normal(1.2, 0.6, n_benign),
        'unique_dst_ports': np.random.normal(2.0, 1.0, n_benign).astype(int),
        'unique_src_ports': np.random.normal(1.2, 0.5, n_benign).astype(int),
        'interval_variance': np.random.gamma(2, 2, n_benign),
        'attack_type': ['BENIGN'] * n_benign
    })
    all_data.append(benign_data)
    
    # DDoS attack traffic
    n_ddos = attack_counts['DDoS']
    ddos_data = pd.DataFrame({
        'packet_count': np.random.normal(120, 30, n_ddos).astype(int),
        'flow_duration': np.random.normal(300, 80, n_ddos),
        'avg_packet_size': np.random.normal(100, 20, n_ddos),
        'flow_packets_per_sec': np.random.normal(8.0, 2.0, n_ddos),
        'flow_bytes_per_sec': np.random.normal(800, 200, n_ddos),
        'rst_count': np.random.normal(10.0, 3.0, n_ddos),
        'urg_count': np.random.normal(8.0, 2.0, n_ddos),
        'fin_count': np.random.normal(0.1, 0.3, n_ddos),
        'syn_count': np.random.normal(15.0, 5.0, n_ddos),
        'psh_count': np.random.normal(12.0, 4.0, n_ddos),
        'ack_count': np.random.normal(5.0, 2.0, n_ddos),
        'arp_count': np.random.normal(0.1, 0.3, n_ddos),
        'icmp_count': np.random.normal(5.0, 2.0, n_ddos),
        'dns_query_count': np.random.normal(0.3, 0.5, n_ddos),
        'unique_dst_ports': np.random.normal(1.0, 0.2, n_ddos).astype(int),
        'unique_src_ports': np.random.normal(10.0, 3.0, n_ddos).astype(int),
        'interval_variance': np.random.gamma(1, 0.5, n_ddos),
        'attack_type': ['DDoS'] * n_ddos
    })
    all_data.append(ddos_data)
    
    # DoS attack traffic
    n_dos = attack_counts['DoS']
    dos_data = pd.DataFrame({
        'packet_count': np.random.normal(80, 20, n_dos).astype(int),
        'flow_duration': np.random.normal(350, 90, n_dos),
        'avg_packet_size': np.random.normal(120, 25, n_dos),
        'flow_packets_per_sec': np.random.normal(6.0, 1.5, n_dos),
        'flow_bytes_per_sec': np.random.normal(600, 150, n_dos),
        'rst_count': np.random.normal(8.0, 2.5, n_dos),
        'urg_count': np.random.normal(6.0, 1.5, n_dos),
        'fin_count': np.random.normal(0.2, 0.4, n_dos),
        'syn_count': np.random.normal(12.0, 4.0, n_dos),
        'psh_count': np.random.normal(10.0, 3.0, n_dos),
        'ack_count': np.random.normal(4.0, 1.5, n_dos),
        'arp_count': np.random.normal(0.1, 0.2, n_dos),
        'icmp_count': np.random.normal(3.0, 1.5, n_dos),
        'dns_query_count': np.random.normal(0.2, 0.4, n_dos),
        'unique_dst_ports': np.random.normal(1.0, 0.1, n_dos).astype(int),
        'unique_src_ports': np.random.normal(8.0, 2.0, n_dos).astype(int),
        'interval_variance': np.random.gamma(1, 0.4, n_dos),
        'attack_type': ['DoS'] * n_dos
    })
    all_data.append(dos_data)
    
    # MITM attack traffic
    n_mitm = attack_counts['MITM']
    mitm_data = pd.DataFrame({
        'packet_count': np.random.normal(40, 10, n_mitm).astype(int),
        'flow_duration': np.random.normal(450, 80, n_mitm),
        'avg_packet_size': np.random.normal(200, 40, n_mitm),
        'flow_packets_per_sec': np.random.normal(2.0, 0.8, n_mitm),
        'flow_bytes_per_sec': np.random.normal(400, 100, n_mitm),
        'rst_count': np.random.normal(0.5, 0.5, n_mitm),
        'urg_count': np.random.normal(0.3, 0.5, n_mitm),
        'fin_count': np.random.normal(0.8, 0.4, n_mitm),
        'syn_count': np.random.normal(2.0, 0.8, n_mitm),
        'psh_count': np.random.normal(2.5, 1.0, n_mitm),
        'ack_count': np.random.normal(10.0, 3.0, n_mitm),
        'arp_count': np.random.normal(3.0, 1.0, n_mitm),
        'icmp_count': np.random.normal(0.2, 0.4, n_mitm),
        'dns_query_count': np.random.normal(2.5, 1.0, n_mitm),
        'unique_dst_ports': np.random.normal(3.0, 1.0, n_mitm).astype(int),
        'unique_src_ports': np.random.normal(1.5, 0.5, n_mitm).astype(int),
        'interval_variance': np.random.gamma(3, 1.5, n_mitm),
        'attack_type': ['MITM'] * n_mitm
    })
    all_data.append(mitm_data)
    
    # Reconnaissance (Recon) attack traffic
    n_recon = attack_counts['Recon']
    recon_data = pd.DataFrame({
        'packet_count': np.random.normal(30, 8, n_recon).astype(int),
        'flow_duration': np.random.normal(400, 90, n_recon),
        'avg_packet_size': np.random.normal(180, 30, n_recon),
        'flow_packets_per_sec': np.random.normal(1.5, 0.6, n_recon),
        'flow_bytes_per_sec': np.random.normal(270, 60, n_recon),
        'rst_count': np.random.normal(1.0, 0.6, n_recon),
        'urg_count': np.random.normal(0.2, 0.4, n_recon),
        'fin_count': np.random.normal(1.2, 0.5, n_recon),
        'syn_count': np.random.normal(8.0, 2.0, n_recon),
        'psh_count': np.random.normal(1.5, 0.7, n_recon),
        'ack_count': np.random.normal(6.0, 2.0, n_recon),
        'arp_count': np.random.normal(0.5, 0.5, n_recon),
        'icmp_count': np.random.normal(2.0, 1.0, n_recon),
        'dns_query_count': np.random.normal(1.0, 0.5, n_recon),
        'unique_dst_ports': np.random.normal(15.0, 5.0, n_recon).astype(int),
        'unique_src_ports': np.random.normal(1.2, 0.4, n_recon).astype(int),
        'interval_variance': np.random.gamma(4, 2, n_recon),
        'attack_type': ['Recon'] * n_recon
    })
    all_data.append(recon_data)
    
    # Mirai botnet traffic
    n_mirai = attack_counts['Mirai']
    mirai_data = pd.DataFrame({
        'packet_count': np.random.normal(100, 25, n_mirai).astype(int),
        'flow_duration': np.random.normal(320, 70, n_mirai),
        'avg_packet_size': np.random.normal(90, 20, n_mirai),
        'flow_packets_per_sec': np.random.normal(7.0, 1.8, n_mirai),
        'flow_bytes_per_sec': np.random.normal(630, 120, n_mirai),
        'rst_count': np.random.normal(5.0, 2.0, n_mirai),
        'urg_count': np.random.normal(4.0, 1.5, n_mirai),
        'fin_count': np.random.normal(0.3, 0.5, n_mirai),
        'syn_count': np.random.normal(10.0, 3.0, n_mirai),
        'psh_count': np.random.normal(8.0, 2.5, n_mirai),
        'ack_count': np.random.normal(3.0, 1.0, n_mirai),
        'arp_count': np.random.normal(0.1, 0.3, n_mirai),
        'icmp_count': np.random.normal(4.0, 1.0, n_mirai),
        'dns_query_count': np.random.normal(1.5, 0.8, n_mirai),
        'unique_dst_ports': np.random.normal(2.0, 0.5, n_mirai).astype(int),
        'unique_src_ports': np.random.normal(5.0, 1.5, n_mirai).astype(int),
        'interval_variance': np.random.gamma(1, 0.3, n_mirai),
        'attack_type': ['Mirai'] * n_mirai
    })
    all_data.append(mirai_data)
    
    # Brute Force attack traffic
    n_brute = attack_counts['Brute_Force']
    brute_data = pd.DataFrame({
        'packet_count': np.random.normal(60, 15, n_brute).astype(int),
        'flow_duration': np.random.normal(380, 85, n_brute),
        'avg_packet_size': np.random.normal(150, 30, n_brute),
        'flow_packets_per_sec': np.random.normal(3.0, 1.0, n_brute),
        'flow_bytes_per_sec': np.random.normal(450, 100, n_brute),
        'rst_count': np.random.normal(2.0, 1.0, n_brute),
        'urg_count': np.random.normal(0.5, 0.5, n_brute),
        'fin_count': np.random.normal(2.0, 0.8, n_brute),
        'syn_count': np.random.normal(4.0, 1.5, n_brute),
        'psh_count': np.random.normal(5.0, 1.5, n_brute),
        'ack_count': np.random.normal(12.0, 3.0, n_brute),
        'arp_count': np.random.normal(0.2, 0.4, n_brute),
        'icmp_count': np.random.normal(0.3, 0.5, n_brute),
        'dns_query_count': np.random.normal(0.5, 0.5, n_brute),
        'unique_dst_ports': np.random.normal(1.2, 0.4, n_brute).astype(int),
        'unique_src_ports': np.random.normal(3.0, 1.0, n_brute).astype(int),
        'interval_variance': np.random.gamma(2, 1, n_brute),
        'attack_type': ['Brute_Force'] * n_brute
    })
    all_data.append(brute_data)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Add some common features for all types
    n_total = len(combined_df)
    combined_df['protocol'] = np.random.choice(['TCP', 'UDP', 'ICMP', 'HTTP'], size=n_total, 
                                             p=[0.6, 0.25, 0.1, 0.05])
    
    # Add some shared features (similar to what we see in real datasets)
    combined_df['Number'] = np.random.normal(50, 15, n_total)
    combined_df['Weight'] = np.random.normal(20, 5, n_total) 
    combined_df['IAT'] = np.random.normal(100, 30, n_total)
    combined_df['Duration'] = np.random.exponential(100, n_total)
    combined_df['Variance'] = np.random.gamma(2, 2, n_total)
    combined_df['Radius'] = np.random.uniform(1, 10, n_total)
    combined_df['Header_Length'] = np.random.normal(40, 10, n_total).astype(int)
    
    # One-hot encode protocol
    combined_df = pd.get_dummies(combined_df, columns=['protocol'])
    
    # Shuffle data
    return combined_df.sample(frac=1).reset_index(drop=True)

def live_detection_demo(model, scaler, feature_names=None, n_samples=100, multi_class=False):
    print(f"\nRunning live detection demo with {n_samples} simulated network flows...")
    
    # 生成模拟流量
    if multi_class:
        data = simulate_multi_attack_traffic(n_samples=n_samples)
        y_true_col = 'attack_type'
        
        # 创建标签到数字的映射（这是新增的）
        attack_labels = np.unique(data[y_true_col])
        attack_to_num = {label: idx for idx, label in enumerate(attack_labels)}
        # 将字符串标签转换为数字
        data['attack_type_num'] = data[y_true_col].map(attack_to_num)
        y_true_col_num = 'attack_type_num'
    else:
        data = simulate_ddos_traffic(n_samples=n_samples, attack_ratio=0.3)
        y_true_col = 'is_ddos'
        y_true_col_num = y_true_col  # 二分类已经是数字标签
    
    print(f"Generated {len(data)} simulated network flows")
    
    # 提取特征和标签
    if feature_names is not None:
        # 确保所有需要的特征都存在
        missing_features = [f for f in feature_names if f not in data.columns]
        if missing_features:
            print(f"Warning: {len(missing_features)} features expected by the model are missing in simulated data.")
            # 添加缺失特征，值为0
            for feature in missing_features:
                data[feature] = 0
        
        X = data[feature_names]
    else:
        # 排除所有标签列
        exclude_cols = [y_true_col, y_true_col_num] if multi_class else [y_true_col]
        X = data.drop(exclude_cols, axis=1)
    
    y_true = data[y_true_col_num]  # 使用数字标签
    
    # 如果提供了scaler，缩放特征
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values
    
    # 进行预测
    start_time = time.time()
    y_pred = model.predict(X_scaled)
    
    # 获取预测概率（如果可用）
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_scaled)
        if not multi_class:
            # 对于二分类，获取正类的概率
            y_proba = y_proba[:, 1]
    else:
        y_proba = y_pred
    
    inference_time = time.time() - start_time
    
    # 将预测添加到数据中
    results = data.copy()
    results['predicted_attack'] = y_pred
    
    if multi_class:
        # 对于多类分类，获取数字预测对应的类名
        num_to_attack = {idx: label for label, idx in attack_to_num.items()}
        results['predicted_attack_name'] = results['predicted_attack'].map(num_to_attack)
        
        # 计算是否正确分类
        results['correctly_classified'] = (results[y_true_col_num] == results['predicted_attack'])
        
        # 预测概率 - 存储最高概率
        if hasattr(model, 'predict_proba'):
            results['attack_probability'] = y_proba.max(axis=1)
    else:
        # 对于二分类
        results['attack_probability'] = y_proba
        results['correctly_classified'] = (y_true == y_pred)
    
    # 计算指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nLive Detection Results:")
    print(f"Processed {n_samples} network flows in {inference_time:.4f} seconds")
    print(f"Detection rate: {inference_time / n_samples:.6f} seconds per flow")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return results

# def live_detection_demo(model, scaler, feature_names=None, n_samples=100, multi_class=False):
    """
    Run a live detection demo using simulated network traffic.
    
    Parameters:
    -----------
    model : trained model
        Pre-trained attack detection model
    scaler : StandardScaler
        Fitted scaler for feature normalization
    feature_names : list or None
        Names of features expected by the model
    n_samples : int
        Number of network traffic samples to simulate
    multi_class : bool
        Whether to perform multi-class detection
        
    Returns:
    --------
    DataFrame with detection results
    """
    print(f"\nRunning live detection demo with {n_samples} simulated network flows...")
    
    # Generate simulated traffic
    if multi_class:
        data = simulate_multi_attack_traffic(n_samples=n_samples)
        y_true_col = 'attack_type'
    else:
        data = simulate_ddos_traffic(n_samples=n_samples, attack_ratio=0.3)
        y_true_col = 'is_ddos'
    
    print(f"Generated {len(data)} simulated network flows")
    
    # Extract features and labels
    if feature_names is not None:
        # Ensure all required features are present
        missing_features = [f for f in feature_names if f not in data.columns]
        if missing_features:
            print(f"Warning: {len(missing_features)} features expected by the model are missing in simulated data.")
            # Add missing features with zeros
            for feature in missing_features:
                data[feature] = 0
        
        X = data[feature_names]
    else:
        X = data.drop(y_true_col, axis=1)
    
    y_true = data[y_true_col]
    
    # Scale features if a scaler is provided
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values
    
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_scaled)
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_scaled)
        if not multi_class:
            # For binary classification, get probability of positive class
            y_proba = y_proba[:, 1]
    else:
        y_proba = y_pred
    
    inference_time = time.time() - start_time
    
    # Add predictions to data
    results = data.copy()
    results['predicted_attack'] = y_pred
    
    if multi_class:
        # For multi-class, get the predicted class name
        attack_types = np.unique(results['attack_type'])
        results['correctly_classified'] = (results['attack_type'] == results['predicted_attack'])
        
        # Prediction probabilities - store highest probability
        if hasattr(model, 'predict_proba'):
            results['attack_probability'] = y_proba.max(axis=1)
    else:
        # For binary classification
        results['attack_probability'] = y_proba
        results['correctly_classified'] = (y_true == y_pred)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    if multi_class:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
    else:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    
    print(f"\nLive Detection Results:")
    print(f"Processed {n_samples} network flows in {inference_time:.4f} seconds")
    print(f"Detection rate: {inference_time / n_samples:.6f} seconds per flow")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return results