"""
Data loading and preprocessing module for SmartGuard
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataProcessor:
    """
    Class for loading and preprocessing network traffic data for attack detection.
    """

    def __init__(self, data_path):
        """
        Initialize the data processor.
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset CSV file.
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.attack_types = None

    def process_data(self, test_size=0.25, multi_class=False):
        """
        Load and preprocess the network traffic dataset.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing.
        multi_class : bool
            Whether to perform multi-class classification instead of binary.
            
        Returns:
        --------
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
            Preprocessed training and testing data.
        """
        print(f"Loading dataset from {self.data_path}...")
        
        try:
            # Load dataset
            data = pd.read_csv(self.data_path)
            print(f"Dataset loaded with shape: {data.shape}")
            
            # Display basic information about the dataset
            print("\nDataset Information:")
            print(f"Total samples: {len(data)}")
            
            # Process data
            processed_data = self._process_dataset(data, multi_class)
            if processed_data is None:
                raise ValueError("Data processing failed.")
            
            # Split data
            if multi_class:
                # For multi-class classification
                X = processed_data.drop(['attack_type'], axis=1)
                y = processed_data['attack_type']
                
                # Encode labels
                y_encoded = self.label_encoder.fit_transform(y)
                self.attack_types = {i: label for i, label in enumerate(self.label_encoder.classes_)}
                
                print("\nAttack Types:")
                for idx, attack_type in self.attack_types.items():
                    print(f"{idx}: {attack_type}")
            else:
                # For binary classification (DDoS or not)
                X = processed_data.drop(['is_ddos'], axis=1)
                y_encoded = processed_data['is_ddos']
            
            self.feature_names = X.columns.tolist()
            
            print("\nSplitting data into training and testing sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            
            print(f"Training set size: {X_train.shape[0]} samples")
            print(f"Testing set size: {X_test.shape[0]} samples")
            
            # Scale features
            print("Scaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test, self.feature_names
            
        except Exception as e:
            print(f"Error loading or preprocessing data: {e}")
            raise

    def _process_dataset(self, data, multi_class=False):
        """
        Process the dataset for attack detection.
        
        Parameters:
        -----------
        data : DataFrame
            Raw dataset.
        multi_class : bool
            Whether to perform multi-class classification.
            
        Returns:
        --------
        DataFrame
            Processed dataset.
        """
        # Identify the label column
        label_column = self._identify_label_column(data)
        if label_column is None:
            print("Could not identify label column. Please check your dataset.")
            return None
        
        if multi_class:
            # For multi-class classification, categorize attacks
            data = self._categorize_attacks(data, label_column)
            
            # 过滤掉样本数量不足 2 的类别
            class_counts = data['attack_type'].value_counts()
            low_count_classes = class_counts[class_counts < 2].index.tolist()
            if low_count_classes:
                print(f"Removing classes with too few samples: {low_count_classes}")
                data = data[~data['attack_type'].isin(low_count_classes)]
        else:
            # For binary classification, create binary labels
            data = self._create_binary_labels(data, label_column)
        
        # Display class distribution
        if multi_class:
            print("\nClass Distribution (Attack Types):")
            print(data['attack_type'].value_counts())
        else:
            print("\nClass Distribution:")
            print(data['is_ddos'].value_counts())
            print(f"DDoS attack percentage: {data['is_ddos'].mean() * 100:.2f}%")
        
        # Select relevant features
        numeric_features, categorical_features = self._identify_features(data, label_column)
        
        # Handle missing values
        data = self._handle_missing_values(data, numeric_features)
        
        # One-hot encode categorical features (if any)
        if categorical_features:
            data = pd.get_dummies(data, columns=categorical_features)
            print(f"Data shape after one-hot encoding: {data.shape}")
        
        # Drop the original label column if it's different from 'is_ddos' or 'attack_type'
        if multi_class and label_column != 'attack_type':
            data = data.drop(label_column, axis=1)
        elif not multi_class and label_column != 'is_ddos':
            data = data.drop(label_column, axis=1)
        
        return data

    def _identify_label_column(self, data):
        """
        Identify the label column in the dataset.
        
        Parameters:
        -----------
        data : DataFrame
            
        Returns:
        --------
        str or None
            Name of the label column.
        """
        label_column = None
        for col in ['Label', 'label', 'Attack', 'attack', 'class', 'Class']:
            if col in data.columns:
                label_column = col
                print(f"Identified label column: {col}")
                break
        
        if label_column is None:
            print("Warning: Could not automatically detect label column.")
            print("Available columns:", data.columns.tolist())
            label_column = input("Please enter the name of the label column: ")
        
        return label_column

    def _create_binary_labels(self, data, label_column):
        """
        Create binary labels for DDoS detection.
        
        Parameters:
        -----------
        data : DataFrame
            Dataset.
        label_column : str
            Name of the label column.
                
        Returns:
        --------
        DataFrame
            Dataset with binary labels.
        """
        print(f"Checking for DDoS in labels. First 10 unique values: {data[label_column].unique()[:10]}")
        
        if data[label_column].dtype == 'object':
            has_ddos = False
            for value in data[label_column].unique():
                if isinstance(value, str) and "DDoS" in value:
                    has_ddos = True
                    break
                    
            if has_ddos:
                print("DDoS label found in dataset using manual check")
                data['is_ddos'] = data[label_column].apply(lambda x: 1 if isinstance(x, str) and x.startswith('DDoS-') else 0)
                ddos_count = data['is_ddos'].sum()
                print(f"Automatically detected {ddos_count} samples as DDoS attacks")
                if ddos_count > 0:
                    print(f"DDoS attack types found: {data[data['is_ddos'] == 1][label_column].unique().tolist()}")
            else:
                print("Warning: 'DDoS' not found in labels. Please check your dataset.")
                print("Available unique labels:")
                unique_labels = data[label_column].unique()
                print(unique_labels[:min(20, len(unique_labels))])
                ddos_types = input("Enter comma-separated attack types to consider as DDoS: ").split(',')
                data['is_ddos'] = data[label_column].isin(ddos_types).astype(int)
        else:
            print("Label column appears to be numeric. Assuming binary classification.")
            data['is_ddos'] = data[label_column]
        
        return data

    def _categorize_attacks(self, data, label_column):
        """
        Categorize different types of attacks for multi-class classification.
        
        Parameters:
        -----------
        data : DataFrame
            Dataset.
        label_column : str
            Name of the label column.
            
        Returns:
        --------
        DataFrame
            Dataset with categorized attack types.
        """
        # Display unique attack types in dataset
        unique_attacks = data[label_column].unique()
        print(f"\nFound {len(unique_attacks)} unique attack labels:")
        for i, attack in enumerate(unique_attacks, 1):
            print(f"{i}. {attack}")
        
        # Create a simplified attack_type column
        data['attack_type'] = data[label_column]
        
        print("\nDo you want to map attack types to higher-level categories? (y/n)")
        response = input()
        
        if response.lower() == 'y':
            print("Enter mapping in format 'Category:type1,type2,...' (one line per category):")
            print("Examples:")
            print("DDoS:DDoS-RSTFINFlood,DDoS-ICMP_Flood,DDoS-TCP_Flood")
            print("MITM:MITM-ArpSpoofing,DNS_Spoofing")
            print("BENIGN:BenignTraffic")
            print("Enter 'done' when finished.")
            
            mapping = {}
            while True:
                line = input()
                if line.lower() == 'done':
                    break
                try:
                    category, types = line.split(':')
                    attack_types = [t.strip() for t in types.split(',')]
                    for attack_type in attack_types:
                        mapping[attack_type] = category.strip()
                except ValueError:
                    print("Invalid format. Please use 'Category:type1,type2,...'")
            
            data['attack_type'] = data['attack_type'].map(mapping).fillna(data['attack_type'])
            print("\nAttack types after mapping:")
            print(data['attack_type'].value_counts())
        
        return data

    def _identify_features(self, data, label_column):
        """
        Identify numeric and categorical features in the dataset.
        
        Parameters:
        -----------
        data : DataFrame.
        label_column : str
        
        Returns:
        --------
        tuple: (numeric_features, categorical_features)
        """
        # 这里假设所有非标签列都是数值型特征
        numeric_features = [col for col in data.columns if col != label_column and data[col].dtype in [np.int64, np.float64]]
        categorical_features = []  # 如果需要处理类别特征，可在此扩展
        return numeric_features, categorical_features

    def _handle_missing_values(self, data, numeric_features):
        """
        Handle missing values in numeric features.
        """
        # 这里简单采用均值填充缺失值
        for col in numeric_features:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mean(), inplace=True)
        return data

    def get_attack_types(self):
        """
        Return the mapping dictionary of attack types (only for multi-class mode).
        """
        return self.attack_types

if __name__ == "__main__":
    # 示例：如何使用 DataProcessor
    dp = DataProcessor("data/CIC-IoT 2023/cic_subset.csv")
    X_train, X_test, y_train, y_test, features = dp.process_data(test_size=0.25, multi_class=True)
    print("预处理完成")