import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Load NSL-KDD dataset (ARFF format)
data_train, meta_train = arff.loadarff('KDDTrain+.arff')
data_test, meta_test = arff.loadarff('KDDTest+.arff')

df_train = pd.DataFrame(data_train)
df_test = pd.DataFrame(data_test)

# Convert bytes type categorical values to strings
for col in df_train.select_dtypes(['object']).columns:
    df_train[col] = df_train[col].str.decode('utf-8')
    df_test[col] = df_test[col].str.decode('utf-8')

# Convert binary features to integers
binary_cols = ['land', 'logged_in', 'is_host_login', 'is_guest_login']
for col in binary_cols:
    if col in df_train.columns:
        df_train[col] = df_train[col].astype(int)
        df_test[col] = df_test[col].astype(int)

# Preprocess 'class' column: strip whitespace and convert to lowercase
df_train['class'] = df_train['class'].str.strip().str.lower()
df_test['class'] = df_test['class'].str.strip().str.lower()

# Optional: Print all attack types in the training set to check for any unmapped values
print("Attack types in training set:", df_train['class'].unique())

# Define attack type mapping (ensure all attack types are mapped; default unmapped ones to 'Other')
attack_mapping = {
    'normal': 'Normal',
    # DDoS-related attacks (DoS category)
    'neptune': 'DDoS', 'smurf': 'DDoS', 'back': 'DDoS', 'teardrop': 'DDoS', 'pod': 'DDoS', 'land': 'DDoS',
    # MITM-related attacks
    'spy': 'MITM', 'multihop': 'MITM', 'phf': 'MITM',
    # Other attack types mapped to Other
    'ipsweep': 'Other', 'nmap': 'Other', 'portsweep': 'Other', 'satan': 'Other',
    'ftp_write': 'Other', 'guess_passwd': 'Other', 'imap': 'Other', 'warezclient': 'Other', 
    'warezmaster': 'Other', 'buffer_overflow': 'Other', 'loadmodule': 'Other', 'perl': 'Other', 'rootkit': 'Other'
}

# Map attack types using the defined mapping; default to 'Other' if not found
df_train['attack_type'] = df_train['class'].map(lambda x: attack_mapping.get(x, 'Other'))
df_test['attack_type'] = df_test['class'].map(lambda x: attack_mapping.get(x, 'Other'))

# Check for unmapped attack types (the count should be 0)
assert df_train['attack_type'].isnull().sum() == 0, "There are unmapped attack types in the training set"
assert df_test['attack_type'].isnull().sum() == 0, "There are unmapped attack types in the test set"

# Split features and labels
X_train = df_train.drop(['class', 'attack_type'], axis=1)
y_train = df_train['attack_type']
X_test = df_test.drop(['class', 'attack_type'], axis=1)
y_test = df_test['attack_type']

# Data preprocessing: One-hot encode categorical features
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
if cat_cols:
    X_train_cat = ohe.fit_transform(X_train[cat_cols])
    X_test_cat = ohe.transform(X_test[cat_cols])
else:
    X_train_cat = np.array([[]])
    X_test_cat = np.array([[]])

# Data preprocessing: Standardize numerical features
num_cols = X_train.select_dtypes(exclude='object').columns
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_cols])
X_test_num = scaler.transform(X_test[num_cols])

# Combine preprocessed features
if X_train_cat.size > 0:
    X_train_processed = np.hstack([X_train_num, X_train_cat])
    X_test_processed = np.hstack([X_test_num, X_test_cat])
else:
    X_train_processed = X_train_num
    X_test_processed = X_test_num

# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_processed, y_train)
y_pred_dt = dt_model.predict(X_test_processed)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_processed, y_train)
y_pred_rf = rf_model.predict(X_test_processed)

# Calculate performance metrics
acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt, average='weighted')
rec_dt = recall_score(y_test, y_pred_dt, average='weighted')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, average='weighted')
rec_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

# Print performance metrics
print("=== Performance Metrics Comparison ===")
print(f"Decision Tree -> Accuracy: {acc_dt:.4f}, Precision: {prec_dt:.4f}, Recall: {rec_dt:.4f}, F1: {f1_dt:.4f}")
print(f"Random Forest -> Accuracy: {acc_rf:.4f}, Precision: {prec_rf:.4f}, Recall: {rec_rf:.4f}, F1: {f1_rf:.4f}")

# Print detailed classification reports
print("\n=== Classification Report: Decision Tree ===")
print(classification_report(y_test, y_pred_dt))

print("=== Classification Report: Random Forest ===")
print(classification_report(y_test, y_pred_rf))

# Visualize performance metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
dt_scores = [acc_dt, prec_dt, rec_dt, f1_dt]
rf_scores = [acc_rf, prec_rf, rec_rf, f1_rf]

x = np.arange(len(metrics))
width = 0.35
plt.figure(figsize=(8, 6))
plt.bar(x - width/2, dt_scores, width, label='Decision Tree')
plt.bar(x + width/2, rf_scores, width, label='Random Forest')
plt.xticks(x, metrics)
plt.ylabel('Score')
plt.title('Model Performance on NSL-KDD (Multi-Class: Normal, DDoS, MITM, Other)')
plt.ylim(0, 1.0)
plt.legend()
plt.show()