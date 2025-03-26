import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the NSL-KDD dataset (TXT format, supports multi-class classification)
df_train = pd.read_csv('KDDTrain+.txt', header=None)
df_test = pd.read_csv('KDDTest+.txt', header=None)

# Add column names (NSL-KDD has 43 columns, with the last column as the difficulty level)
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'class',  # The 42nd column is the attack type
    'difficulty'  # The 43rd column is the difficulty level
]
df_train.columns = columns
df_test.columns = columns

# Verify the number of columns
print("Number of columns in training set:", len(df_train.columns))
print("Number of columns in test set:", len(df_test.columns))

# 2. Preprocess: Convert binary features to integers
binary_cols = ['land', 'logged_in', 'is_host_login', 'is_guest_login']
for col in binary_cols:
    df_train[col] = df_train[col].astype(int)
    df_test[col] = df_test[col].astype(int)

# 3. Preprocess the 'class' column: strip whitespace and convert to lowercase
df_train['class'] = df_train['class'].str.strip().str.lower()
df_test['class'] = df_test['class'].str.strip().str.lower()

# Check original labels
print("Unique original labels in training set:", df_train['class'].unique())
print("Unique original labels in test set:", df_test['class'].unique())
print("Label distribution in training set:\n", df_train['class'].value_counts())
print("Label distribution in test set:\n", df_test['class'].value_counts())

# 4. Define attack mapping, focusing on DDoS and MITM
attack_mapping = {
    'normal': 'Normal',
    # DDoS-related attacks
    'neptune': 'DDoS', 'smurf': 'DDoS', 'back': 'DDoS', 'teardrop': 'DDoS', 'pod': 'DDoS', 'land': 'DDoS',
    # MITM-related attacks
    'spy': 'MITM', 'multihop': 'MITM', 'phf': 'MITM'
}
df_train['attack_type'] = df_train['class'].map(lambda x: attack_mapping.get(x, 'Other'))
df_test['attack_type'] = df_test['class'].map(lambda x: attack_mapping.get(x, 'Other'))

# Check mapping results
print("\nMapped label distribution in training set:\n", df_train['attack_type'].value_counts())
print("Mapped label distribution in test set:\n", df_test['attack_type'].value_counts())

# 5. Split features and labels (exclude the 'difficulty' column)
X_train = df_train.drop(['class', 'attack_type', 'difficulty'], axis=1)
y_train = df_train['attack_type']
X_test = df_test.drop(['class', 'attack_type', 'difficulty'], axis=1)
y_test = df_test['attack_type']

# 6. Data Preprocessing
# One-hot encode categorical features
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat = ohe.fit_transform(X_train[cat_cols])
X_test_cat = ohe.transform(X_test[cat_cols])

# Standardize numerical features
num_cols = X_train.select_dtypes(exclude='object').columns
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_cols])
X_test_num = scaler.transform(X_test[num_cols])

# Combine the preprocessed features
X_train_processed = np.hstack([X_train_num, X_train_cat])
X_test_processed = np.hstack([X_test_num, X_test_cat])

# 7. Train models
# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_processed, y_train)
y_pred_dt = dt_model.predict(X_test_processed)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_processed, y_train)
y_pred_rf = rf_model.predict(X_test_processed)

# 8. Calculate performance metrics
acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt, average='weighted')
rec_dt = recall_score(y_test, y_pred_dt, average='weighted')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, average='weighted')
rec_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

# Output overall performance metrics
print("\n=== Performance Comparison ===")
print(f"Decision Tree -> Accuracy: {acc_dt:.4f}, Precision: {prec_dt:.4f}, Recall: {rec_dt:.4f}, F1: {f1_dt:.4f}")
print(f"Random Forest -> Accuracy: {acc_rf:.4f}, Precision: {prec_rf:.4f}, Recall: {rec_rf:.4f}, F1: {f1_rf:.4f}")

# Output detailed classification reports
print("\n=== Classification Report: Decision Tree ===")
print(classification_report(y_test, y_pred_dt))
print("\n=== Classification Report: Random Forest ===")
print(classification_report(y_test, y_pred_rf))

# 9. Visualize performance metrics by class
metrics = ['Precision', 'Recall', 'F1-Score']
dt_report = classification_report(y_test, y_pred_dt, output_dict=True)
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)

# Define labels (ensure these match your mapping: Normal, DDoS, MITM, Other)
labels = ['Normal', 'DDoS', 'MITM', 'Other']
dt_scores = [[dt_report[label][m.lower()] for m in metrics] for label in labels if label in dt_report]
rf_scores = [[rf_report[label][m.lower()] for m in metrics] for label in labels if label in rf_report]

x = np.arange(len(metrics))
width = 0.15
plt.figure(figsize=(12, 6))
for i, label in enumerate(labels):
    if label in dt_report:
        plt.bar(x + i * width, dt_scores[i], width, label=f'DT {label}')
    if label in rf_report:
        plt.bar(x + i * width + len(labels) * width, rf_scores[i], width, label=f'RF {label}')
plt.xticks(x + width * len(labels) / 2, metrics)
plt.ylabel('Score')
plt.title('Model Performance by Class (NSL-KDD)')
plt.ylim(0, 1.0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()