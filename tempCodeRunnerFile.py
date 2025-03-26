import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载 NSL-KDD 数据集（TXT 格式，支持多分类）
df_train = pd.read_csv('KDDTrain+.txt', header=None)
df_test = pd.read_csv('KDDTest+.txt', header=None)

# 添加列名（NSL-KDD 有 43 列，最后一列为难度等级）
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
    'class',  # 第 42 列为攻击类型
    'difficulty'  # 第 43 列为难度等级
]
df_train.columns = columns
df_test.columns = columns

# 验证列数
print("训练集列数：", len(df_train.columns))
print("测试集列数：", len(df_test.columns))

# 2. 预处理：将二元特征转换为整数
binary_cols = ['land', 'logged_in', 'is_host_login', 'is_guest_login']
for col in binary_cols:
    df_train[col] = df_train[col].astype(int)
    df_test[col] = df_test[col].astype(int)

# 3. 对 class 列进行预处理：去除空格并统一转为小写
df_train['class'] = df_train['class'].str.strip().str.lower()
df_test['class'] = df_test['class'].str.strip().str.lower()

# 检查原始标签
print("训练集原始标签：", df_train['class'].unique())
print("测试集原始标签：", df_test['class'].unique())
print("训练集标签分布：", df_train['class'].value_counts())
print("测试集标签分布：", df_test['class'].value_counts())

# 4. 定义攻击类型映射，聚焦 DDoS 和 MITM
attack_mapping = {
    'normal': 'Normal',
    # DDoS 相关攻击
    'neptune': 'DDoS', 'smurf': 'DDoS', 'back': 'DDoS', 'teardrop': 'DDoS', 'pod': 'DDoS', 'land': 'DDoS',
    # MITM 相关攻击
    'spy': 'MITM', 'multihop': 'MITM', 'phf': 'MITM'
}
df_train['attack_type'] = df_train['class'].map(lambda x: attack_mapping.get(x, 'Other'))
df_test['attack_type'] = df_test['class'].map(lambda x: attack_mapping.get(x, 'Other'))

# 检查映射结果
print("\n训练集映射后标签分布：", df_train['attack_type'].value_counts())
print("测试集映射后标签分布：", df_test['attack_type'].value_counts())

# 5. 划分特征和标签（排除 difficulty 列）
X_train = df_train.drop(['class', 'attack_type', 'difficulty'], axis=1)
y_train = df_train['attack_type']
X_test = df_test.drop(['class', 'attack_type', 'difficulty'], axis=1)
y_test = df_test['attack_type']

# 6. 数据预处理
# 独热编码类别特征
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat = ohe.fit_transform(X_train[cat_cols])
X_test_cat = ohe.transform(X_test[cat_cols])

# 标准化数值特征
num_cols = X_train.select_dtypes(exclude='object').columns
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_cols])
X_test_num = scaler.transform(X_test[num_cols])

# 合并预处理后的特征
X_train_processed = np.hstack([X_train_num, X_train_cat])
X_test_processed = np.hstack([X_test_num, X_test_cat])

# 7. 训练模型
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_processed, y_train)
y_pred_dt = dt_model.predict(X_test_processed)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_processed, y_train)
y_pred_rf = rf_model.predict(X_test_processed)

# 8. 计算性能指标
acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt, average='weighted')
rec_dt = recall_score(y_test, y_pred_dt, average='weighted')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, average='weighted')
rec_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

# 输出总体性能指标
print("\n=== 性能指标对比 ===")
print(f"Decision Tree -> Accuracy: {acc_dt:.4f}, Precision: {prec_dt:.4f}, Recall: {rec_dt:.4f}, F1: {f1_dt:.4f}")
print(f"Random Forest -> Accuracy: {acc_rf:.4f}, Precision: {prec_rf:.4f}, Recall: {rec_rf:.4f}, F1: {f1_rf:.4f}")

# 输出详细分类报告
print("\n=== Classification Report: Decision Tree ===")
print(classification_report(y_test, y_pred_dt))
print("\n=== Classification Report: Random Forest ===")
print(classification_report(y_test, y_pred_rf))

# 9. 可视化性能指标（按类别）
metrics = ['Precision', 'Recall', 'F1-Score']
dt_report = classification_report(y_test, y_pred_dt, output_dict=True)
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)

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