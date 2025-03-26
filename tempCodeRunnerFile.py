import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# 加载 NSL-KDD 数据集（ARFF 格式）
data_train, meta_train = arff.loadarff('KDDTrain+.arff')
data_test, meta_test = arff.loadarff('KDDTest+.arff')

df_train = pd.DataFrame(data_train)
df_test = pd.DataFrame(data_test)

# 将 bytes 类型的类别值转换为字符串
for col in df_train.select_dtypes(['object']).columns:
    df_train[col] = df_train[col].str.decode('utf-8')
    df_test[col] = df_test[col].str.decode('utf-8')

# 将二元特征转换为整数
binary_cols = ['land', 'logged_in', 'is_host_login', 'is_guest_login']
for col in binary_cols:
    if col in df_train.columns:
        df_train[col] = df_train[col].astype(int)
        df_test[col] = df_test[col].astype(int)

# 对 class 列进行预处理：去除空格并统一转为小写
df_train['class'] = df_train['class'].str.strip().str.lower()
df_test['class'] = df_test['class'].str.strip().str.lower()

# 可选：输出训练集中所有攻击类型，便于检查是否有未映射的值
print("训练集中的攻击类型：", df_train['class'].unique())

# 定义攻击类型映射（确保所有出现的攻击类型都有映射，如果缺失的可以统一归为 'Other'）
attack_mapping = {
    'normal': 'Normal',
    # DDoS 相关攻击（DoS 类）
    'neptune': 'DDoS', 'smurf': 'DDoS', 'back': 'DDoS', 'teardrop': 'DDoS', 'pod': 'DDoS', 'land': 'DDoS',
    # MITM 相关攻击
    'spy': 'MITM', 'multihop': 'MITM', 'phf': 'MITM',
    # 其他攻击类型归为 Other
    'ipsweep': 'Other', 'nmap': 'Other', 'portsweep': 'Other', 'satan': 'Other',
    'ftp_write': 'Other', 'guess_passwd': 'Other', 'imap': 'Other', 'warezclient': 'Other', 
    'warezmaster': 'Other', 'buffer_overflow': 'Other', 'loadmodule': 'Other', 'perl': 'Other', 'rootkit': 'Other'
}

# 使用映射时，若字典中找不到对应的攻击类型，默认映射为 'Other'
df_train['attack_type'] = df_train['class'].map(lambda x: attack_mapping.get(x, 'Other'))
df_test['attack_type'] = df_test['class'].map(lambda x: attack_mapping.get(x, 'Other'))

# 检查是否存在未映射的攻击类型（理论上应该为 0）
assert df_train['attack_type'].isnull().sum() == 0, "训练集中存在未映射的攻击类型"
assert df_test['attack_type'].isnull().sum() == 0, "测试集中存在未映射的攻击类型"

# 划分特征和标签
X_train = df_train.drop(['class', 'attack_type'], axis=1)
y_train = df_train['attack_type']
X_test = df_test.drop(['class', 'attack_type'], axis=1)
y_test = df_test['attack_type']

# 数据预处理：独热编码类别特征
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
if cat_cols:
    X_train_cat = ohe.fit_transform(X_train[cat_cols])
    X_test_cat = ohe.transform(X_test[cat_cols])
else:
    X_train_cat = np.array([[]])
    X_test_cat = np.array([[]])

# 数据预处理：标准化数值特征
num_cols = X_train.select_dtypes(exclude='object').columns
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_cols])
X_test_num = scaler.transform(X_test[num_cols])

# 合并预处理后的特征
if X_train_cat.size > 0:
    X_train_processed = np.hstack([X_train_num, X_train_cat])
    X_test_processed = np.hstack([X_test_num, X_test_cat])
else:
    X_train_processed = X_train_num
    X_test_processed = X_test_num

# 训练决策树模型
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_processed, y_train)
y_pred_dt = dt_model.predict(X_test_processed)

# 训练随机森林模型
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_processed, y_train)
y_pred_rf = rf_model.predict(X_test_processed)

# 计算性能指标
acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt, average='weighted')
rec_dt = recall_score(y_test, y_pred_dt, average='weighted')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, average='weighted')
rec_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

# 输出性能指标
print("=== 性能指标对比 ===")
print(f"Decision Tree -> Accuracy: {acc_dt:.4f}, Precision: {prec_dt:.4f}, Recall: {rec_dt:.4f}, F1: {f1_dt:.4f}")
print(f"Random Forest -> Accuracy: {acc_rf:.4f}, Precision: {prec_rf:.4f}, Recall: {rec_rf:.4f}, F1: {f1_rf:.4f}")

# 输出详细分类报告
print("\n=== Classification Report: Decision Tree ===")
print(classification_report(y_test, y_pred_dt))

print("=== Classification Report: Random Forest ===")
print(classification_report(y_test, y_pred_rf))

# 可视化性能指标
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