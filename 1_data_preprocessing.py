import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 创建输出目录
os.makedirs('output', exist_ok=True)
os.makedirs('output/figures', exist_ok=True)
os.makedirs('output/models', exist_ok=True)

print("开始数据预处理...")

# 设置 NSL-KDD 数据集的列名（共 43 列）
col_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
    "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

# 指定训练集文件路径（可替换为 KDDTrain+_20Percent.txt 以减少数据量）
train_data_path = 'KDDTrain+.txt'
print(f"加载训练集: {train_data_path}")

try:
    # 读取 NSL-KDD 训练集（没有表头，手动指定列名）
    data = pd.read_csv(train_data_path, header=None, names=col_names)
    print(f"成功加载训练集，形状: {data.shape}")
except Exception as e:
    print(f"加载训练集时出错: {e}")
    exit(1)

# 定义 DOS 攻击名称（NSL-KDD 中常见的 DOS 攻击）
dos_attacks = ["neptune", "smurf", "pod", "teardrop", "land", "back", "apache2", "udpstorm", "processtable", "worm"]

# 创建二分类标签：normal -> 0；若 label 在 dos_attacks 中则为 1；其他攻击暂时过滤掉
data['label_binary'] = data['label'].apply(lambda x: 0 if x == 'normal' else (1 if x in dos_attacks else np.nan))
# 过滤掉其他类型（如 Probe、R2L、U2R）
data = data.dropna(subset=['label_binary'])
data['label_binary'] = data['label_binary'].astype(int)
print(f"筛选后的数据集形状 (只保留 normal 和 DOS): {data.shape}")

# 预览数据
print("\n数据集预览:")
print(data.head())

# 检查缺失值
print("\n检查缺失值:")
missing_data = data.isnull().sum()
print(missing_data[missing_data > 0])

# 处理缺失值（此处简单用 0 填充）
data = data.fillna(0)

# 分离特征和目标变量
# 删除原始 label 与 difficulty 列
drop_cols = ['label', 'difficulty', 'label_binary']  # 目标变量单独处理
X = data.drop(drop_cols, axis=1)
y = data['label_binary']

print(f"特征集形状: {X.shape}")
print(f"标签集形状: {y.shape}")

# 处理分类特征（NSL-KDD 中通常有 protocol_type, service, flag）
categorical_columns = ['protocol_type', 'service', 'flag']
print("\n处理分类特征...")
print(f"发现的分类特征: {categorical_columns}")
X = pd.get_dummies(X, columns=categorical_columns)
print(f"One-hot编码后的特征集形状: {X.shape}")

# 标准化数值特征
print("\n标准化数值特征...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# 保存 scaler 以便后续处理测试集时使用
joblib.dump(scaler, 'output/models/scaler.joblib')
print("已保存特征标准化器")

# 绘制标签分布图
print("\n绘制数据分布图...")
plt.figure(figsize=(10, 6))
sns.countplot(x=y)
plt.xlabel('Label (0=Normal, 1=DOS)')
plt.title('NSL-KDD: Normal vs DOS 流量分布')
plt.savefig('output/figures/data_distribution.png')
print("已保存数据分布图")

# 划分训练集和验证集（这里从训练集中划分 80%/20% 进行内部验证）
print("\n划分训练集和验证集...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集形状: {X_train.shape}, y_train {y_train.shape}")
print(f"验证集形状: {X_val.shape}, y_val {y_val.shape}")

# 保存处理后的数据
X_train.to_csv('output/X_train.csv', index=False)
X_val.to_csv('output/X_val.csv', index=False)
pd.DataFrame(y_train).to_csv('output/y_train.csv', index=False)
pd.DataFrame(y_val).to_csv('output/y_val.csv', index=False)
print("已保存预处理后的数据")

# 保存特征名称供后续使用
with open('output/feature_names.txt', 'w') as f:
    for feature in X.columns:
        f.write(f"{feature}\n")
print("已保存特征名称")

print("数据预处理完成!")
