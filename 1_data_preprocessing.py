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

# 加载Bot-IoT数据集 (可能需要根据实际文件名调整)
# 注意: 根据实际情况修改文件名和路径
data_path = 'part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv'  # 修改为你的文件路径
print(f"加载数据集: {data_path}")

try:
    # 首先读取一行来获取列名
    headers = pd.read_csv(data_path, nrows=0).columns
    print(f"数据集列名: {headers}")
    
    # 然后加载完整数据集
    data = pd.read_csv(data_path)
    print(f"成功加载数据集，形状: {data.shape}")
except Exception as e:
    print(f"加载数据集时出错: {e}")
    exit(1)

# 检查数据集中的攻击类型
if 'attack' in data.columns:
    print("攻击类型分布:")
    print(data['attack'].value_counts())
    # 筛选出DDoS攻击和正常流量
    data = data[(data['attack'] == 'DDoS') | (data['attack'] == 'Normal')]
    # 创建二进制标签列: 1表示DDoS攻击，0表示正常
    data['label'] = (data['attack'] == 'DDoS').astype(int)
    
elif 'category' in data.columns:  # 根据实际数据集调整
    print("攻击类型分布:")
    print(data['category'].value_counts())
    # 筛选出DDoS攻击和正常流量
    data = data[(data['category'] == 'DDoS') | (data['category'] == 'Normal')]
    # 创建二进制标签列
    data['label'] = (data['category'] == 'DDoS').astype(int)

print(f"筛选后的数据集形状: {data.shape}")

# 查看数据集的前几行
print("\n数据集预览:")
print(data.head())

# 检查缺失值
print("\n检查缺失值:")
missing_data = data.isnull().sum()
print(missing_data[missing_data > 0])

# 处理缺失值
data = data.fillna(0)  # 用0填充缺失值，或者使用其他策略

# 识别特征和目标变量
print("\n识别特征和目标变量...")
if 'label' in data.columns:
    y = data['label']
    # 删除不需要的列，如ID、时间戳等
    # 注意：根据实际数据集调整要删除的列
    drop_cols = ['label']
    if 'attack' in data.columns:
        drop_cols.append('attack')
    if 'category' in data.columns:
        drop_cols.append('category')
    
    X = data.drop(drop_cols, axis=1)
else:
    print("错误: 未找到'label'列")
    exit(1)

print(f"特征集形状: {X.shape}")
print(f"标签集形状: {y.shape}")

# 处理分类特征
print("\n处理分类特征...")
categorical_columns = [col for col in X.columns if X[col].dtype == 'object']
if categorical_columns:
    print(f"发现的分类特征: {categorical_columns}")
    X = pd.get_dummies(X, columns=categorical_columns)
    print(f"One-hot编码后的特征集形状: {X.shape}")

# 标准化数值特征
print("\n标准化数值特征...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# 保存scaler供后续使用
joblib.dump(scaler, 'output/models/scaler.joblib')
print("已保存特征标准化器")

# 绘制数据分布图
print("\n绘制数据分布图...")
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=data)
plt.title('DDoS攻击 vs 正常流量分布')
plt.savefig('output/figures/data_distribution.png')
print("已保存数据分布图")

# 划分训练集和测试集
print("\n划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集形状: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"测试集形状: X_test {X_test.shape}, y_test {y_test.shape}")

# 保存处理后的数据
X_train.to_csv('output/X_train.csv', index=False)
X_test.to_csv('output/X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('output/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('output/y_test.csv', index=False)
print("已保存预处理后的数据")

# 保存列名供后续使用
with open('output/feature_names.txt', 'w') as f:
    for feature in X.columns:
        f.write(f"{feature}\n")
print("已保存特征名称")

print("数据预处理完成!")