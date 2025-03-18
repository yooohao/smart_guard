import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. 加载数据
data = pd.read_csv('part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv')

# 2. 处理标签
data['binary_label'] = data['label'].apply(lambda x: 1 if 'DDoS' in x else 0)
print(data['binary_label'].value_counts())

# 3. 下采样以平衡数据 
normal_data = data[data['binary_label'] == 0]
ddos_data = data[data['binary_label'] == 1].sample(n=len(normal_data), random_state=42)
balanced_data = pd.concat([normal_data, ddos_data])
print(balanced_data['binary_label'].value_counts())

# 4. 选择特征
features = ['flow_duration', 'Header_Length', 'Protocol Type', 'Rate', 'Srate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'IAT', 'ack_count', 'syn_count', 'Tot size']
X = balanced_data[features]
y = balanced_data['binary_label']

# 5. 数据预处理
X = X.dropna()
y = y[X.index]
X = X[X['flow_duration'] >= 0]
y = y[X.index]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. 训练模型
normal_data = X_scaled[y == 0]
model = IsolationForest(contamination=0.4, random_state=42)  # 调整 contamination
model.fit(normal_data)

# 7. 测试模型
predictions = model.predict(X_scaled)
predictions = [1 if p == -1 else 0 for p in predictions]

# 8. 评估结果
print(classification_report(y, predictions, target_names=['Normal', 'DDoS']))

# 9. 可视化
cm = confusion_matrix(y, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'DDoS'])
disp.plot()
plt.show()