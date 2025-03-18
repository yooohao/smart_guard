import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

print("启动DDoS攻击检测模拟系统...")

# 加载测试数据（用于模拟实时流量）
X_test = pd.read_csv('output/X_test.csv')
y_test = pd.read_csv('output/y_test.csv').values.ravel()

# 加载最佳模型
model_path = 'output/models/optimized_random_forest.joblib'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"已加载模型: {model_path}")
else:
    model_path = 'output/models/random_forest.joblib'
    model = joblib.load(model_path)
    print(f"已加载模型: {model_path}")

# 创建输出目录
os.makedirs('output/simulation', exist_ok=True)

# 模拟参数
window_size = 100  # 每个时间窗口的数据包数量
total_windows = 20  # 总时间窗口数
detection_results = []

print(f"开始模拟检测，共{total_windows}个时间窗口，每个窗口{window_size}个数据包")

# 进行模拟检测
for window in range(total_windows):
    print(f"处理时间窗口 {window+1}/{total_windows}")
    
    # 获取当前窗口的数据
    start_idx = window * window_size
    end_idx = min(start_idx + window_size, len(X_test))
    
    if start_idx >= len(X_test):
        break
    
    window_data = X_test.iloc[start_idx:end_idx]
    true_labels = y_test[start_idx:end_idx]
    
    # 测量检测时间
    start_time = time.time()
    predictions = model.predict(window_data)
    detection_time = time.time() - start_time
    
    # 计算结果
    total_packets = len(predictions)
    ddos_packets = np.sum(predictions == 1)
    normal_packets = total_packets - ddos_packets
    
    # 计算准确率
    accuracy = np.mean(predictions == true_labels)
    
    # 保存结果
    detection_results.append({
        'window': window + 1,
        'total_packets': total_packets,
        'ddos_packets': ddos_packets,
        'normal_packets': normal_packets,
        'detection_time': detection_time,
        'accuracy': accuracy
    })
    
    # 短暂停顿以模拟实时性
    time.sleep(0.5)

# 将结果转换为DataFrame
results_df = pd.DataFrame(detection_results)
results_df.to_csv('output/simulation/detection_results.csv', index=False)
print("已保存检测结果")

# 绘制检测结果
plt.figure(figsize=(15, 10))

# 1. 每个窗口的数据包分类
plt.subplot(2, 2, 1)
plt.bar(results_df['window'], results_df['normal_packets'], label='Normal', color='green')
plt.bar(results_df['window'], results_df['ddos_packets'], bottom=results_df['normal_packets'], label='DDoS', color='red')