import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import os

print("启动 DOS 攻击检测模拟系统...")

# 加载预处理后的测试数据（假设你已用类似预处理流程处理 KDDTest+.txt）
X_test = pd.read_csv('output/X_val.csv')  # 此处使用验证集作为模拟测试集，可替换为正式测试集
y_test = pd.read_csv('output/y_val.csv').values.ravel()

# 加载优化后的模型（优先加载优化后模型，否则加载随机森林模型）
model_path = 'output/models/optimized_random_forest.joblib'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"已加载模型: {model_path}")
else:
    model_path = 'output/models/random_forest.joblib'
    model = joblib.load(model_path)
    print(f"已加载模型: {model_path}")

os.makedirs('output/simulation', exist_ok=True)

# 模拟检测参数
window_size = 100   # 每个时间窗口包含的数据条数
total_windows = 20  # 总共模拟 20 个时间窗口
detection_results = []

print(f"开始模拟检测，共 {total_windows} 个时间窗口，每个窗口 {window_size} 个数据包")

for window in range(total_windows):
    print(f"处理时间窗口 {window+1}/{total_windows}")
    
    start_idx = window * window_size
    end_idx = min(start_idx + window_size, len(X_test))
    
    if start_idx >= len(X_test):
        break
    
    window_data = X_test.iloc[start_idx:end_idx]
    true_labels = y_test[start_idx:end_idx]
    
    start_time = time.time()
    predictions = model.predict(window_data)
    detection_time = time.time() - start_time
    
    total_packets = len(predictions)
    dos_packets = np.sum(predictions == 1)
    normal_packets = total_packets - dos_packets
    
    accuracy = np.mean(predictions == true_labels)
    
    detection_results.append({
        'window': window + 1,
        'total_packets': total_packets,
        'dos_packets': dos_packets,
        'normal_packets': normal_packets,
        'detection_time': detection_time,
        'accuracy': accuracy
    })
    
    time.sleep(0.5)  # 模拟实时延时

results_df = pd.DataFrame(detection_results)
results_df.to_csv('output/simulation/detection_results.csv', index=False)
print("已保存检测结果")

# 绘制检测结果图
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.bar(results_df['window'], results_df['normal_packets'], label='Normal', color='green')
plt.bar(results_df['window'], results_df['dos_packets'], bottom=results_df['normal_packets'], label='DOS', color='red')
plt.xlabel('窗口')
plt.ylabel('数据包数量')
plt.title('每个窗口数据包分类')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(results_df['window'], results_df['accuracy'], marker='o')
plt.xlabel('窗口')
plt.ylabel('准确率')
plt.title('每个窗口准确率')

plt.subplot(2, 2, 3)
plt.plot(results_df['window'], results_df['detection_time'], marker='o', color='orange')
plt.xlabel('窗口')
plt.ylabel('检测时间 (秒)')
plt.title('每个窗口检测时间')

plt.tight_layout()
plt.savefig('output/simulation/detection_overview.png')
print("已保存检测结果图")
