import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time

print("开始模型训练...")

# 加载预处理后的数据（使用 1_data_preprocessing.py 生成的文件）
X_train = pd.read_csv('output/X_train.csv')
X_val = pd.read_csv('output/X_val.csv')
y_train = pd.read_csv('output/y_train.csv').values.ravel()
y_val = pd.read_csv('output/y_val.csv').values.ravel()

print(f"加载的数据集: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"加载的数据集: X_val {X_val.shape}, y_val {y_val.shape}")

# 确保输出目录存在
os.makedirs('output/models', exist_ok=True)
os.makedirs('output/figures', exist_ok=True)
os.makedirs('output/results', exist_ok=True)

# 定义评估函数
def evaluate_model(model, X_val, y_val, model_name):
    print(f"\n评估 {model_name} 模型...")
    start_time = time.time()
    y_pred = model.predict(X_val)
    prediction_time = time.time() - start_time

    # 计算评估指标（macro 平均）
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='macro')
    recall = recall_score(y_val, y_pred, average='macro')
    f1 = f1_score(y_val, y_pred, average='macro')
    
    print(f"{model_name} 准确率: {accuracy:.4f}")
    print(f"{model_name} 精确率: {precision:.4f}")
    print(f"{model_name} 召回率: {recall:.4f}")
    print(f"{model_name} F1得分: {f1:.4f}")
    print(f"{model_name} 预测时间: {prediction_time:.4f}秒")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_name} 混淆矩阵')
    plt.savefig(f'output/figures/{model_name}_confusion_matrix.png')
    
    # 保存详细分类报告
    with open(f'output/results/{model_name}_classification_report.txt', 'w') as f:
        f.write(classification_report(y_val, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'prediction_time': prediction_time
    }

# 训练决策树模型
print("\n训练决策树模型...")
dt = DecisionTreeClassifier(random_state=42, max_depth=10)
dt.fit(X_train, y_train)
dt_results = evaluate_model(dt, X_val, y_val, 'DecisionTree')

# 保存决策树模型
joblib.dump(dt, 'output/models/decision_tree.joblib')
print("已保存决策树模型")

# 绘制决策树特征重要性图
dt_feature_importance = pd.Series(dt.feature_importances_, index=X_train.columns)
plt.figure(figsize=(12, 8))
dt_feature_importance.nlargest(15).plot(kind='barh')
plt.title('决策树 - 前15个最重要特征')
plt.savefig('output/figures/dt_feature_importance.png')
print("已保存决策树特征重要性图")

# 训练随机森林模型
print("\n训练随机森林模型...")
rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)
rf_results = evaluate_model(rf, X_val, y_val, 'RandomForest')

# 保存随机森林模型
joblib.dump(rf, 'output/models/random_forest.joblib')
print("已保存随机森林模型")

# 绘制随机森林特征重要性图
rf_feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
plt.figure(figsize=(12, 8))
rf_feature_importance.nlargest(15).plot(kind='barh')
plt.title('随机森林 - 前15个最重要特征')
plt.savefig('output/figures/rf_feature_importance.png')
print("已保存随机森林特征重要性图")

# 模型性能对比
models = ['决策树', '随机森林']
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
results = [dt_results, rf_results]

plt.figure(figsize=(12, 8))
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    values = [result[metric] for result in results]
    bars = plt.bar(models, values)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
    plt.title(f'{metric} 对比')
    plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('output/figures/model_performance_comparison.png')
print("已保存模型性能对比图")

plt.figure(figsize=(8, 6))
times = [result['prediction_time'] for result in results]
bars = plt.bar(models, times)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}秒', ha='center', va='bottom')
plt.title('预测时间对比（秒）')
plt.savefig('output/figures/prediction_time_comparison.png')
print("已保存预测时间对比图")

comparison_df = pd.DataFrame([dt_results, rf_results], index=models)
comparison_df.to_csv('output/results/model_comparison.csv')
print("已保存模型比较结果")

print("模型训练和评估完成!")
