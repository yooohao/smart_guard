import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import time
import matplotlib.pyplot as plt
import os

print("开始模型优化...")

# 加载预处理后的数据
X_train = pd.read_csv('output/X_train.csv')
X_test = pd.read_csv('output/X_test.csv')
y_train = pd.read_csv('output/y_train.csv').values.ravel()
y_test = pd.read_csv('output/y_test.csv').values.ravel()

print(f"加载的数据集: X_train {X_train.shape}, y_train {y_train.shape}")

# 确保输出目录存在
os.makedirs('output/models', exist_ok=True)
os.makedirs('output/results', exist_ok=True)

# 为RandomForest定义参数空间
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# 创建RandomForest分类器
rf = RandomForestClassifier(random_state=42)

# 创建RandomizedSearchCV对象
print("开始随机搜索最佳参数...")
rf_random = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_dist, 
    n_iter=20,
    cv=3,
    verbose=2, 
    random_state=42, 
    n_jobs=-1,
    scoring='f1_macro'  # 使用适用于多分类的F1评分
)


# 拟合随机搜索模型
start_time = time.time()
rf_random.fit(X_train, y_train)
optimization_time = time.time() - start_time

# 输出最佳参数
print(f"\n随机搜索完成。耗时: {optimization_time:.2f}秒")
print(f"最佳参数: {rf_random.best_params_}")
print(f"最佳交叉验证分数: {rf_random.best_score_:.4f}")

# 使用最佳参数创建模型
best_rf = rf_random.best_estimator_

# 在测试集上评估最佳模型
start_time = time.time()
y_pred = best_rf.predict(X_test)
prediction_time = time.time() - start_time

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


print("\n优化后模型在测试集上的性能:")
print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1得分: {f1:.4f}")
print(f"预测时间: {prediction_time:.4f}秒")

# 保存优化后的模型
joblib.dump(best_rf, 'output/models/optimized_random_forest.joblib')
print("已保存优化后的随机森林模型")

# 保存优化结果
optimization_results = {
    'best_params': rf_random.best_params_,
    'best_cv_score': rf_random.best_score_,
    'test_accuracy': accuracy,
    'test_precision': precision,
    'test_recall': recall,
    'test_f1': f1,
    'optimization_time': optimization_time,
    'prediction_time': prediction_time
}

# 将结果保存为CSV
pd.DataFrame([optimization_results]).to_csv('output/results/optimization_results.csv', index=False)
print("已保存优化结果")

# 绘制特征重要性
feature_importance = pd.Series(best_rf.feature_importances_, index=X_train.columns)
plt.figure(figsize=(12, 8))
feature_importance.nlargest(15).plot(kind='barh')
plt.title('优化后的随机森林 - 前15个最重要特征')
plt.savefig('output/figures/optimized_rf_feature_importance.png')
print("已保存优化后的特征重要性图")

print("模型优化完成!")