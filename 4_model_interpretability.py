import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import export_text, plot_tree
import os

print("开始模型解释性分析...")

# 加载数据和模型
X_test = pd.read_csv('output/X_test.csv')
y_test = pd.read_csv('output/y_test.csv').values.ravel()
dt_model = joblib.load('output/models/decision_tree.joblib')
rf_model = joblib.load('output/models/optimized_random_forest.joblib')

# 确保输出目录存在
os.makedirs('output/interpretability', exist_ok=True)

# 1. 决策树规则可视化 (用文本表示)
print("生成决策树规则...")
tree_rules = export_text(dt_model, feature_names=list(X_test.columns))
with open('output/interpretability/decision_tree_rules.txt', 'w') as f:
    f.write(tree_rules)

# 2. 决策树可视化 (仅限于较小的树)
if dt_model.max_depth is not None and dt_model.max_depth <= 5:
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, 
              feature_names=list(X_test.columns),
              class_names=['Normal', 'DDoS'],
              filled=True,
              rounded=True,
              fontsize=10)
    plt.savefig('output/interpretability/decision_tree_visualization.png', dpi=300)
    print("已保存决策树可视化图")

# 3. 随机森林特征重要性分析
feature_importance = pd.Series(rf_model.feature_importances_, index=X_test.columns)
feature_importance = feature_importance.sort_values(ascending=False)

# 保存特征重要性为CSV
feature_importance.to_csv('output/interpretability/feature_importance.csv')

# 绘制所有特征的重要性
plt.figure(figsize=(10, len(feature_importance) * 0.3))  # 动态调整图像高度
plt.barh(feature_importance.index, feature_importance.values)
plt.title('所有特征重要性排序')
plt.tight_layout()
plt.savefig('output/interpretability/all_features_importance.png', dpi=300)
print("已保存所有特征重要性图")

# 4. 特征重要性组分析
# 根据特征名称进行分组
feature_groups = {}
for feature in X_test.columns:
    # 示例：将特征按前缀分组
    # 实际需要根据你的特征命名规则调整
    parts = feature.split('_')
    if len(parts) > 1:
        prefix = parts[0]
    else:
        prefix = 'other'
    
    if prefix not in feature_groups:
        feature_groups[prefix] = 0
    feature_groups[prefix] += feature_importance.get(feature, 0)

# 绘制特征组重要性
group_importance = pd.Series(feature_groups).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
group_importance.plot(kind='bar')
plt.title('特征组重要性')
plt.ylabel('累计重要性')
plt.tight_layout()
plt.savefig('output/interpretability/feature_group_importance.png')
print("已保存特征组重要性图")

# 5. 样本分析：检查几个预测正确和错误的样本
print("\n样本分析...")
# 获取模型预测
y_pred = rf_model.predict(X_test)
# 找出预测正确和错误的样本
correct_indices = np.where(y_pred == y_test)[0]
incorrect_indices = np.where(y_pred != y_test)[0]

# 生成样本分析报告
with open('output/interpretability/sample_analysis.txt', 'w') as f:
    f.write("样本分析报告\n")
    f.write("="*50 + "\n\n")
    
    # 分析预测错误的样本
    f.write(f"预测错误的样本数量: {len(incorrect_indices)}\n")
    if len(incorrect_indices) > 0:
        f.write("\n错误预测样本分析:\n")
        # 选择最多5个错误样本进行分析
        sample_count = min(5, len(incorrect_indices))
        for i in range(sample_count):
            idx = incorrect_indices[i]
            f.write(f"\n样本 #{idx}:\n")
            f.write(f"真实标签: {'DDoS' if y_test[idx] == 1 else 'Normal'}\n")
            f.write(f"预测标签: {'DDoS' if y_pred[idx] == 1 else 'Normal'}\n")
            
            # 获取样本特征值
            sample = X_test.iloc[idx]
            
            # 获取最重要的10个特征及其值
            top_features = feature_importance.nlargest(10).index
            f.write("\n最重要特征的值:\n")
            for feature in top_features:
                f.write(f"{feature}: {sample[feature]:.4f}\n")

    # 分析一些预测正确的样本
    f.write("\n\n" + "="*50 + "\n\n")
    f.write(f"预测正确的样本数量: {len(correct_indices)}\n")
    if len(correct_indices) > 0:
        f.write("\n正确预测样本分析:\n")
        # 选择DDoS和Normal各2个样本
        ddos_correct = [idx for idx in correct_indices if y_test[idx] == 1][:2]
        normal_correct = [idx for idx in correct_indices if y_test[idx] == 0][:2]
        
        for category, indices in [("DDoS", ddos_correct), ("Normal", normal_correct)]:
            f.write(f"\n{category}类别正确预测样本:\n")
            for idx in indices:
                f.write(f"\n样本 #{idx}:\n")
                
                # 获取样本特征值
                sample = X_test.iloc[idx]
                
                # 获取最重要的10个特征及其值
                top_features = feature_importance.nlargest(10).index
                f.write("最重要特征的值:\n")
                for feature in top_features:
                    f.write(f"{feature}: {sample[feature]:.4f}\n")
                f.write("\n")

print("模型解释性分析完成!")