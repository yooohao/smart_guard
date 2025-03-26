import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import export_text, plot_tree
import os

print("开始模型解释性分析...")

# 加载测试数据和模型（这里使用验证集 X_val 作为示例）
X_val = pd.read_csv('output/X_val.csv')
y_val = pd.read_csv('output/y_val.csv').values.ravel()
dt_model = joblib.load('output/models/decision_tree.joblib')
rf_model = joblib.load('output/models/optimized_random_forest.joblib')

os.makedirs('output/interpretability', exist_ok=True)

# 1. 决策树规则文本表示
print("生成决策树规则...")
tree_rules = export_text(dt_model, feature_names=list(X_val.columns))
with open('output/interpretability/decision_tree_rules.txt', 'w') as f:
    f.write(tree_rules)

# 2. 决策树可视化（仅当树较小时）
if dt_model.max_depth is not None and dt_model.max_depth <= 5:
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, 
              feature_names=list(X_val.columns),
              class_names=['Normal', 'DOS'],
              filled=True,
              rounded=True,
              fontsize=10)
    plt.savefig('output/interpretability/decision_tree_visualization.png', dpi=300)
    print("已保存决策树可视化图")

# 3. 随机森林特征重要性分析
feature_importance = pd.Series(rf_model.feature_importances_, index=X_val.columns)
feature_importance = feature_importance.sort_values(ascending=False)
feature_importance.to_csv('output/interpretability/feature_importance.csv')
plt.figure(figsize=(10, len(feature_importance)*0.3))
plt.barh(feature_importance.index, feature_importance.values)
plt.title('所有特征重要性排序')
plt.tight_layout()
plt.savefig('output/interpretability/all_features_importance.png', dpi=300)
print("已保存所有特征重要性图")

# 4. 特征组重要性分析（根据特征名前缀分组）
feature_groups = {}
for feature in X_val.columns:
    parts = feature.split('_')
    prefix = parts[0] if len(parts) > 1 else 'other'
    feature_groups[prefix] = feature_groups.get(prefix, 0) + feature_importance.get(feature, 0)
group_importance = pd.Series(feature_groups).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
group_importance.plot(kind='bar')
plt.title('特征组重要性')
plt.ylabel('累计重要性')
plt.tight_layout()
plt.savefig('output/interpretability/feature_group_importance.png')
print("已保存特征组重要性图")

# 5. 样本分析：生成预测正确与错误样本的报告
print("\n样本分析...")
y_pred = rf_model.predict(X_val)
correct_indices = np.where(y_pred == y_val)[0]
incorrect_indices = np.where(y_pred != y_val)[0]

with open('output/interpretability/sample_analysis.txt', 'w') as f:
    f.write("样本分析报告\n")
    f.write("="*50 + "\n\n")
    
    f.write(f"预测错误的样本数量: {len(incorrect_indices)}\n")
    if len(incorrect_indices) > 0:
        f.write("\n错误预测样本分析:\n")
        sample_count = min(5, len(incorrect_indices))
        for i in range(sample_count):
            idx = incorrect_indices[i]
            f.write(f"\n样本 #{idx}:\n")
            f.write(f"真实标签: {'DOS' if y_val[idx]==1 else 'Normal'}\n")
            f.write(f"预测标签: {'DOS' if y_pred[idx]==1 else 'Normal'}\n")
            sample = X_val.iloc[idx]
            top_features = feature_importance.nlargest(10).index
            f.write("\n最重要特征的值:\n")
            for feature in top_features:
                f.write(f"{feature}: {sample[feature]:.4f}\n")
    
    f.write("\n" + "="*50 + "\n\n")
    f.write(f"预测正确的样本数量: {len(correct_indices)}\n")
    if len(correct_indices) > 0:
        f.write("\n正确预测样本分析:\n")
        # 分别选择 DOS 和 Normal 各两个样本
        dos_correct = [idx for idx in correct_indices if y_val[idx]==1][:2]
        normal_correct = [idx for idx in correct_indices if y_val[idx]==0][:2]
        for category, indices in [("DOS", dos_correct), ("Normal", normal_correct)]:
            f.write(f"\n{category} 类别正确预测样本:\n")
            for idx in indices:
                f.write(f"\n样本 #{idx}:\n")
                sample = X_val.iloc[idx]
                top_features = feature_importance.nlargest(10).index
                f.write("最重要特征的值:\n")
                for feature in top_features:
                    f.write(f"{feature}: {sample[feature]:.4f}\n")
                f.write("\n")
print("模型解释性分析完成!")
