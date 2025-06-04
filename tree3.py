import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, classification_report, confusion_matrix,
                           roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import warnings

# 设置中文显示和美化图形
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 负号显示
plt.style.use('ggplot')                       # 美化样式
warnings.filterwarnings('ignore')             # 忽略警告

# ====================== 通用分类任务函数 ======================
def run_classification_task(dataset_name, X, y, feature_names, class_names):
    print("\n" + "="*60)
    print(f"分类任务：{dataset_name}数据集 (ID3 vs C4.5 vs CART)")
    print("="*60)

    # 划分训练测试集（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # 定义模型参数网格
    param_grid = {
        'ID3': {
            'criterion': ['entropy'],
            'max_depth': [None, 3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'random_state': [42]
        },
        'C4.5': {
            'criterion': ['entropy'],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'random_state': [42]
        },
        'CART': {
            'criterion': ['gini'],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'random_state': [42]
        }
    }

    # 训练和评估模型
    results = {}
    for name in param_grid.keys():
        print(f"\n正在优化{name}算法...")
        grid_search = GridSearchCV(
            DecisionTreeClassifier(),
            param_grid[name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        # 存储结果
        results[name] = {
            'model': best_model,
            'params': grid_search.best_params_,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'f1': f1_score(y_test, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importances': best_model.feature_importances_
        }

        print(f"\n{name} 最佳参数: {grid_search.best_params_}")
        print(f"{name} 分类报告:")
        print(classification_report(y_test, y_pred, target_names=class_names))

    return results, X_train, X_test, y_train, y_test, feature_names, class_names

# ====================== 可视化函数（无Graphviz） ======================
def visualize_results(results, dataset_name, feature_names, class_names):
    # 1. 性能指标对比
    metrics_df = pd.DataFrame({
        '算法': list(results.keys()),
        '准确率': [results[name]['accuracy'] for name in results],
        '精确率': [results[name]['precision'] for name in results],
        '召回率': [results[name]['recall'] for name in results],
        'F1分数': [results[name]['f1'] for name in results]
    })

    plt.figure(figsize=(12, 6))
    metrics_df.set_index('算法').plot(kind='bar', rot=0)
    plt.title(f'{dataset_name}数据集 - 算法性能对比')
    plt.ylabel('分数')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_performance_comparison.png', dpi=300)
    plt.show()

    # 2. 特征重要性对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{dataset_name}数据集 - 特征重要性对比', fontsize=16)
    for i, (name, result) in enumerate(results.items()):
        importances = result['feature_importances']
        indices = np.argsort(importances)[-10:]  # 显示最重要的10个特征
        axes[i].barh(range(len(indices)), importances[indices])
        axes[i].set_yticks(range(len(indices)))
        axes[i].set_yticklabels([feature_names[j] for j in indices])
        axes[i].set_title(f'{name}算法')
        for j, v in enumerate(importances[indices]):
            axes[i].text(v + 0.01, j, f"{v:.3f}", color='black')
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_feature_importance.png', dpi=300)
    plt.show()

    # 3. 决策树可视化（使用plot_tree）
    for name in results:
        plt.figure(figsize=(20, 10))
        plot_tree(
            results[name]['model'],
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            impurity=False,
            proportion=True,
            max_depth=3  # 控制显示深度
        )
        plt.title(f'{dataset_name} - {name} 决策树')
        plt.savefig(f'{dataset_name}_decision_tree_{name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存决策树图片：{dataset_name}_decision_tree_{name}.png")

# ====================== 主程序 ======================
if __name__ == "__main__":
    # 葡萄酒数据集
    wine = load_wine()
    wine_results, X_train_wine, X_test_wine, y_train_wine, y_test_wine, wine_feature_names, wine_class_names = \
        run_classification_task("葡萄酒", wine.data, wine.target, wine.feature_names, wine.target_names)
    visualize_results(wine_results, "葡萄酒", wine_feature_names, wine_class_names)

    # 乳腺癌数据集
    cancer = load_breast_cancer()
    cancer_class_names = ['恶性', '良性']  # 重命名类别
    cancer_results, X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer, cancer_feature_names, _ = \
        run_classification_task("乳腺癌", cancer.data, cancer.target, cancer.feature_names, cancer_class_names)
    visualize_results(cancer_results, "乳腺癌", cancer_feature_names, cancer_class_names)

    # 综合对比分析
    print("\n" + "="*60)
    print("跨数据集算法综合对比")
    print("="*60)
    comparison_df = pd.concat([
        pd.DataFrame({
            '数据集': '葡萄酒',
            '算法': list(wine_results.keys()),
            '准确率': [wine_results[name]['accuracy'] for name in wine_results],
            'F1分数': [wine_results[name]['f1'] for name in wine_results]
        }),
        pd.DataFrame({
            '数据集': '乳腺癌',
            '算法': list(cancer_results.keys()),
            '准确率': [cancer_results[name]['accuracy'] for name in cancer_results],
            'F1分数': [cancer_results[name]['f1'] for name in cancer_results]
        })
    ])
    print(comparison_df.sort_values(['数据集', '准确率'], ascending=[True, False]))