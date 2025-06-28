import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import cv2
import os
import requests
import time
import sys
import subprocess
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, \
    classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from PIL import Image
from ultralytics import YOLO
from sklearn.model_selection import cross_val_score

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# DeepSeek API配置
DEEPSEEK_API_URL = "https://chatapi.littlewheat.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# 设置页面配置
st.set_page_config(
    page_title="神经网络分析平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 检查并安装TensorFlow的函数
def check_and_install_tensorflow():
    try:
        import tensorflow as tf
        st.success("TensorFlow 已安装!")
        return True
    except ImportError:
        st.warning("TensorFlow 未安装，图像分类功能将受限")

        if st.button("尝试安装 TensorFlow"):
            with st.spinner("正在安装 TensorFlow，这可能需要几分钟..."):
                try:
                    # 使用pip安装tensorflow
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
                    st.success("TensorFlow 安装成功! 请重启应用")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"安装失败: {str(e)}")
                    st.info("""
                    **手动安装指南:**
                    1. 打开命令行/终端
                    2. 运行: `pip install tensorflow`
                    3. 重启应用
                    """)
        return False


# 加载乳腺癌数据集
@st.cache_data
def load_breast_cancer_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['diagnosis'] = df['target'].map({0: '恶性', 1: '良性'})
    return df, data


# 加载模型
@st.cache_resource
def load_model():
    return YOLO('models/yolov8n.pt')


# DeepSeek API调用函数
def analyze_with_deepseek(api_key, context, data_summary=None, prompt=None):
    """使用DeepSeek API进行数据分析"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 构建系统提示
    system_prompt = """
    你是一位专业的数据科学家，擅长分析医疗数据集特别是乳腺癌数据。
    请根据用户提供的数据分析报告和上下文信息，提供深入的洞察、建议和总结。
    回答应专业、简洁且具有可操作性，使用中文回答。
    """

    # 构建用户提示
    user_prompt = f"""
    ### 数据分析上下文:
    {context}

    ### 数据摘要:
    {data_summary if data_summary else '无附加数据摘要'}

    ### 用户问题:
    {prompt if prompt else '请分析数据并提供专业见解'}
    """

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1500
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"API调用失败: {str(e)}"


# 训练和评估模型
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 交叉验证
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted').mean()

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # ROC曲线数据
    fpr, tpr, roc_auc = None, None, None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

    return {
        'model_name': model_name,
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_accuracy': cv_accuracy,
        'cv_f1': cv_f1,
        'train_time': train_time,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }


# 模型比较页面
def model_comparison_page():
    st.title("乳腺癌数据集机器学习模型比较")
    st.write("""
    在此页面中，我们将使用多种机器学习算法对乳腺癌数据集进行建模，并比较它们的性能。
    """)

    # 加载数据
    df, data = load_breast_cancer_data()

    # 数据预处理
    X = df[data.feature_names]
    y = df['target']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.divider()
    st.subheader("数据集信息")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"总样本数: {len(df)}")
        st.write(f"特征数: {len(data.feature_names)}")
        st.write(f"训练集大小: {X_train_scaled.shape[0]} 样本")
        st.write(f"测试集大小: {X_test_scaled.shape[0]} 样本")

    with col2:
        diagnosis_counts = df['diagnosis'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(diagnosis_counts, labels=diagnosis_counts.index,
               autopct='%1.1f%%', startangle=90,
               colors=['#ff9999', '#66b3ff'])
        ax.set_title('诊断结果分布')
        st.pyplot(fig)

    st.divider()
    st.subheader("模型选择和训练")

    # 选择要训练的模型
    models_to_train = st.multiselect(
        "选择要训练的模型:",
        options=[
            "K近邻(KNN)", "决策树", "随机森林",
            "支持向量机(SVM)", "逻辑回归",
            "梯度提升树(GBDT)", "XGBoost",
            "感知器", "多层感知器(MLP)"
        ],
        default=[
            "K近邻(KNN)", "决策树", "随机森林",
            "支持向量机(SVM)", "逻辑回归"
        ]
    )

    # 高级选项
    with st.expander("高级选项"):
        use_feature_selection = st.checkbox("使用特征选择", value=False)
        hyperparameter_tuning = st.checkbox("超参数调优", value=False)

    # 特征选择
    if use_feature_selection:
        st.info("使用PCA进行特征降维（保留95%方差）")
        pca = PCA(n_components=0.95)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        st.write(f"降维后特征数: {X_train_scaled.shape[1]}")

    # 训练按钮
    if st.button("训练模型", type="primary", use_container_width=True):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 初始化模型
        models = []
        if "K近邻(KNN)" in models_to_train:
            models.append(("K近邻(KNN)", KNeighborsClassifier()))
        if "决策树" in models_to_train:
            models.append(("决策树", DecisionTreeClassifier(random_state=42)))
        if "随机森林" in models_to_train:
            models.append(("随机森林", RandomForestClassifier(random_state=42)))
        if "支持向量机(SVM)" in models_to_train:
            models.append(("支持向量机(SVM)", SVC(probability=True, random_state=42)))
        if "逻辑回归" in models_to_train:
            models.append(("逻辑回归", LogisticRegression(max_iter=1000, random_state=42)))
        if "梯度提升树(GBDT)" in models_to_train:
            models.append(("梯度提升树(GBDT)", GradientBoostingClassifier(random_state=42)))
        if "XGBoost" in models_to_train:
            models.append(("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)))
        if "感知器" in models_to_train:
            models.append(("感知器", Perceptron(random_state=42)))
        if "多层感知器(MLP)" in models_to_train:
            models.append(
                ("多层感知器(MLP)", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)))

        # 超参数调优
        if hyperparameter_tuning:
            st.info("正在进行超参数调优，可能需要更长时间...")
            tuned_models = []
            for name, model in models:
                if name == "K近邻(KNN)":
                    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
                elif name == "决策树":
                    param_grid = {'max_depth': [3, 5, 7, 9, None]}
                elif name == "随机森林":
                    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
                elif name == "支持向量机(SVM)":
                    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 'scale']}
                elif name == "逻辑回归":
                    param_grid = {'C': [0.1, 1, 10]}
                elif name == "梯度提升树(GBDT)":
                    param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
                elif name == "XGBoost":
                    param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
                elif name == "感知器":
                    param_grid = {'alpha': [0.0001, 0.001, 0.01], 'max_iter': [100, 500, 1000]}
                elif name == "多层感知器(MLP)":
                    param_grid = {'hidden_layer_sizes': [(64,), (64, 32)], 'alpha': [0.0001, 0.001]}

                grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                tuned_models.append((name, grid_search.best_estimator_))
                st.success(f"{name} 最佳参数: {grid_search.best_params_}")

            models = tuned_models

        # 训练和评估模型
        total_models = len(models)
        for i, (name, model) in enumerate(models):
            status_text.text(f"正在训练 {name} ({i + 1}/{total_models})...")
            result = train_and_evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
            results.append(result)
            progress_bar.progress((i + 1) / total_models)

        # 保存结果到session state
        st.session_state.model_results = results
        st.success("所有模型训练完成!")

    # 显示结果
    if 'model_results' in st.session_state and st.session_state.model_results:
        results = st.session_state.model_results

        st.divider()
        st.subheader("模型性能比较")

        # 创建性能比较数据框
        metrics_df = pd.DataFrame({
            '模型': [r['model_name'] for r in results],
            '准确率': [r['accuracy'] for r in results],
            '精确率': [r['precision'] for r in results],
            '召回率': [r['recall'] for r in results],
            'F1分数': [r['f1'] for r in results],
            '交叉验证准确率': [r['cv_accuracy'] for r in results],
            '交叉验证F1': [r['cv_f1'] for r in results],
            '训练时间(秒)': [r['train_time'] for r in results]
        })

        # 排序并显示表格
        sort_by = st.selectbox("按指标排序:",
                               ['准确率', 'F1分数', '召回率', '精确率', '训练时间(秒)'],
                               index=0)
        metrics_df = metrics_df.sort_values(by=sort_by, ascending=False)

        # 修复：只对数值列应用格式化
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        styled_metrics = metrics_df.style.format("{:.4f}", subset=numeric_cols)
        st.dataframe(styled_metrics.background_gradient(cmap='Blues', subset=['准确率', 'F1分数', '召回率', '精确率']))

        # 绘制性能比较图表
        st.subheader("性能指标可视化")

        # 选择要可视化的指标
        metric_to_plot = st.selectbox("选择要可视化的指标:",
                                      ['准确率', '精确率', '召回率', 'F1分数', '交叉验证准确率', '交叉验证F1'],
                                      index=0)

        # 创建条形图
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=metrics_df,
            x='模型',
            y=metric_to_plot,
            palette='viridis',
            ax=ax
        )
        ax.set_title(f'模型{metric_to_plot}比较')
        ax.set_ylabel(metric_to_plot)
        ax.set_xlabel('模型')
        plt.xticks(rotation=45)

        # 添加数值标签
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.4f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 5),
                        textcoords='offset points')

        st.pyplot(fig)

        # 绘制多个指标对比雷达图
        st.subheader("多指标性能雷达图")

        # 选择要显示的指标
        selected_metrics = st.multiselect(
            "选择要比较的指标:",
            options=['准确率', '精确率', '召回率', 'F1分数'],
            default=['准确率', '精确率', '召回率', 'F1分数']
        )

        if selected_metrics:
            # 设置雷达图角度
            angles = np.linspace(0, 2 * np.pi, len(selected_metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            for result in results:
                # 获取当前模型的指标值
                model_metrics = metrics_df[metrics_df['模型'] == result['model_name']].iloc[0]
                values = [model_metrics[metric] for metric in selected_metrics]
                values += values[:1]  # 闭合图形

                ax.plot(angles, values, linewidth=1, label=result['model_name'])
                ax.fill(angles, values, alpha=0.1)

            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(selected_metrics)
            ax.set_yticklabels([])
            ax.set_title('模型性能雷达图', size=15)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

            st.pyplot(fig)

        # 绘制ROC曲线
        st.subheader("ROC曲线比较")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot([0, 1], [0, 1], 'k--', label='随机猜测 (AUC=0.5)')

        for result in results:
            if result['fpr'] is not None and result['tpr'] is not None:
                ax.plot(result['fpr'], result['tpr'],
                        label=f"{result['model_name']} (AUC={result['roc_auc']:.4f})")

        ax.set_xlabel('假阳性率 (FPR)')
        ax.set_ylabel('真阳性率 (TPR)')
        ax.set_title('ROC曲线')
        ax.legend(loc='lower right')

        st.pyplot(fig)

        # 模型详细报告
        st.divider()
        st.subheader("模型详细报告")

        selected_model = st.selectbox("选择模型查看详细报告:", [r['model_name'] for r in results])
        model_result = next(r for r in results if r['model_name'] == selected_model)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("混淆矩阵")
            cm = model_result['confusion_matrix']
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('预测标签')
            ax.set_ylabel('真实标签')
            ax.set_title('混淆矩阵')
            st.pyplot(fig)

        with col2:
            st.subheader("分类报告")
            report_df = pd.DataFrame(model_result['classification_report']).transpose()

            # 修复：只对数值列应用格式化
            numeric_cols = report_df.select_dtypes(include=[np.number]).columns
            styled_report = report_df.style.format("{:.4f}", subset=numeric_cols)
            st.dataframe(styled_report.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))

            st.metric("训练时间", f"{model_result['train_time']:.4f} 秒")
            if model_result['roc_auc'] is not None:
                st.metric("AUC分数", f"{model_result['roc_auc']:.4f}")

        # 特征重要性（如果可用）
        if hasattr(model_result['model'], 'feature_importances_'):
            st.subheader("特征重要性")

            try:
                # 获取特征名称
                if use_feature_selection:
                    feature_names = [f"主成分_{i + 1}" for i in range(X_train_scaled.shape[1])]
                else:
                    feature_names = data.feature_names

                # 获取特征重要性
                importances = model_result['model'].feature_importances_
                indices = np.argsort(importances)[::-1]

                # 创建DataFrame
                importance_df = pd.DataFrame({
                    '特征': [feature_names[i] for i in indices],
                    '重要性': importances[indices]
                }).head(15)

                # 绘制条形图
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(
                    data=importance_df,
                    x='重要性',
                    y='特征',
                    palette='viridis',
                    ax=ax
                )
                ax.set_title('Top 15 重要特征')
                ax.set_xlabel('重要性')
                ax.set_ylabel('特征')

                st.pyplot(fig)

                # 显示表格
                st.dataframe(importance_df.style.format({'重要性': '{:.4f}'}).background_gradient(cmap='Blues',
                                                                                                  subset=['重要性']))
            except Exception as e:
                st.warning(f"无法获取特征重要性: {str(e)}")

    st.divider()
    st.info("""
    **性能指标说明:**
    - **准确率 (Accuracy)**: 正确预测的样本比例
    - **精确率 (Precision)**: 预测为正例的样本中实际为正例的比例
    - **召回率 (Recall)**: 实际为正例的样本中被正确预测为正例的比例
    - **F1分数 (F1 Score)**: 精确率和召回率的调和平均，综合衡量模型性能
    - **AUC (Area Under Curve)**: ROC曲线下面积，衡量分类器整体性能
    """)


# 图像分类迁移学习页面
def image_classification_page():
    st.title("图像分类迁移学习")
    st.write("""
    在此页面中，我们将使用预训练网络模型对图像数据集进行迁移学习。
    """)

    # 检查TensorFlow是否安装
    if not check_and_install_tensorflow():
        return

    try:
        import tensorflow as tf
        from tensorflow.keras.datasets import mnist, cifar10
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input, Lambda
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        st.error("TensorFlow 未安装，无法使用图像分类功能")
        return

    # 数据集选择
    dataset_options = {
        "MNIST (手写数字)": mnist,
        "CIFAR-10 (物体识别)": cifar10
    }
    selected_dataset = st.selectbox("选择数据集:", list(dataset_options.keys()))

    # 模型选择
    model_options = {
        "LeNet": "lenet",
        "AlexNet": "alexnet",
        "VGG16 (迁移学习)": "vgg16"
    }
    selected_model = st.selectbox("选择模型架构:", list(model_options.keys()))

    # 训练参数
    st.subheader("训练参数")
    epochs = st.slider("训练轮数 (Epochs):", 1, 20, 5)
    batch_size = st.slider("批大小 (Batch Size):", 16, 128, 32, 16)

    # 加载数据
    if st.button("加载数据并训练模型", type="primary"):
        st.info(f"正在加载 {selected_dataset} 数据集...")

        # 加载数据集
        if selected_dataset == "MNIST (手写数字)":
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            input_shape = (32, 32, 3)  # 修改为32x32 RGB
            num_classes = 10

            # 预处理MNIST数据
            # 1. 将28x28填充为32x32
            X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
            X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)

            # 2. 添加通道维度
            X_train = np.expand_dims(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)

            # 3. 复制单通道为三通道
            X_train = np.repeat(X_train, 3, axis=-1)
            X_test = np.repeat(X_test, 3, axis=-1)

            # 4. 归一化
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0

            # 5. 标签编码
            y_train = to_categorical(y_train, num_classes)
            y_test = to_categorical(y_test, num_classes)

        else:  # CIFAR-10
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
            input_shape = (32, 32, 3)
            num_classes = 10

            # 预处理CIFAR-10数据
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
            y_train = to_categorical(y_train, num_classes)
            y_test = to_categorical(y_test, num_classes)

        # 显示数据集信息
        st.success(f"数据集加载完成!")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"训练集大小: {X_train.shape[0]} 样本")
            st.write(f"测试集大小: {X_test.shape[0]} 样本")
            st.write(f"图像尺寸: {input_shape}")
            st.write(f"类别数: {num_classes}")

        with col2:
            # 显示样本图像
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i in range(5):
                axes[i].imshow(X_train[i])
                axes[i].axis('off')
            st.pyplot(fig)

        # 构建模型
        st.info(f"正在构建 {selected_model} 模型...")

        if model_options[selected_model] == "lenet":
            model = Sequential([
                Conv2D(6, (5, 5), activation='relu', input_shape=input_shape),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(16, (5, 5), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(120, activation='relu'),
                Dense(84, activation='relu'),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        elif model_options[selected_model] == "alexnet":
            # 修正的AlexNet模型，适用于32x32输入
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        elif model_options[selected_model] == "vgg16":
            # 修正的VGG16模型，适用于32x32输入
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

            # 冻结预训练层
            for layer in base_model.layers:
                layer.trainable = False

            # 添加新的分类层
            x = base_model.output
            x = Flatten()(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(num_classes, activation='softmax')(x)

            model = Model(inputs=base_model.input, outputs=predictions)
            model.compile(optimizer=Adam(learning_rate=0.0001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        # 训练模型
        st.info("开始训练模型...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 创建容器用于显示训练进度
        history_container = st.empty()
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        # 自定义回调函数来更新Streamlit界面
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                history['loss'].append(logs['loss'])
                history['accuracy'].append(logs['accuracy'])
                history['val_loss'].append(logs['val_loss'])
                history['val_accuracy'].append(logs['val_accuracy'])

                # 更新进度条
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(
                    f"训练中: Epoch {epoch + 1}/{epochs} - 准确率: {logs['accuracy']:.4f}, 验证准确率: {logs['val_accuracy']:.4f}")

                # 更新训练历史图表
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                ax1.plot(history['loss'], label='训练损失')
                ax1.plot(history['val_loss'], label='验证损失')
                ax1.set_title('训练和验证损失')
                ax1.set_xlabel('Epochs')
                ax1.set_ylabel('损失')
                ax1.legend()

                ax2.plot(history['accuracy'], label='训练准确率')
                ax2.plot(history['val_accuracy'], label='验证准确率')
                ax2.set_title('训练和验证准确率')
                ax2.set_xlabel('Epochs')
                ax2.set_ylabel('准确率')
                ax2.legend()

                history_container.pyplot(fig)

        # 训练模型
        model.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_split=0.2,
                  callbacks=[StreamlitCallback()],
                  verbose=0)

        # 评估模型
        st.info("正在评估模型...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        st.success(f"测试集准确率: {test_acc:.4f}")

        # 显示预测结果
        st.subheader("测试集预测示例")
        n_samples = 10
        sample_indices = np.random.choice(len(X_test), n_samples)
        sample_images = X_test[sample_indices]
        sample_labels = np.argmax(y_test[sample_indices], axis=1)

        predictions = model.predict(sample_images)
        predicted_labels = np.argmax(predictions, axis=1)

        # 定义CIFAR-10类别名称
        cifar10_labels = [
            '飞机', '汽车', '鸟', '猫', '鹿',
            '狗', '青蛙', '马', '船', '卡车'
        ]

        # 显示预测结果
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i, idx in enumerate(sample_indices):
            axes[i].imshow(X_test[idx])

            true_label = sample_labels[i]
            pred_label = predicted_labels[i]
            color = 'green' if true_label == pred_label else 'red'

            # 根据数据集选择标签显示方式
            if selected_dataset == "CIFAR-10 (物体识别)":
                true_label_name = cifar10_labels[true_label]
                pred_label_name = cifar10_labels[pred_label]
            else:  # MNIST
                true_label_name = str(true_label)
                pred_label_name = str(pred_label)

            axes[i].set_title(f"真实: {true_label_name}\n预测: {pred_label_name}", color=color)
            axes[i].axis('off')
        st.pyplot(fig)

        # 分析传统神经网络与深度网络的差异
        st.divider()
        st.subheader("神经网络性能分析")

        st.markdown("""
        **传统神经网络（感知器、MLP）与深度网络（CNN）的差异分析：**

        1. **特征提取能力**:
           - 传统神经网络：需要手动特征工程，处理图像等高维数据效果有限
           - 深度网络：自动学习层次化特征表示，特别适合图像数据

        2. **参数数量**:
           - 传统神经网络：参数较少，训练速度快
           - 深度网络：参数较多，需要更多计算资源和训练时间

        3. **小数据集表现**:
           - 传统神经网络：在小数据集上不容易过拟合
           - 深度网络：在小数据集上容易过拟合，需要使用数据增强、正则化等技术

        4. **迁移学习**:
           - 传统神经网络：迁移学习效果有限
           - 深度网络：可以通过预训练模型进行迁移学习，大幅提升小数据集上的性能

        **当前实验结果分析：**
        - 在图像分类任务上，深度网络（如LeNet、AlexNet、VGG）通常优于传统神经网络
        - 迁移学习（如VGG16）在小数据集上表现尤为出色，因为可以利用预训练的特征提取能力
        - 对于乳腺癌等结构化数据，传统神经网络可能表现足够好，且训练更快
        """)

        # 添加大模型分析
        if deepseek_key:
            st.divider()
            st.subheader("AI深度分析")

            context = f"用户使用{selected_model}模型在{selected_dataset}数据集上进行了图像分类实验，测试准确率为{test_acc:.4f}。"
            prompt = "请分析传统神经网络与深度网络在小数据集和复杂任务上的效果差异"

            if st.button("获取AI分析"):
                with st.spinner("AI正在分析..."):
                    analysis_result = analyze_with_deepseek(deepseek_key, context, prompt=prompt)
                    st.success("AI分析结果:")
                    st.markdown(analysis_result)


# 侧边栏导航
with st.sidebar:
    st.title("神经网络分析平台")
    st.divider()

    # 添加DeepSeek API密钥输入
    st.subheader("大模型集成")
    deepseek_key = st.text_input("DeepSeek API密钥", type="password",
                                 help="输入您的DeepSeek API密钥以启用智能分析功能")

    # 创建导航菜单
    nav_selection = st.radio(
        "选择功能模块:",
        ["首页", "乳腺癌EDA分析", "YOLOv8 物体检测演示", "模型比较", "图像分类迁移学习"],
        index=0
    )

    st.divider()

    # 乳腺癌EDA分析特定设置
    if nav_selection == "乳腺癌EDA分析":
        st.subheader("乳腺癌分析设置")
        analysis_section = st.selectbox(
            "选择分析部分",
            ["数据集概览", "特征分布", "特征相关性", "特征与诊断关系", "降维分析", "分析结论"],
            index=0
        )

        features = load_breast_cancer().feature_names
        selected_features = st.multiselect(
            "选择要分析的特征",
            options=features,
            default=features[:5]
        )

        st.divider()
        st.info("""
        **数据集信息**  
        - 样本数: 569  
        - 特征数: 30  
        - 目标变量: 诊断结果 (0=恶性, 1=良性)  
        """)

if nav_selection == "首页":
    st.title("🧠 神经网络项目小平台")
    st.divider()

    st.header("欢迎使用三人小组的一个关于神经网络的小平台")
    st.subheader("使用方法如下：")

    # 欢迎信息
    st.success("""
    - 左边选择功能模块
    - 打开功能模块，继续向下选择，直到满意为止
    """)

    # 添加一些装饰元素
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("https://img.keaitupian.cn/newupload/11/1669620398726353.jpg")
    st.subheader("小备注：")
    st.markdown("""
        1. **项目小组成员**：
           - 2300131006 郝珊
           - 2300131044 陈梦佳
           - 2300131020 尹锦庭

        2. **项目分工**：
           - 郝珊：
           - 陈梦佳：
           - 尹锦庭：

        """)

# 乳腺癌EDA分析
elif nav_selection == "乳腺癌EDA分析":
    df, data = load_breast_cancer_data()

    # 预先计算诊断结果分布，所有部分都会用到
    diagnosis_counts = df['diagnosis'].value_counts()

    st.title("乳腺癌数据集探索性分析(EDA)")
    st.divider()

    # 数据集概览 - 保持不变
    if analysis_section == "数据集概览":
        st.header("数据集概览")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("数据摘要")
            st.write(f"样本数: {df.shape[0]}")
            st.write(f"特征数: {df.shape[1] - 2} (不包括目标变量)")

            st.subheader("诊断结果分布")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(diagnosis_counts, labels=diagnosis_counts.index,
                   autopct='%1.1f%%', startangle=90,
                   colors=['#ff9999', '#66b3ff'])
            ax.set_title('诊断结果分布')
            st.pyplot(fig)

        with col2:
            st.subheader("数据预览")
            st.dataframe(df.head(10))

            st.subheader("特征描述统计")
            st.dataframe(df[data.feature_names].describe().T.style.background_gradient(cmap='Blues'))

        # 添加大模型分析
        if deepseek_key:
            st.divider()
            st.subheader("智能数据分析")

            # 准备分析数据
            context = "用户正在查看乳腺癌数据集概览，包括数据摘要、诊断结果分布和特征描述统计。"
            data_summary = f"""
            数据集包含{len(df)}个样本，{len(data.feature_names)}个特征。
            诊断结果分布: {diagnosis_counts['恶性']}个恶性样本({diagnosis_counts['恶性'] / len(df) * 100:.1f}%), 
            {diagnosis_counts['良性']}个良性样本({diagnosis_counts['良性'] / len(df) * 100:.1f}%)。
            """

            # 用户提示输入
            user_prompt = st.text_area("向AI提问或请求分析:",
                                       "请分析乳腺癌数据集的基本特征和诊断分布情况")

            if st.button("获取AI分析"):
                with st.spinner("AI正在分析数据..."):
                    analysis_result = analyze_with_deepseek(deepseek_key, context, data_summary, user_prompt)
                    st.success("AI分析结果:")
                    st.markdown(analysis_result)

    # 特征分布分析部分
    elif analysis_section == "特征分布":
        st.header("特征分布分析")

        st.subheader("特征直方图")
        cols = st.columns(3)
        for i, feature in enumerate(selected_features):
            with cols[i % 3]:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df[feature], kde=True, ax=ax, color='skyblue')
                ax.set_title(f"{feature} 分布")
                st.pyplot(fig)

        st.subheader("特征箱线图")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_melt = pd.melt(df, value_vars=selected_features)
        sns.boxplot(x='variable', y='value', data=df_melt, ax=ax)
        plt.xticks(rotation=45)
        ax.set_title('特征箱线图')
        ax.set_xlabel('特征')
        ax.set_ylabel('值')
        st.pyplot(fig)

        # 添加大模型分析
        if deepseek_key:
            st.divider()
            st.subheader("智能数据分析")

            # 准备分析数据 - 更新为特征分布相关内容
            context = "用户正在分析乳腺癌数据集的特征分布情况。"
            data_summary = f"""
            已选择分析以下特征: {', '.join(selected_features)}
            数据集包含{len(df)}个样本，{len(data.feature_names)}个特征。
            """

            # 用户提示输入
            user_prompt = st.text_area("向AI提问或请求分析:",
                                       "请分析这些特征分布的特点及其对乳腺癌诊断的意义")

            if st.button("获取AI分析"):
                with st.spinner("AI正在分析数据..."):
                    analysis_result = analyze_with_deepseek(deepseek_key, context, data_summary, user_prompt)
                    st.success("AI分析结果:")
                    st.markdown(analysis_result)

    # 特征相关性分析部分
    elif analysis_section == "特征相关性":
        st.header("特征相关性分析")

        st.subheader("特征相关矩阵")
        corr_matrix = df[data.feature_names].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
        ax.set_title('特征相关矩阵')
        st.pyplot(fig)

        st.subheader("高度相关特征 (|r| > 0.8)")
        high_corr = corr_matrix.abs()
        high_corr = high_corr.unstack()
        high_corr = high_corr.sort_values(ascending=False)
        high_corr = high_corr[high_corr > 0.8]
        high_corr = high_corr[high_corr < 1]  # 排除对角线
        high_corr = high_corr.reset_index()
        high_corr.columns = ['特征1', '特征2', '相关系数']
        st.dataframe(high_corr.drop_duplicates().style.background_gradient(cmap='YlOrRd'))

        # 添加大模型分析
        if deepseek_key:
            st.divider()
            st.subheader("智能数据分析")

            # 准备分析数据 - 更新为特征相关性相关内容
            context = "用户正在分析乳腺癌数据集的特征相关性情况。"
            data_summary = f"""
            已识别出{len(high_corr)}对高度相关特征(|r| > 0.8)。
            数据集包含{len(df)}个样本，{len(data.feature_names)}个特征。
            """

            # 用户提示输入
            user_prompt = st.text_area("向AI提问或请求分析:",
                                       "请解释特征相关性分析结果对构建乳腺癌诊断模型的意义")

            if st.button("获取AI分析"):
                with st.spinner("AI正在分析数据..."):
                    analysis_result = analyze_with_deepseek(deepseek_key, context, data_summary, user_prompt)
                    st.success("AI分析结果:")
                    st.markdown(analysis_result)

    # 特征与诊断关系分析部分
    elif analysis_section == "特征与诊断关系":
        st.header("特征与诊断结果的关系")

        st.subheader("诊断结果与特征关系图")
        cols = st.columns(3)
        for i, feature in enumerate(selected_features):
            with cols[i % 3]:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(x='diagnosis', y=feature, data=df, ax=ax, palette='Set2')
                ax.set_title(f"{feature} vs 诊断结果")
                st.pyplot(fig)

        st.subheader("特征均值对比")
        mean_features = [f for f in data.feature_names if 'mean' in f]
        benign_means = df[df['diagnosis'] == '良性'][mean_features].mean()
        malignant_means = df[df['diagnosis'] == '恶性'][mean_features].mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        index = np.arange(len(mean_features))
        bar_width = 0.35
        ax.bar(index, benign_means, bar_width, label='良性', color='#66c2a5')
        ax.bar(index + bar_width, malignant_means, bar_width, label='恶性', color='#fc8d62')
        ax.set_xlabel('特征')
        ax.set_ylabel('均值')
        ax.set_title('良性与恶性样本特征均值对比')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(mean_features, rotation=90)
        ax.legend()
        st.pyplot(fig)

        # 添加大模型分析
        if deepseek_key:
            st.divider()
            st.subheader("智能数据分析")

            # 准备分析数据 - 更新为特征与诊断关系相关内容
            context = "用户正在分析乳腺癌数据集的特征与诊断结果的关系。"
            data_summary = f"""
            已选择分析以下特征: {', '.join(selected_features)}
            对比了良性与恶性肿瘤样本的特征均值。
            """

            # 用户提示输入
            user_prompt = st.text_area("向AI提问或请求分析:",
                                       "请解释这些特征与乳腺癌诊断的关系及其临床意义")

            if st.button("获取AI分析"):
                with st.spinner("AI正在分析数据..."):
                    analysis_result = analyze_with_deepseek(deepseek_key, context, data_summary, user_prompt)
                    st.success("AI分析结果:")
                    st.markdown(analysis_result)

    # 降维分析部分
    elif analysis_section == "降维分析":
        st.header("降维分析")

        st.subheader("主成分分析(PCA)")

        # 标准化数据
        X = df[data.feature_names]
        y = df['diagnosis']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 应用PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca_df['diagnosis'] = y

        # 解释方差
        st.write(f"第一主成分解释方差: {pca.explained_variance_ratio_[0]:.2%}")
        st.write(f"第二主成分解释方差: {pca.explained_variance_ratio_[1]:.2%}")
        st.write(f"累计解释方差: {pca.explained_variance_ratio_.sum():.2%}")

        # PCA可视化
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='diagnosis', data=pca_df,
                        palette={'良性': 'green', '恶性': 'red'}, alpha=0.7, ax=ax)
        ax.set_title('PCA: 前两个主成分')
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        st.pyplot(fig)

        # 特征重要性
        st.subheader("主成分特征重要性")
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=data.feature_names)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(loadings_df, cmap='coolwarm', annot=True, fmt=".2f", ax=ax)
        ax.set_title('特征对主成分的贡献')
        st.pyplot(fig)

        # 添加大模型分析
        if deepseek_key:
            st.divider()
            st.subheader("智能数据分析")

            # 准备分析数据 - 更新为降维分析相关内容
            context = "用户正在对乳腺癌数据集进行降维分析（PCA）。"
            data_summary = f"""
            前两个主成分累计解释方差: {pca.explained_variance_ratio_.sum():.2%}
            数据集包含{len(df)}个样本，{len(data.feature_names)}个特征。
            """

            # 用户提示输入
            user_prompt = st.text_area("向AI提问或请求分析:",
                                       "请解释PCA降维结果对乳腺癌数据可视化和特征提取的意义")

            if st.button("获取AI分析"):
                with st.spinner("AI正在分析数据..."):
                    analysis_result = analyze_with_deepseek(deepseek_key, context, data_summary, user_prompt)
                    st.success("AI分析结果:")
                    st.markdown(analysis_result)

    # 分析结论部分 - 保持不变
    elif analysis_section == "分析结论":
        st.header("乳腺癌数据集分析结论")
        st.divider()

        st.subheader("关键发现")
        st.markdown("""
        1. **诊断结果分布**：  
           - 数据集包含569个样本，其中良性肿瘤357例(62.7%)，恶性肿瘤212例(37.3%)
           - 类别分布存在轻微不平衡，在建模时需注意

        2. **特征分布**：  
           - 所有特征均为连续数值型变量
           - 大多数特征呈现右偏分布，表明存在一些极端值
           - 不同特征的值范围差异较大，需要标准化处理

        3. **特征相关性**：  
           - 特征间存在高度相关性，特别是同一特征的均值、标准差和最差值之间
           - 例如，`radius_mean`与`perimeter_mean`、`area_mean`相关系数超过0.99
           - 高相关性表明可以进行特征选择或降维以减少冗余

        4. **特征与诊断的关系**：  
           - 恶性肿瘤样本在大多数特征上的值显著高于良性样本
           - 关键区分特征包括：`radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`
           - 肿瘤大小相关特征(半径、周长、面积)是重要的预测指标

        5. **降维分析**：  
           - PCA分析显示前两个主成分可以解释约63%的总方差
           - 在二维空间中，良性和恶性肿瘤有较好的分离趋势
           - 纹理特征(`texture_mean`)对第一主成分贡献最大
        """)

        st.subheader("建模建议")
        st.markdown("""
        1. **数据预处理**：
           - 对连续特征进行标准化或归一化
           - 处理可能的离群值
           - 考虑处理类别不平衡问题(过采样/欠采样)

        2. **特征工程**：
           - 使用PCA或其他降维技术减少特征维度
           - 移除高度相关的特征以减少多重共线性
           - 创建新特征如特征比率(如面积/半径)

        3. **模型选择**：
           - 逻辑回归(可解释性高)
           - 支持向量机(适合高维数据)
           - 随机森林(处理非线性关系)
           - 神经网络(处理复杂模式)

        4. **模型评估**：
           - 使用精确率、召回率和F1分数(由于类别不平衡)
           - ROC曲线和AUC值
           - 混淆矩阵分析
        """)

        st.subheader("潜在挑战")
        st.markdown("""
        - **特征冗余**：高度相关的特征可能降低某些模型性能
        - **类别不平衡**：良性样本多于恶性样本，可能导致模型偏向多数类
        - **维度灾难**：30个特征相对样本量可能较多，需考虑降维
        - **过拟合风险**：需要适当使用正则化技术
        """)

        st.success("""
        **总结**：乳腺癌数据集质量良好，特征与目标变量有明确关系。通过适当的特征选择和预处理，
        可以构建高性能的诊断模型。建议重点关注肿瘤大小相关特征和纹理特征，并采用正则化方法防止过拟合。
        """)

        # 添加深度AI分析
        if deepseek_key:
            st.divider()
            st.subheader("AI深度分析报告")

            # 准备分析数据
            context = "用户已完成乳腺癌数据集的全面EDA分析，现在需要专业的数据科学建议。"
            data_summary = """
            关键发现总结:
            1. 诊断分布: 62.7%良性, 37.3%恶性
            2. 特征高度相关，特别是同一特征的均值、标准差和最差值之间
            3. 恶性肿瘤样本在大多数特征上的值显著高于良性样本
            4. PCA分析显示前两个主成分可以解释约63%的总方差
            """

            # 用户提示输入
            user_prompt = st.text_area("向AI提问或请求分析:",
                                       "基于分析结论，请提供详细的建模策略和医疗应用建议")

            if st.button("获取专业建议"):
                with st.spinner("AI正在生成专业建议..."):
                    analysis_result = analyze_with_deepseek(deepseek_key, context, data_summary, user_prompt)
                    st.success("AI专业建议:")
                    st.markdown(analysis_result)

                    # 添加下载报告功能
                    st.download_button(
                        label="下载分析报告",
                        data=analysis_result,
                        file_name="乳腺癌分析报告.md",
                        mime="text/markdown"
                    )

# YOLOv8 物体检测演示
elif nav_selection == "YOLOv8 物体检测演示":
    st.title("YOLOv8 物体检测演示")
    model = load_model()
    # 侧边栏设置
    with st.sidebar:
        st.header("上传配置")

        # 选择分析类型
        analysis_type = st.radio(
            "选择分析类型",
            ["图片分析", "视频分析"],
            index=0,
            horizontal=True
        )

        # 根据选择类型显示不同上传器
        if analysis_type == "图片分析":
            uploaded_file = st.file_uploader(
                "上传图片",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=False
            )
        else:  # 视频分析
            uploaded_file = st.file_uploader(
                "上传视频",
                type=['mp4', 'avi', 'mov', 'mkv'],
                accept_multiple_files=False
            )

        # 检测参数
        st.subheader("检测参数")
        conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.25, 0.05)
        line_width = st.slider("边界框宽度", 1, 5, 2)

        # 视频特有参数
        if analysis_type == "视频分析":
            frame_skip = st.slider("帧跳过率", 1, 10, 2,
                                   help="值越大处理越快但可能漏检对象")
            st.divider()
            st.info("""
                    **视频处理说明:**
                    - 处理速度取决于视频长度和帧跳过率
                    - 处理完成后可下载分析结果视频
                    """)

        # 说明信息
        st.divider()
        st.info("""
                ### 使用说明:
                1. 上传图片或视频文件
                2. 调整检测参数
                3. 查看右侧检测结果
                """)

    # 主内容区域
    col1, col2 = st.columns(2)

    # 添加AI分析选项
    if deepseek_key:
        st.divider()
        st.subheader("AI分析设置")
        ai_analysis_prompt = st.text_area("AI分析提示:",
                                          "请描述图片中的物体及其相互关系")


    # 图片分析功能
    def process_image(uploaded_file):
        # 读取上传的图片
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        with col1:
            st.subheader("原始图片")
            st.image(image, use_column_width=True)

        # 执行物体检测
        with st.spinner("检测中..."):
            results = model.predict(
                source=img_array,
                conf=conf_threshold,
                line_width=line_width,
                save=False  # 不在本地保存结果
            )

        # 处理检测结果
        if results:
            # 获取带标注的图像
            annotated_frame = results[0].plot(line_width=line_width)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # 检测结果统计
            detected_objects = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    confidence = float(box.conf)
                    detected_objects.append({
                        "对象": class_name,
                        "置信度": f"{confidence:.2f}"
                    })

            with col2:
                st.subheader("检测结果")
                st.image(annotated_frame, use_column_width=True)

                if detected_objects:
                    st.subheader(f"检测到 {len(detected_objects)} 个对象")
                    st.table(detected_objects)
                else:
                    st.warning("未检测到任何对象")

        # 在检测结果后添加AI分析
        if deepseek_key and detected_objects:
            st.divider()
            st.subheader("AI场景分析")

            # 构建AI分析上下文
            context = f"用户上传了一张图片，YOLOv8检测到以下物体: {', '.join([obj['对象'] for obj in detected_objects])}"

            if st.button("请求AI分析场景"):
                with st.spinner("AI正在分析场景..."):
                    analysis_result = analyze_with_deepseek(
                        deepseek_key,
                        context,
                        prompt=ai_analysis_prompt
                    )
                    st.success("场景分析结果:")
                    st.markdown(analysis_result)


    # 视频分析功能
    def process_video(uploaded_file):
        # 创建临时文件保存上传的视频
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_file.read())
        temp_file.close()

        # 打开视频文件
        cap = cv2.VideoCapture(temp_file.name)

        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建临时文件保存处理后的视频
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_file.close()

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))

        # 显示原始视频
        with col1:
            st.subheader("原始视频")
            st.video(temp_file.name)

        # 进度条和状态信息
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 处理视频帧
        processed_frames = []
        frame_count = 0
        objects_detected = {}

        # 创建容器显示实时处理
        frame_container = col2.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 跳过指定帧数
            if frame_count % frame_skip != 0:
                continue

            # 更新进度
            progress = min(100, int((frame_count / total_frames) * 100))
            progress_bar.progress(progress)
            status_text.text(f"处理中: {frame_count}/{total_frames} 帧 ({progress}%)")

            # 执行物体检测
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                line_width=line_width,
                save=False
            )

            # 处理检测结果
            if results:
                # 获取带标注的帧
                annotated_frame = results[0].plot(line_width=line_width)

                # 添加到输出视频
                out.write(annotated_frame)

                # 更新检测结果
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = model.names[class_id]
                        if class_name not in objects_detected:
                            objects_detected[class_name] = 0
                        objects_detected[class_name] += 1

                # 显示当前处理帧
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_container.image(display_frame, caption=f"处理帧: {frame_count}", use_column_width=True)

        # 释放资源
        cap.release()
        out.release()

        # 完成处理
        progress_bar.progress(100)
        status_text.text(f"处理完成! 共处理 {frame_count} 帧")

        # 显示检测结果
        with col2:
            st.subheader("检测结果")
            st.video(output_file.name)

            # 提供下载链接
            with open(output_file.name, 'rb') as f:
                video_bytes = f.read()
            st.download_button(
                label="下载处理后的视频",
                data=video_bytes,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

            # 显示检测统计
            if objects_detected:
                st.subheader("检测统计")
                st.write(f"共检测到 {sum(objects_detected.values())} 个对象")

                # 转换为表格显示
                detected_table = []
                for obj, count in objects_detected.items():
                    detected_table.append({
                        "对象": obj,
                        "出现次数": count
                    })
                st.table(detected_table)
            else:
                st.warning("未检测到任何对象")

        # 清理临时文件
        os.unlink(temp_file.name)
        os.unlink(output_file.name)

        # 在视频分析后添加AI分析
        if deepseek_key and objects_detected:
            st.divider()
            st.subheader("AI视频内容分析")

            # 构建AI分析上下文
            context = f"分析了一个视频，检测到以下物体: {', '.join([f'{obj}({count}次)' for obj, count in objects_detected.items()])}"

            if st.button("请求AI分析视频内容"):
                with st.spinner("AI正在分析视频内容..."):
                    analysis_result = analyze_with_deepseek(
                        deepseek_key,
                        context,
                        prompt="请总结视频中的主要内容和物体互动关系"
                    )
                    st.success("视频内容分析:")
                    st.markdown(analysis_result)


    # 主逻辑
    if uploaded_file is not None:
        if analysis_type == "图片分析":
            process_image(uploaded_file)
        else:
            process_video(uploaded_file)

# 模型训练模块
elif nav_selection == "模型比较":
    model_comparison_page()

# 图像分类迁移学习
elif nav_selection == "图像分类迁移学习":
    image_classification_page()


# 添加页脚
st.divider()
st.caption("© 2025 神经网络分析平台 | 版本 v2.0")
