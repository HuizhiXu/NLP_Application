# -*- encoding: utf-8 -*-
"""
@Author : Sophia Xu
@File : lightgbm.py
@Time : 2025/03/14 15:26:43
@Desc : 
"""


from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import numpy as np

# 加载数据集
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# 简化为 10 个主题
selected_categories = data.target_names[:10]
data = fetch_20newsgroups(subset='all', categories=selected_categories, remove=('headers', 'footers', 'quotes'))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 初始化 TF-IDF 向量化器
vectorizer = TfidfVectorizer(max_features=5000)

# 将文本数据转换为 TF-IDF 特征
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 初始化 LightGBM 分类器
lgb_clf = lgb.LGBMClassifier(
    objective='multiclass',  # 多分类任务
    num_class=10,            # 10 个类别
    boosting_type='gbdt',    # 使用 GBDT
    num_leaves=31,           # 叶子节点数
    learning_rate=0.05,      # 学习率
    n_estimators=100,        # 树的数量
    random_state=42          # 随机种子
)

# 训练模型
lgb_clf.fit(X_train_tfidf, y_train)

# 预测测试集
y_pred = lgb_clf.predict(X_test_tfidf)

# 计算准确率
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.4f}')

# 新的文本
new_text = ["This is a sample text about technology and computers."]

# 将新文本转换为 TF-IDF 特征
new_text_tfidf = vectorizer.transform(new_text)

# 预测新文本的主题
predicted_class = lgb_clf.predict(new_text_tfidf)
predicted_label = data.target_names[predicted_class[0]]

print(f'Predicted class: {predicted_class[0]}')
print(f'Predicted label: {predicted_label}')