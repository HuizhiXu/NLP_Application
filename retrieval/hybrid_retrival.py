# -*- encoding: utf-8 -*-
"""
@Author : Sophia Xu
@File : hybrid_retrival.py
@Time : 2025/02/28 14:38:34
@Desc : 
"""

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

# 示例文档集合
corpus = [
    "量子计算是一种基于量子力学原理的计算方式。",
    "人工智能是计算机科学的一个分支。",
    "深度学习是机器学习的一个子领域。",
    "量子计算可以解决传统计算机无法处理的问题。"
]

# 初始化 BM25
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# 初始化向量搜索模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
corpus_embeddings = model.encode(corpus)

# 用户查询
query = "量子计算是什么？"
tokenized_query = query.split()
query_embedding = model.encode([query])[0]

# BM25 检索
bm25_scores = bm25.get_scores(tokenized_query)
bm25_top_k_indices = np.argsort(bm25_scores)[::-1][:2]  # 取 Top-2

# 向量搜索检索
cosine_scores = np.dot(corpus_embeddings, query_embedding) / (np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_embedding))
vector_top_k_indices = np.argsort(cosine_scores)[::-1][:2]  # 取 Top-2

# 融合检索结果
combined_indices = set(bm25_top_k_indices).union(set(vector_top_k_indices))
combined_scores = {i: bm25_scores[i] + cosine_scores[i] for i in combined_indices}
final_top_k_indices = sorted(combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True)[:2]

# 输出检索结果
print("BM25 检索结果：")
for i in bm25_top_k_indices:
    print(f"文档 {i}: {corpus[i]} (BM25 得分: {bm25_scores[i]})")

print("\n向量搜索检索结果：")
for i in vector_top_k_indices:
    print(f"文档 {i}: {corpus[i]} (余弦相似度: {cosine_scores[i]})")

print("\n混合检索结果：")
for i in final_top_k_indices:
    print(f"文档 {i}: {corpus[i]} (综合得分: {combined_scores[i]})")