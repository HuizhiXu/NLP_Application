# -*- encoding: utf-8 -*-
"""
@Author : Sophia Xu
@File : bm25.py
@Time : 2025/02/28 14:14:30
@Desc : 
"""
import math
from collections import defaultdict

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        self.doc_freqs = self._compute_doc_freqs()
        self.idf = self._compute_idf()

    def _compute_doc_freqs(self):
        doc_freqs = defaultdict(int)
        for doc in self.corpus:
            for word in set(doc):
                doc_freqs[word] += 1
        return doc_freqs

    def _compute_idf(self):
        idf = {}
        num_docs = len(self.corpus)
        for word, freq in self.doc_freqs.items():
            idf[word] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)
            
        return idf

    def _score(self, doc, query):
        score = 0.0
        for word in query:
            if word in self.idf:
                tf = doc.count(word)
                numerator = self.idf[word] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (len(doc) / self.avgdl))
                score += numerator / denominator
        return score

    def get_scores(self, query):
        scores = [self._score(doc, query) for doc in self.corpus]
        return scores

# 示例用法
corpus = [
    ["the", "quick", "brown", "fox"],
    ["jumped", "over", "the", "lazy", "dog"],
    ["the", "dog", "was", "lazy"],
    ["the", "fox", "was", "quick"]
]

bm25 = BM25(corpus)
query = ["quick", "fox"]
scores = bm25.get_scores(query)
print(scores)  # 输出每个文档的 BM25 评分