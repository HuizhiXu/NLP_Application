# -*- encoding: utf-8 -*-
"""
@Author : Sophia Xu
@File : exrtacive_summary.py
@Time : 2025/02/28 13:49:01
@Desc : 抽取式摘要生成:基于句子的位置、关键词频率以及句子长度来评估句子的重要性，并从中抽取关键句子组成摘要。
Extractive Summary Generation: Evaluate the importance of sentences based on their position, keyword frequency, and sentence length, and extract key sentences to form the summary.
"""



from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """
    文本预处理：分句、分词、去除停用词
    """

    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return sentences, words

def calculate_sentence_scores(sentences, words):
    """
    计算句子得分
    基于关键词频率、句子位置和句子长度
    """
    word_freq = FreqDist(words)
    sentence_scores = defaultdict(int)
    
    for index, sentence in enumerate(sentences):
        # 分词
        sentence_words = word_tokenize(sentence.lower())
        sentence_length = len(sentence_words)
        
        # 计算关键词得分
        for word in sentence_words:
            if word in word_freq:
                sentence_scores[index] += word_freq[word]
        
        # 考虑句子位置（首句和末句得分更高）
        if index == 0 or index == len(sentences) - 1:
            sentence_scores[index] += 1.5
        
        # 考虑句子长度（适中长度的句子得分更高）
        if 10 <= sentence_length <= 30:
            sentence_scores[index] += 1
    
    return sentence_scores

def generate_summary(text, num_sentences=3):
    """
    生成摘要
    """
    sentences, words = preprocess_text(text)
    sentence_scores = calculate_sentence_scores(sentences, words)
    
    # 选择得分最高的句子
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: x[0])  # 按原文顺序排序
    
    # 生成摘要
    summary = ' '.join([sentences[index] for index, score in top_sentences])
    return summary


text = """
Natural language processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and humans through natural language.
The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful.
Some of the key applications of NLP include machine translation, sentiment analysis, and text summarization.
In recent years, deep learning techniques have significantly advanced the capabilities of NLP systems.
However, traditional NLP methods still play an important role in many applications.
"""

# 生成摘要
summary = generate_summary(text, num_sentences=2)
print("原文：")
print(text)
print("\n摘要：")
print(summary)