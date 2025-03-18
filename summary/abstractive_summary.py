# -*- encoding: utf-8 -*-
"""
@Author : Sophia Xu
@File : abstractive_summary.py
@Time : 2025/02/28 13:52:57
@Desc : 概述性摘要：用模型序列到序列模型（Seq2Seq）和注意力机制（Attention Mechanism）生成新的文本
"""


from transformers import pipeline

def generate_abstractive_summary(text, model_name="sshleifer/distilbart-cnn-12-6", max_length=50):
    """
    使用预训练模型生成概述式摘要。
    
    参数:
    - text: 输入文本
    - model_name: 使用的预训练模型，默认为 distilbart-cnn-12-6
    - max_length: 摘要的最大长度
    """
    # 初始化摘要生成器
    summarizer = pipeline("summarization", model=model_name)
    
    # 生成摘要
    summary = summarizer(text, max_length=max_length, clean_up_tokenization_spaces=True)
    
    # 返回摘要文本
    return summary[0]['summary_text']


text = """
Natural language processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and humans through natural language.
The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful.
Some of the key applications of NLP include machine translation, sentiment analysis, and text summarization.
In recent years, deep learning techniques have significantly advanced the capabilities of NLP systems.
However, traditional NLP methods still play an important role in many applications.
"""

# 生成概述式摘要
summary = generate_abstractive_summary(text, max_length=50)
print("原文：")
print(text)
print("\n摘要：")
print(summary)