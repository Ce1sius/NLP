"""
在这个文件中，我们要对数据进行预先的处理，从原有的jsonl格式中，提取出对话文本，然后分词，转换为ID
并且创建出窗口大小为5的input以及目标target
同时将原数据集划分为训练集，测试集和验证集
"""


import jieba
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split

from tokenizer import JiebaTokenizer
import config

def build_dataset(sentences, tokenizer):
    dataset = []
    indexed_sentences = [tokenizer.encode(sentence) for sentence in sentences]
    for sentence in tqdm(indexed_sentences, desc="构建词表"):
        for i in range(len(sentence) - config.SEQ_LEN):
            input = sentence[i: i + config.SEQ_LEN]
            target = sentence[i + config.SEQ_LEN]
            dataset.append({'input': input, 'target': target})
    return dataset

def process():
    #1.读取原始数据
    df = pd.read_json(config.RAW_DATA_DIR / "synthesized_.jsonl", lines=True, orient='records')

    #2.提取所有句子
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split('：')[1])

    #3.划分数据集
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2)

    #4.构建词表
    JiebaTokenizer.build_vocab(train_sentences, config.MODELS_DIR / "vocab_list .txt")

    #6.构建训练集
    tokenizer = JiebaTokenizer.from_vocab( config.MODELS_DIR / "vocab_list .txt")
    train_dataset = build_dataset(train_sentences, tokenizer)
    #7.保存训练集
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR / 'train_dataset.jsonl', orient='records', lines=True)

    #8.构建测试集
    test_dataset = build_dataset(test_sentences, tokenizer)
    #9.保存测试集
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / 'test_dataset.jsonl', orient='records', lines=True)

if __name__ == '__main__':
    process()