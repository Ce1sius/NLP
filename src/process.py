"""
在这个文件中，我们要对数据进行预先的处理，从原有的jsonl格式中，提取出对话文本，然后分词，转换为ID
并且创建出窗口大小为5的input以及目标target
同时将原数据集划分为训练集，测试集和验证集
"""
from email.encoders import encode_noop

import jieba
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split

import config

def build_dataset(sentences, word2index):
    dataset = []
    indexed_sentences = [[word2index.get(token, word2index['<unk>']) for token in jieba.lcut(sentence)] for
                               sentence in sentences]
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
    vocab_set = set()
    for sentence in train_sentences:
        vocab_set.update(jieba.lcut(sentence))
    vocab_list = ['<unk>'] + list(vocab_set)
    #print(len(vocab_list))

    #5.保存词表
    with open(config.MODELS_DIR / 'vocab_list.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab_list))

    #6.构建训练集
    word2index = {word: index for index, word in enumerate(vocab_list)}
    train_dataset = build_dataset(train_sentences, word2index)
    #7.保存训练集
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR / 'train_dataset.jsonl', orient='records', lines=True)

    #8.构建测试集
    test_dataset = build_dataset(test_sentences, word2index)
    #9.保存测试集
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / 'test_dataset.jsonl', orient='records', lines=True)

if __name__ == '__main__':
    process()