from tqdm import tqdm

import jieba


class JiebaTokenizer:
    unk_token = '<unk>'
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(self.vocab_list)
        self.word2index = {word: index for index, word in enumerate(self.vocab_list)}
        self.index2word = {index: word for index, word in enumerate(self.vocab_list)}
        self.unk_token_id = self.word2index[self.unk_token]

    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.word2index.get(token, self.unk_token_id) for token in tokens]

    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        # 4.构建词表
        vocab_set = set()
        for sentence in tqdm(sentences,desc="Building vocabulary"):
            vocab_set.update(jieba.lcut(sentence))

        vocab_list = [cls.unk_token] + list(vocab_set)

        # 5.保存词表
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [vocab.strip() for vocab in f.readlines()]
        return cls(vocab_list)
