"""
读取原始数据，将csv，jsonl等文件格式转换成模型能识别的张量
定义dataloader，getitem,len等基础方法
"""
import pandas as pd
from sympy import false
from torch.utils.data import Dataset, DataLoader
import torch

import config


class InputMethodDataset(Dataset):

    def __init__(self, path):
        self.data = pd.read_json(path, orient='records', lines=True).sample(frac=0.1).to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['input'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['target'], dtype=torch.long)
        return input_tensor, target_tensor

def get_dataloader(train=True):
    path = config.PROCESSED_DATA_DIR / ('train_dataset.jsonl' if train else 'test_dataset.jsonl')
    dataset = InputMethodDataset(path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=false)
    print(len(test_dataloader))
    print(len(train_dataloader))

