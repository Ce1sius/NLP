import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import InputMethodModel


def train_one_epoch(model, dataloader, loss_f, optimizer, device):
    """

    :param model: 模型
    :param dataloader:数据集
    :param loss_f: 损失函数
    :param optimizer: adam优化器
    :param device: mps
    :return: 平均损失
    """
    total_loss = 0
    model.train()  # 调整为训练模式（开启dropout等）
    for inputs, targets in tqdm(dataloader, desc='训练'):
        inputs = inputs.to(device)
        targets = targets.to(device)
        # 前向传播
        outputs = model(inputs)
        loss = loss_f(outputs, targets)
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def train():
    # 1.确定设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # 2.数据集
    dataloader = get_dataloader()

    # 3.词表
    with open(config.MODELS_DIR / 'vocab_list.txt', 'r', encoding='utf-8') as f:
        vocab_list = [line.strip() for line in f.readlines()]

    # 4.模型
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)

    # 5.损失函数
    loss_f = torch.nn.CrossEntropyLoss()

    # 6.优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    #7.准备tensorboard的writer
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    # 8.训练
    best_loss = float('inf')
    for epoch in range(1, 1 + config.EPOCHS):
        print('=' * 10, f" Epoch: {epoch}", '=' * 10)
        loss = train_one_epoch(model, dataloader, loss_f, optimizer, device)
        print(f"loss: {loss}")

        # 9.记录训练结果,使用tensorboard
        writer.add_scalar('loss', loss, epoch)

        #10.保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pt')
            print('模型保存成功')

    writer.close()


if __name__ == '__main__':
    train()
