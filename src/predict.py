import jieba
import torch
import config
from model import InputMethodModel

def predict(text, device, model,  word2index, vocab_list):

    #4.处理输入
    tokens = jieba.lcut(text)
    indexes = [word2index.get(token, word2index['<unk>']) for token in tokens]
    input_tensor = torch.tensor([indexes], dtype=torch.long).to(device)

    #5.预测逻辑
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        top5_indices = torch.topk(outputs, 5, dim=1).indices.tolist()[0]
        top5_tokens = [vocab_list[index] for index in top5_indices]
        return top5_tokens

def run_predict():
    # 1.确定设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # 2.词表
    with open(config.MODELS_DIR / 'vocab_list.txt', 'r', encoding='utf-8') as f:
        vocab_list = [line.strip() for line in f.readlines()]
    word2index = {word: index for index, word in enumerate(vocab_list)}

    # 3.模型
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt', map_location=device))

    print("输入q或quit退出")
    input_hist = ''
    while True:
        user_input = input('> ')
        if user_input in ['q', 'quit']:
            break
        if user_input.strip() == '':
            continue
        input_hist += user_input
        prediction = predict(input_hist, device, model, word2index,vocab_list)
        print(f"输入历史: {input_hist}")
        print(prediction)

if __name__ == '__main__':
    run_predict()