import jieba
import torch
import config
from tokenizer import JiebaTokenizer
from model import InputMethodModel

def predict(text, device, model,tokenizer):

    #4.处理输入
    indices = tokenizer.encode(text)
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)

    #5.预测逻辑
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        top5_indices = torch.topk(outputs, 5, dim=1).indices.tolist()[0]
        top5_tokens = [tokenizer.index2word[index] for index in top5_indices]
        return top5_tokens

def run_predict():
    # 1.确定设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # 2.词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')

    # 3.模型
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
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
        prediction = predict(input_hist, device, model, tokenizer)
        print(f"输入历史: {input_hist}")
        print(prediction)

if __name__ == '__main__':
    run_predict()