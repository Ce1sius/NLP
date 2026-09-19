from torch import nn

import config


class InputMethodModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM)
        self.rnn = nn.RNN(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True)
        self.Linear = nn.Linear(in_features=config.HIDDEN_SIZE,
                                out_features=vocab_size)

    def forward(self, x):
        # x.shape : (batch_size,seq_len)
        embed = self.embedding(x)
        # embed.shape : (batch_size,seq_len,embedding_dim)
        output, _ = self.rnn(embed)
        # output.shape : (batch_size,seq_len,hidden_dim)
        last_hidden_state = output[:, -1, :]
        output = self.Linear(last_hidden_state)
        return output
