import torch.nn as nn
import math

from token_embedding import TokenEmbedding
from positional_embedding import PositionalEmbedding


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding( d_model, max_len, device)
        self.dropout = nn.Dropout(p=drop_prob)
        self.d_moel = d_model

    def forward(self, x):
        token_emb = self.token_embedding(x)    # 调用父类 nn.Embedding 的forward
        position_emb = self.positional_embedding(x)

        # 将 token embedding 进行 scale 后再和 positional embedding 相加
        token_emb = token_emb * math.sqrt(self.d_moel)
        return self.dropout(token_emb + position_emb)