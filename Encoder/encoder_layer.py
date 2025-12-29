import torch.nn as nn

from MultiHead_Attention.attention import MultiHeadAttention
from Normalization.layer_norm import LayerNorm
from Embedding.transformer_embedding import TransformerEmbedding
from Encoder.feed_forward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        _x = x      # 保留原始输入用于残差连接
        x, _ = self.attention(x, x, x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)      # Add & Norm

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x



# 复现多层encoder layer组成完整的encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, ffn_hidden, n_head, n_layer, dropout, device, padding_idx):
        super(Encoder, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout, device, padding_idx)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, ffn_hidden, n_head, dropout)
                for _  in range(n_layer)
            ]
        )

    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)

        return x

