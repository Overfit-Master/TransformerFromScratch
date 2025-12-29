import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 将多头拼接后的结果映射回d_model
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, q, k, v, mask = None):
        # 输入维度应该为[batch_size, max_len, d_model]，即qkv的维度
        batch_size, max_len, _ = q.size()

        # 线性变换得到真正参与attention计算的QKV
        # [batch_size, max_len, d_model]
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # 分头
        # [batch_size, max_len, d_model] --> [batch_size, max_len, n_head, head_dim]
        # attention_score的矩阵计算是在序列内，每个头单独进行，所以需要进行维度变换
        # [batch_size, max_len, n_head, head_dim] --> [batch_size, n_head, max_len, head_dim]
        Q = Q.view(batch_size, self.n_head, max_len, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, self.n_head, max_len, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, self.n_head, max_len, self.head_dim).transpose(1, 2)

        # 注意力得分的计算
        attention_score = (Q @ K.transpose(-1, -2)) / math.sqrt(self.head_dim)      # 注意此处是每个头单独进行所以是head_dim而不是d_model

        if mask is not None:
            # 以极小的值填充保证softmax后接近0
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        attention_weights = self.softmax(attention_score)
        attention_score = attention_weights @ V     # [batch_size, n_head, max_len, head_dim]

        # 拼接多头
        # 先变换回原始维度，再通过contiguous保证内存空间的连续，不影响view的拼接操作
        attention_score = attention_score.transpose(1, 2).contiguous()
        attention_score = attention_score.view(batch_size, max_len, self.d_model)

        # 线性变换得到最终输出
        output = self.w_combine(attention_score)
        return output, attention_weights
