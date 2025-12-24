import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEmbedding, self).__init__()

        # 初始化全0的矩阵
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False     # 位置编码通过硬编码得到，不需要计算梯度

        # 生成 max_len 长度的向量，并转为浮点和二维
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        # 区分奇数偶数位置进行
        # 偶数 = sin(pos / (10000^(2i/d_model)))
        # 奇数 = cos(pos / (10000^(2i/d_model)))
        _2i = torch.arange(0, d_model, step=2, device=device).float()       # 注意此处的i指的是词嵌入层维度

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))


    def forward(self, x):
        batch_size, sequence_len = x.size()
        # 位置编码是二维的，每一行代表一个位置的编码结果，只返回有效位数的结果
        return self.encoding[:sequence_len, :]


if __name__ == '__main__':
    pos = torch.arange(0, 100, device="cuda")
    print(pos)
    pos = pos.float().unsqueeze(dim=1)
    print(pos)