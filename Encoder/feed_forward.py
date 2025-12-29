import torch.nn as nn
import torch.nn.functional as F

"""
FFN是有双层的全连接网络组成，先将模型维度升维至hidden再降维至d_model
两层全连接神经网络之间加入激活函数引入非线性，其公式表达如下
FFN = W2 * (ReLU(W1 * x + B1)) + B2
"""


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)      # 此处可以进行替换
        x = self.dropout(x)
        x = self.fc2(x)
        return x
