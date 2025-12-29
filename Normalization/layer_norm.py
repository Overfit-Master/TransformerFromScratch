import torch
import torch.nn as nn

"""
y = ((x - μ) / (σ + ε)) * γ + β
其中：

标准化：norm_x = (x - μ) / (sqrt(σ ** 2 + ε))
μ：表示输入向量在特征维度上的均值
σ：表示输入向量在特征维度上的方差
ε：极小的常数，防止除以0

仿射变换：y = norm_x * γ + β
γ：缩放参数
β：平移参数
"""


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-12):
        """
        :param d_model: 模型维度
        :param eps: 防止分母为0的稳定系数
        """
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))      # 全1表示初始不缩放
        self.beta = nn.Parameter(torch.zeros(d_model))      # 全0表示初始不平移
        self.eps = eps

    def forward(self, x):
        # 输入的数据维度为：[batch_size, max_len, d_model]
        # layernorm只在d_model的维度上进行处理
        # 标准化部分
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased除以的是N，认为数据即为整体，单纯描述数据的离散程度而不是以样本去估计整体
        variance = x.var(dim=-1, unbiased=False, keepdim=True)

        norm = (x - mean) / torch.sqrt(variance + self.eps)

        # 仿射变换
        return norm * self.gamma + self.beta


