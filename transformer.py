import torch
import torch.nn as nn

from Decoder.decoder_layer import Decoder
from Encoder.encoder_layer import Encoder


class Transformer(nn.Module):
    def __init__(self, s_padding_idx, t_padding_idx, s_vocab_size, t_vocab_size,
                 max_len, d_model, n_head, ffn_hidden, n_layers, dropout_prob, device):
        """
        :param s_padding_index: 原始状态padding索引值
        :param t_padding_index: 目标状态padding索引值
        :param s_vocab_size: 原始状态词表大小
        :param t_vocab_size: 目标状态词表大小
        :param max_len: 最大长度
        :param d_model: 词嵌入维度
        :param n_head: 多头数量
        :param ffn_hidden: 前馈神经网络隐层维度
        :param n_layers: 编码解码器的堆叠次数
        :param dropout_prob: dropout概率
        :param device: 设备
        """
        super(Transformer, self).__init__()

        self.s_padding_idx = s_padding_idx
        self.t_padding_idx = t_padding_idx
        self.device = device

        self.encoder = Encoder(s_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layers, dropout_prob, device, s_padding_idx)
        self.decoder = Decoder(t_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layers, dropout_prob, device, t_padding_idx)


    def make_s_mask(self, s):
        """
        为encoder生成mask，用于忽略padding的部分
        :param s: [batch_size, s_len]
        :return: [batch_size, 1, 1, s_len]
        attention中的维度（分头后）是：[batch_size, n_head, max_len, d_model]，将两个1自动拉伸至需要的数值
        """

        # bool矩阵，不等于padding的部分为True，padding的部分为False
        s_mask = (s != self.s_padding_idx)      # [batch_size, s_len]
        # 拓展维度，与分头后的attention进行广播应用至所有对应维度上
        s_mask = s_mask.unsqueeze(1).unsqueeze(2)

        return s_mask

    def make_t_mask(self, t):
        """
        1. 忽略padding
        2. 以下三角矩阵掩盖“答案”，预测t只能看到0~t-1的位置
        :param t: [batch_size, t_len]
        """
        t_pad_mask = (t != self.t_padding_idx).unsqueeze(1).unsqueeze(2)
        t_len = t.shape[1]

        # 生成全为1的下三角矩阵
        t_sub_mask = torch.tril(torch.ones((t_len, t_len), device=self.device)).bool()

        # 矩阵既不是padding的部分，也要位于当前预测位置之前
        t_mask = t_pad_mask & t_sub_mask

        return t_mask

    def forward(self, s, t):
        """
        :param s: 原始序列[batch_size, s_len]
        :param t: 目标序列[batch_size, t_len]
        """

        s_mask = self.make_s_mask(s)
        t_mask = self.make_t_mask(t)

        enc_output = self.encoder(s, s_mask)
        output = self.decoder(enc_output, t, t_mask, s_mask)

        return output


# --- 测试代码 ---
if __name__ == '__main__':
    # 定义超参数
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 100
    trg_vocab_size = 100
    max_len = 50
    d_model = 512
    n_head = 8
    ffn_hidden = 2048
    n_layers = 6
    drop_prob = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型
    model = Transformer(src_pad_idx, trg_pad_idx, src_vocab_size, trg_vocab_size,
                        max_len, d_model, n_head, ffn_hidden, n_layers, drop_prob, device).to(device)

    # 模拟输入数据
    # 假设 Batch Size = 2, Src Len = 10, Trg Len = 12 (不同长度)
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0, 0],
                      [1, 8, 7, 3, 4, 5, 6, 7, 2, 0]]).to(device)  # 0 是 padding
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0, 0, 0, 0, 0],
                        [1, 5, 6, 2, 4, 7, 6, 2, 3, 8, 5, 0]]).to(device)

    # 前向传播
    out = model(x, trg)

    print(f"输入 shape: {x.shape}")  # [2, 10]
    print(f"目标 shape: {trg.shape}")  # [2, 12]
    print(f"输出 shape: {out.shape}")  # [2, 12, 100] (100 是 trg_vocab_size)
    print("模型运行成功！")