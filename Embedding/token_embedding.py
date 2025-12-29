import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, padding_idx):
        # todo: 此处可以将硬编码修改为超参
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx)
        """
        :param vocab_size: 词表大小
        :param d_model: 目标的embedding维度
        """