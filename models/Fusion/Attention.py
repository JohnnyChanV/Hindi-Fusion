import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()

        self.linear_k = nn.Linear(dim,dim)
        self.linear_q = nn.Linear(dim,dim)
        self.linear_v = nn.Linear(dim,dim)

    def forward(self, dev,rom,add):
        dev_p = self.linear_k(dev)
        rom_p = self.linear_k(rom)
        add_p = self.linear_v(add)

        # 计算注意力权重
        attention_weights = torch.bmm(dev_p,rom_p.transpose(1, 2))  # [batch_size, query_len, context_len]
        attention_weights = F.softmax(attention_weights, dim=-1)

        # 对Context序列进行加权求和
        attended_context = torch.bmm(attention_weights, add_p)  # [batch_size, query_len, context_dim]

        return attended_context