import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import pdb
import pickle

class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder  # 编码器
        self.decoder = decoder  # 解码器
        self.src_embed = src_embed  # 输入数据的向量
        self.tgt_embed = tgt_embed  # 输出数据的向量
        self.generator = generator  # 输出数据后面的全连接+softmax

    def forward(self, src, tgt, src_mask, tgt_mask):
        "接收和处理原序列,目标序列,以及他们的mask"
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
        # 编码单元是由输入向量及其掩膜。

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        # 解码单元不仅需要输出向量与其掩膜，还需要编码单元的输出向量

class Generator(nn.Module):
    "定义标准的linear+softmax生成步骤"
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# Encoder部分
def clones(module, N):
    "产生N个相同的层"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """N层堆叠的Encoder"""

    def __init__(self, layer, N,d_model, vocab_size):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        # self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask):
        "每层layer依次通过输入序列与mask"
        break_probs = []
        group_prob = 0
        for layer in self.layers:
            x, group_prob = layer(x, mask, group_prob)
            # break_probs.append(break_prob)
        x = self.norm(x)
        # break_probs = torch.stack(break_probs, dim=1)
        return x

class LayerNorm(nn.Module):
    """构造一个layernorm模块"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        "Norm"
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Add+Norm"""

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "add norm"
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder分为两层Self-Attn和Feed Forward"""

    def __init__(self, size, self_attn, feed_forward,group_attn, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.group_attn = group_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask,group_prob):
        "Self-Attn和Feed Forward"
        group_prob= self.group_attn(x, mask, group_prob)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, group_prob,mask))
        return self.sublayer[1](x, self.feed_forward), group_prob


# Decoder部分
class Decoder(nn.Module):
    """带mask功能的通用Decoder结构"""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "将decoder的三个Sublayer串联起来"
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """
    mask后续的位置，返回[size, size]尺寸下三角Tensor
    对角线及其左下角全是1，右上角全是0
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# Attention
def attention(query, key, value, mask=None, dropout=None,group_prob=None):
    "计算Attention即点乘V"
    # print("attention!!!!")
    d_k = query.size(-1)
    # [B, h, L, L]
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        seq_len = query.size()[-2]
        b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32), 0)).cuda()
        scores = scores.masked_fill((mask | b.bool()) == 0, -1e9)

    if group_prob is not None:
        p_attn = F.softmax(scores, dim=-1)
        p_attn = p_attn * group_prob.unsqueeze(1).float()
        # print("CCCCCCCCCCCC")
    else:
        p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, group_prob=None,mask=None):
        """
        实现MultiHeadedAttention。
           输入的q，k，v是形状 [batch, L, d_model]。
           输出的x 的形状同上。
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) 这一步qkv变化:[batch, L, d_model] ->[batch, h, L, d_model/h]
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) 计算注意力attn 得到attn*v 与attn
        # qkv :[batch, h, L, d_model/h] -->x:[b, h, L, d_model/h], attn[b, h, L, L]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout,group_prob=group_prob)
        # 3) 上一步的结果合并在一起还原成原始输入序列的形状
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 最后再过一个线性层
        return self.linears[-1](x)


class GroupAttention(nn.Module):
    def __init__(self, d_model, dropout=0.8):
        super(GroupAttention, self).__init__()
        self.d_model = 256.
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        # self.linear_output = nn.Linear(d_model, d_model)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, eos_mask, prior):

        batch_size, seq_len = context.size()[:2]
        context = self.norm(context)

        a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), 1)).cuda()
        b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32), 0)).cuda()
        c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), -1)).cuda()
        tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len, seq_len], dtype=np.float32), 0)).cuda()

        # mask = eos_mask & (a+c) | b

        mask = eos_mask & (a+c)

        key = self.linear_key(context)
        query = self.linear_query(context)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model   #Q·K转置/d

        scores = scores.masked_fill(mask == 0, -1e9)
        neibor_attn = F.softmax(scores, dim=-1)
        neibor_attn = torch.sqrt(neibor_attn * neibor_attn.transpose(-2, -1) + 1e-9)
        neibor_attn = prior + (1. - prior) * neibor_attn

        t = torch.log(neibor_attn + 1e-9).masked_fill(a == 0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int() - b) == 0, 0)
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b == 0, 1e-9)
        return g_attn
        # return g_attn, neibor_attn   #C先验矩阵 每层的a序列

# Position-wise Feed-Forward Networks
class PositionwiseFeedForward(nn.Module):
    "实现FFN函数"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # return self.w_2(self.dropout(F.relu(self.w_1(x))))
        return self.w_2(self.dropout(gelu(self.w_1(x))))

# Embeddings
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model  # 表示embedding的维度

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# Positional Encoding
class PositionalEncoding(nn.Module):
    "实现PE功能"

    def __init__(self, d_model, dropout, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
# 定义一个接受超参数并生成完整模型的函数
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "根据输入的超参数构建一个模型"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    group_attn = GroupAttention(d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), group_attn,dropout), N,d_model,src_vocab),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # 使用xavier初始化参数，这个很重要
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model