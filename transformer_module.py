'''使用Pytorch复现transformer'''
####################################
#基础的Transformer编码器层：
#缩放点积注意力机制、多头注意力机制、前馈神经网络、
#合起来：Transformer编码器层结合以上，使用层归一化

import torch
import torch.nn as nn
import torch.nn.functional as F

#缩放点积注意力机制  
# 在多头注意力机制中，每个头都会使用缩放点积注意力机制来计算注意力
class ScaleDotProductAttention(nn.Module):
    def __init__(self,dropout = 0.1):
        super(ScaleDotProductAttention,self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,query,key,value,mask = None):
        d_k = query.size(-1)
        scores = torch.matmul(query,key.transpose(-2,-1)) / torch.sqrt(torch.tensor(d_k,dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask==0,-1e9)

        attn = F.softmax(scores,dim = -1)
        attn = self.dropout(attn)
        output = torch.matmul(attn,value)
        return output,attn

#在 Transformer 的编码器和解码器中，
# 多头注意力机制用于捕捉输入序列中不同位置之间的依赖关系。
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,d_model,dropout = 0.1):
        super(MultiHeadAttention,self).__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads#存储了每个头的特征维度大小
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
        self.attention = ScaleDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self,query,key,value,mask = None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        K = self.W_k(key).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        V = self.W_v(value).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)

        output,attn = self.attention(Q,K,V,mask)
        output = output.transpose(1,2).contiguous().view(batch_size,-1,self.num_heads * self.d_k)
        output = self.W_o(output)
        output = self.dropout(output)

        return output,attn
        
#在 Transformer 的编码器和解码器中，
# 位置前馈神经网络用于对多头注意力机制的输出进行进一步的处理。
#位置前馈神经网络，对每个位置的输入独立地进行非线性变换。
# 它由两个线性层和一个 ReLU 激活函数组成，用于增加模型的表达能力。
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout = 0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

#实现了 Transformer 的编码器层，
#通常会堆叠多个编码器层来构建一个深度的特征提取器。
class TransformerEncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout = 0.1):
        super(TransformerEncoderLayer,self).__init__()
        self.self_attn = MultiHeadAttention(num_heads,d_model,dropout)
        self.feed_forward = PositionwiseFeedForward(d_model,d_ff,dropout)
        self.norm1 = nn.LayerNorm(d_model)#层归一化
        self.norm2 = nn.LayerNorm(d_model)#层归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self,src,src_mask = None):
        attn_output,_ = self.self_attn(src,src,src,src_mask)
        src = self.norm1(src + self.dropout(attn_output))
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        return src
    
if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 32
    seq_length = 10

    encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
    src = torch.randn(batch_size, seq_length, d_model)
    src_mask = None
    output = encoder_layer(src, src_mask)
    print(output.shape)