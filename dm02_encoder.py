import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dm01_input import * 


# 定义一个函数：生成下三角矩阵
def subsequent_mask(size):
    temp = np.triu(m=np.ones((1,size,size)),k=1).astype('uint8')
    # print(temp)
    # 将temp上三角矩阵变成下三角矩阵
    return torch.from_numpy(1-temp)

# 定义函数：进行注意力的计算
def attention(query,key,value,mask=None,dropout = None):
    # 获取词嵌入维度
    d_k = query.size(-1)
    # 将query和key的转置进行矩阵运算
    atten_weight = torch.matmul(query,key.transpose(-1,-2)) / math.sqrt(d_k)
    # 是否需要进行掩码
    if mask is not None:
        atten_weight = atten_weight.masked_fill(mask==0,-1e9)
    # 权重分数进行softmax
    p_atten = F.softmax(atten_weight,dim=1)
    # 是否需要随机失活
    if dropout is not None:
        p_atten = dropout(p_atten)
    
    return torch.matmul(p_atten,value),p_atten



# 定义clones函数
def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 定义一个多头注意力层
class MutiHeadAttention(nn.Module):
    def __init__(self,embed_dim,head,dropout_p=0.1):
        super().__init__()
        # 确保embed_dim能够整除head
        assert embed_dim % head == 0
        # 每个头的词嵌入维度
        self.embed_dim = embed_dim
        self.d_k = embed_dim // head
        self.head = head 
        # 定义4个线性层
        self.linears = clones(nn.Linear(embed_dim,embed_dim),4)
        # 定义dropout层
        self.dropout = nn.Dropout(p=dropout_p)
        # 权重
        self.atten = None

    def forward(self,query,key,value,mask):
        # mask -->三维--> eg-->[8,4,4]
        if mask is not None:
            mask = mask.unsqueeze(0)

        # 获取batch_size
        self.batch_size = query.size(0)
        # 进行query,key,value 的多头注意力计算的变换
        # model(x) -->[2,4,512] -->[2,4,8,64]-->[2,8,4,64]
        query,key,value =   [model(x).view(self.batch_size,-1,self.head,self.d_k).transpose(1,2) for model,x in zip(self.linears,(query,key,value))]

        # 将新转换的q,k,v进行注意力的计算 --》[2,8,4,64] ;self.atten-->[2,8,4,4]
        x,self.atten = attention(query,key,value,mask=mask,dropout=self.dropout)
        
        # 将多头注意力的结果进行合并-->[2,4,512]
        atten_x = x.transpose(1,2).contiguous().view(self.batch_size,-1,self.d_k * self.head)
        return self.linears[-1](atten_x)
    

#前馈全连接层PositionwiseFeedForward实现思路分析
#1init函数（self，d_model, d_ff,dropout=0.1):
#定义线性层self.w1 self.w2，self.dropout层
#2 forward(self,x)
#数据经过self.w1（x）->F.relu()->self.dropout（）->self.w2 返回
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff, dropout=0.1):
    # d_model第1个线性层输入维度
    #d_ff第2个线性层输出维度
        super().__init__()
        #定义线性层w1w2 dropout
        self.linear1 =nn.Linear(d_model,d_ff)
        self.linear2 =nn.Linear(d_ff,d_model)
        self.dropout =nn.Dropout(p= dropout)
    def forward(self,x):
        #数据依次经过第1个线性层relu激活层dropout层，然后是第2个线性层
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
 
# 定义一个规范化层
class LayerNorm(nn.Module):
    def __init__(self,feature,eps=1e-6):
        # 这个feature是代表词嵌入维度
        super().__init__()
        # 定义系数a
        self.a = nn.Parameter(torch.ones(feature))
        # 定义偏置bias
        self.b = nn.Parameter(torch.zeros(feature))
        self.eps = eps
    def forward(self,x):
        # 求出x的均值
        x_mean = x.mean(-1,keepdim=True)
        # 求出x的标准差
        x_std = x.std(-1,keepdim=True)
        # 进行规范化
        return self.a * (x-x_mean)/(x_std + self.eps) + self.b



# 定义一个子层连接结构
class SublayerConnection(nn.Module):
    def __init__(self,size,dropout_p=0.1):
        # size ：代表词嵌入维度
        super().__init__()
        # 实例化一个norm层
        self.norm = LayerNorm(size)
        # 定义一个dropout层
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self,x,sublayer):
        # 第一种方式：先norm
        result = x + self.dropout(sublayer(self.norm(x)))
        # 第二种方式：后norm
        # result = x + self.dropout(self.norm(sublayer(x)))
        return result

# 定义一个编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size,self_attn,feed_forward,dropout):
        super().__init__()
        # size-->词嵌入维度
        self.size = size
        # self_attn ->多头子注意力机制的对象
        self.self_attn = self_attn
        # feed_forward-->前馈全连接层的对象
        self.ff = feed_forward

        # 定义两个子层链接结构
        self.sublayers = clones(SublayerConnection(size,dropout),2)

    def forward(self,x,mask):
        # 先经过第一个子层连接结构
        x1 = self.sublayers[0](x,lambda x:self.self_attn(x,x,x,mask))
        # 再进过第二个子层链接结构
        x2 = self.sublayers[1](x1,self.ff)
        return x2

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, layer,N):
        # layer-->编码器层对象
        # N --》代表几个编码器
        super().__init__()
        # 克隆N个编码器对象
        self.layers = clones(layer,N)
        # 实例化规范化层
        self.norm = LayerNorm(layer.size)

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    

def test_encoer():
    # 定义编码器的输入：
    x = torch.tensor([[100,2,421,508],
                     [491,998,1,221]])
    # 将x送入文本嵌入层进行word embedding
    vocab_size,d_model = 1000,512
    my_embed = Embeddings(vocab_size,d_model)
    embed_x = my_embed(x)
    print(f'embedding之后的结果-->{embed_x.shape}')
    # 将embed_x送入位置编码层进行位置信息的添加
    my_posotion = PositionEncoding(d_model=d_model,dropout=0.1,max_len=60)
    position_x = my_posotion(embed_x)
    print(f'embedding+位置编码信息之后的结果-->{position_x.shape}')

    # 实例化多头注意力机制层
    embed_dim ,head,dropout_p = 512,8,0.1
    my_atten = MutiHeadAttention(embed_dim,head,dropout_p)
    mask = torch.zeros(8,4,4)
    #实例化前馈 全连接层对象
    my_ff = FeedForward(d_model=512,d_ff=1024)
    #实例化编码器层对象
    my_encoderlayer = EncoderLayer(size=512,self_attn=my_atten,feed_forward=my_ff,dropout=0.1)

    # 实例化编码器的对象
    my_encodr = Encoder(layer=my_encoderlayer,N=6)
    encoder_result = my_encodr(position_x,mask)
    # print(f"encoder_result-->{encoder_result.shape}")
    return encoder_result


def test_encoer_layer():
    # 定义编码器的输入：
    x = torch.tensor([[100,2,421,508],
                     [491,998,1,221]])
    # 将x送入文本嵌入层进行word embedding
    vocab_size,d_model = 1000,512
    my_embed = Embeddings(vocab_size,d_model)
    embed_x = my_embed(x)
    print(f'embedding之后的结果-->{embed_x.shape}')
    # 将embed_x送入位置编码层进行位置信息的添加
    my_posotion = PositionEncoding(d_model=d_model,dropout=0.1,max_len=60)
    position_x = my_posotion(embed_x)
    print(f'embedding+位置编码信息之后的结果-->{position_x.shape}')

    # 实例化多头注意力机制层
    embed_dim ,head,dropout_p = 512,8,0.1
    my_atten = MutiHeadAttention(embed_dim,head,dropout_p)
    mask = torch.zeros(8,4,4)
    #实例化前馈 全连接层对象
    my_ff = FeedForward(d_model=512,d_ff=1024)
    #实例化编码器层对象
    my_encoderlayer = EncoderLayer(size=512,self_attn=my_atten,feed_forward=my_ff,dropout=0.1)

    result =my_encoderlayer(position_x,mask)
    print(f"第一个编码器层的结果result-->{result.shape}")


def test_sublayer():
    # 定义编码器的输入：
    x = torch.tensor([[100,2,421,508],
                     [491,998,1,221]])
    # 将x送入文本嵌入层进行word embedding
    vocab_size,d_model = 1000,512
    my_embed = Embeddings(vocab_size,d_model)
    embed_x = my_embed(x)
    print(f'embedding之后的结果-->{embed_x.shape}')
    # 将embed_x送入位置编码层进行位置信息的添加
    my_posotion = PositionEncoding(d_model=d_model,dropout=0.1,max_len=60)
    position_x = my_posotion(embed_x)
    print(f'embedding+位置编码信息之后的结果-->{position_x.shape}')

    # 实例化多头注意力机制层
    embed_dim ,head,dropout_p = 512,8,0.1
    my_atten = MutiHeadAttention(embed_dim,head,dropout_p)
    mask = torch.zeros(8,4,4)
    sublayer = lambda x: my_atten(x,x,x,mask)
    # 实例化子层链接结构对象
    my_sublayer = SublayerConnection(size=512)
    result = my_sublayer(position_x,sublayer)
    print(f"第一个子层连接结构的输出结果--》{result.shape}")

def test_layernorm():
    my_attention = MutiHeadAttention(512,8)
    position_x = test_postiton()
    # 因为是子注意力机制,query=key=value
    query = key = value = position_x
    mask = torch.zeros(8,4,4)
    # 将数据送人模型得到多头注意力的结果
    atten_x = my_attention(query,key,value,mask)
    print(f'results-->{atten_x.size()}')
    # 实例化前馈全连接层对象
    my_ff = FeedForward(d_model=512,d_ff=1024)
    # 将注意力得到的结果送入前馈全连接层
    ff_x = my_ff(atten_x)
    # 实例化LayerNorm
    my_layernorm = LayerNorm(feature=512)
    results =  my_layernorm(ff_x)
    print(f'规范化后的结果results-->{results.shape}')

def ff():
    my_attention = MutiHeadAttention(512,8)
    position_x = test_postiton()
    # 因为是子注意力机制,query=key=value
    query = key = value = position_x
    mask = torch.zeros(8,4,4)
    # 将数据送人模型得到多头注意力的结果
    atten_x = my_attention(query,key,value,mask)
    print(f'results-->{atten_x.size()}')
    # 实例化前馈全连接层对象
    my_ff = FeedForward(d_model=512,d_ff=1024)
    # 将注意力得到的结果送入前馈全连接层
    results = my_ff(atten_x)
    print(f'results-->{results.shape}')

def test_mutihead():
    my_attention = MutiHeadAttention(512,8)
    position_x = test_postiton()
    # 因为是子注意力机制,query=key=value
    query = key = value = position_x
    mask = torch.zeros(8,4,4)
    # 将数据送人模型得到多头注意力的结果
    results = my_attention(query,key,value,mask)
    print(f'results-->{results.size()}')


def test_attention():
    # 获取位置编码之后的结果：（wordEmbedding+position）
    position_x = test_postiton()
    # 因为是子注意力机制,query=key=value
    query = key = value = position_x
    # 没有掩码 ：调用attention方法
    results1,p_atten = attention(query,key,value)
    print(f'results1-->{results1.shape}')
    print(f'p_atten-->{p_atten}')
    mask = torch.zeros(2,4,4)
    # 没有掩码 ：调用attention方法
    results2,p_atten2 = attention(query,key,value,mask)
    print(f'results1-->{results2}')
    print(f'p_atten-->{p_atten2}')
    mask = torch.zeros(2,4,4)


if __name__ == "__main__":
    
    # print(subsequent_mask(5))
    # test_attention()
    # test_mutihead()
    # ff()
    # test_layernorm()
    # test_sublayer()
    test_encoer()