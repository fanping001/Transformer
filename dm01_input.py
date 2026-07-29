import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

# 输入部分:wordEmbedding+positionEncoding (词嵌入层，编码)
# 定义wordEmbedding层：文本嵌入层
class Embeddings(nn.Module):
    def __init__(self, vocab_size,d_model):
        super().__init__()
        # vocab_size :去重之后的单词总个数
        # d_model:词嵌入维度
        self.vocab_size = vocab_size
        self.d_model = d_model
        # 定义一个词嵌入层
        self.embed = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embed(x) * math.sqrt(self.d_model)


# 定义位置编码层
class PositionEncoding(nn.Module):
    def __init__(self, d_model,dropout,max_len=60):
        super().__init__()
        # 定义dropout 层
        self.dropout = nn.Dropout(p=dropout)
        # d_model 词嵌入表示维度
        # 定义pe，初始化一个全零的位置编码矩阵
        pe = torch.zeros(max_len,d_model)
        # print(f"pe-->{pe.shape}")
        # 定义一个位置列
        position = torch.arange(0,max_len).unsqueeze(dim=1)
        # print(f'position-->{position.shape}')
        # 定义一个转换矩阵，本质是公式里面的1/10000^(2i/d_model):^ 代表指数次方
        div_term = torch.exp(torch.arange(0,d_model,2) * -(math.log(10000)/d_model))
        # print(f"div_term-->{div_term.shape}")
        # 计算三角函数里面的值
        position_value = position * div_term
        # 进行pe的赋值
        pe[:,0::2] = torch.sin(position_value)
        pe[:,1::2] = torch.cos(position_value)
        # 将pe进行升维度
        pe = pe.unsqueeze(dim=0)
        # print(f"pe-->{pe.shape}")
        # print(f"pe-->{pe}")
        # 将pe注册到模型的缓存区，利用他但是步更新它的参数
        self.register_buffer('pe',pe)
        
        
    def forward(self,x):
        # x-->代表词嵌入层 
        # 需要将x的embededing结果和对应的位置编码结果进行融合（相加）;pe = [1,60,512]
        x = x +self.pe[:,:x.size(1)]
        return self.dropout(x)

def test_postiton():
    vocab_size = 1000
    d_model = 512
    my_embed = Embeddings(vocab_size=vocab_size,d_model=d_model)
    x = torch.tensor([[100,2,421,508],
                     [491,998,1,221]])
    embed_x = my_embed(x)
    my_position = PositionEncoding(d_model=512,dropout=0.1)
    position_x = my_position(embed_x)
    # print(f'position_x-->{position_x.shape}')
    return position_x

def plot_posstion():
    # 实例化位置编码
    my_position = PositionEncoding(d_model=20,dropout=0,max_len=100)
    y = my_position(torch.zeros(1,100,20))
    print(f"y-->{y.shape}")
    # 画图
    plt.figure(figsize=(20,20))
    plt.plot(np.arange(100),y[0,:,4:8])
    plt.legend(["dim %d" %p for p in [4,5,6,7]])
    plt.show()


if __name__ =="__main__":
    # test_postiton()
    plot_posstion()
