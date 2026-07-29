import torch
import torch.nn as nn
from dm02_encoder import * 

# 定义解码器层
class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout_p):
        super().__init__()
        # size: 词嵌入维度
        self.size = size
        # self_attn :Q=K=V 自注意力机制对象
        self.self_attn = self_attn
        # self_attn :Q=K=V 一般注意力机制
        self.src_attn = src_attn
        # feed_forward :Q=K=V 前馈全连接层对象
        self.ff = feed_forward
        # clone出三个子层链接结构
        self.sublayers = clones(SublayerConnection(size,dropout_p),3)


    def forward(self,x,memory,source_mask,target_mask):
        # x-->解码器的输入
        # memory-->编码器输出结果
        # source_mask -->>pad--mask
        # target_mask -->sentence--mask
        x1 = self.sublayers[0](x,lambda x:self.self_attn(x,x,x,target_mask))
        # 经过第二个子层连接结构
        x2 = self.sublayers[1](x1,lambda x:self.src_attn(x,memory,memory,source_mask))  
        # 经过第三个子层连接结构
        x3 = self.sublayers[2](x2,self.ff)
        return x3

# 定义解码器
class Decoder(nn.Module):
    def __init__(self,layer,N):
        super().__init__()
        # layer -->解码器层的对象：N--》N个解码器层
        # clone 出N个解码器
        self.layers = clones(layer,N)
        # 再实例化规范化层
        self.norm = LayerNorm(layer.size)
    def forward(self,x,memory,source_mask,traget_mask):
        for layer in self.layers:
            x = layer(x,memory,source_mask,traget_mask)
        return self.norm(x)

def test_decoder():
    # 获取输入
    x = torch.tensor([[10,30,1,2],[5,9,10,20]])
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
    self_attn = copy.deepcopy(my_atten)
    src_attn = copy.deepcopy(my_atten)
     # 实例化前馈全连接层对象
    my_ff = FeedForward(d_model=512,d_ff=1024)
    #获得编码器的输出结果
    encoder_result = test_encoer()
    mask = torch.zeros(8,4,4)
    source_mask = target_mask = mask
    # 实例化解码器层对象
    my_decoderlayer = DecoderLayer(size=512,self_attn=self_attn,src_attn=src_attn,feed_forward=my_ff,dropout_p=0.1)
   
    # 实例化解码器对象
    my_decoder = Decoder(layer=my_decoderlayer,N=6)
    # 将数据送入解码器
    decoder_result = my_decoder(position_x,encoder_result,source_mask,target_mask)
    print(f'decoder_result-->{decoder_result.shape}')

    return decoder_result


#测试解码器层
def test_decoderlayer():
    # 获取输入
    x = torch.tensor([[10,30,1,2],[5,9,10,20]])
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
    self_attn = copy.deepcopy(my_atten)
    src_attn = copy.deepcopy(my_atten)
     # 实例化前馈全连接层对象
    my_ff = FeedForward(d_model=512,d_ff=1024)
    #获得编码器的输出结果
    encoder_result = test_encoer()
    mask = torch.zeros(8,4,4)
    source_mask =target_mask = mask
    # 实例化解码器层对象
    my_decoderlayer = DecoderLayer(size=512,self_attn=self_attn,src_attn=src_attn,feed_forward=my_ff,dropout_p=0.1)
    results = my_decoderlayer(position_x,encoder_result,source_mask,target_mask)
    print(f"解码器层的输出层结果-->{results.shape}")




if __name__ =="__main__":
    # test_decoderlayer()
    test_decoder()