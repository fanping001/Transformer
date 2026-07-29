from dm02_encoder import * 
from dm03_decoder import * 
from dm04_generator import * 


class EncoderDecoder(nn.Module):
    def __init__(self, encoder,decoder,src_embed,tgt_embed,generator):
        super().__init__()
        # 编码器对象
        self.encoder = encoder
        # 解码器对象
        self.decoder = decoder
        # 编码器输入部分对象：wordEmbedding+positionEncoding    
        self.src_embed = src_embed
        # 解码器输入部分对象：wordEmbedding+positionEncoding    
        self.tgt_embed = tgt_embed
        # 输出部分对象
        self.generator = generator

    def forward(self,source,target,source_mask,target_mask):
        # source代表原始编码器的输入：比如[2,4]
        # target 代表原始解码器的输入：比如[2，4]
        # source_mask本质代表是padding一mask，[8,4,4]用在编码器的多头自注意力计算以及，解码器第二个子层的多头注意力计算
        # target_mask本质代表是sentence-mask，[8,4,4]用在解码器的第一个子层的多头自注意力计算
        # 将source送入编码器输入部分对象
        embed_x = self.src_embed(source) # [2,4,512]
        # 将embed_x送入encoder
        encoder_result = self.encoder(embed_x,source_mask)# [2,4,512]
        # 将target送入编码器输入部分对象wordEmbedding+positionEncoding 
        decoder_embed_x = self.tgt_embed(target)
        # 将数据送入解码器，得到解码结果
        decoder_result = self.decoder(decoder_embed_x,encoder_result,source_mask,target_mask)
        # 将解码器的输出送入输出层
        output = self.generator(decoder_result)
        return output
    
# 定义测试函数
def make_model(source_vocab,target_vocab,N = 6,d_model=512,d_ff=1024,head=8,dropout_p=0.1):
    # 得到一个深拷贝函数对象
    c = copy.deepcopy
    # 实例化多头注意力机制对象
    attn = MutiHeadAttention(embed_dim=d_model,head=head,dropout_p=dropout_p)
    # 实例化前馈全连接层对象
    ff = FeedForward(d_model=d_model,d_ff=d_ff,dropout=dropout_p)
    # 文本嵌入层实例化对象:编码器
    encode_embed = Embeddings(vocab_size=source_vocab,d_model=d_model)
    # 位置编码实例化对象
    position = PositionEncoding(d_model=d_model,dropout=dropout_p,max_len=2000)
    # 文本嵌入层实例化对象：解码器
    decode_embed = Embeddings(vocab_size=target_vocab,d_model=d_model)
    # 实例化输出层
    generator = Generator(d_model=d_model,vocab_size=target_vocab)
    # 实例化encoder_decoder模型对象
    model = EncoderDecoder(encoder=Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout_p),N),
                   decoder=Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout_p),N),
                   src_embed=nn.Sequential(encode_embed,c(position)),
                   tgt_embed=nn.Sequential(decode_embed,c(position)),
                   generator=generator)
    
    # 对参数进行初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

if __name__ =="__main__":
    model = make_model(source_vocab=1000,target_vocab=1000)
    # print(model)
    source = torch.tensor([[1,2,3,4],[2,4,6,8]])
    target = torch.tensor([[5,3,3,4],[2,40,6,80]])
    source_mask = target_mask = torch.zeros(8,4,4)
    result =model(source,target,source_mask,target_mask)
    print(f"transformer的输出结果-->{result.shape}")