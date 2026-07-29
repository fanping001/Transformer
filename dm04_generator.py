import torch
import torch.nn as nn
import torch.nn.functional as F
from dm03_decoder import * 

class Generator(nn.Module):
    def __init__(self, d_model,vocab_size):
        super().__init__()
        # d_model -->词嵌入维度：vocab_size--》解码器端单词的总个数
        # 定义输出层
        self.out = nn.Linear(d_model,vocab_size)
    
    def forward(self,x):
        return F.log_softmax(self.out(x),dim=-1)
    
def test_generator():
    decoder_result = test_decoder()
    # 实例化输出层对象
    my_generator = Generator(d_model=512,vocab_size=1000)   
    output = my_generator(decoder_result)
    print(f"模型输出结果是-->{output.shape}")
    
if __name__ == "__main__":
    test_generator()