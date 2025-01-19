import torch
from torch import Tensor
from torch import nn

from .config import TransConfig
from .embedding import PositionalEmbedding
from .utils import LayerNorm, FeedForwardGELU
import math

class Attention(nn.Module):
    def __init__(self, config:TransConfig):
        super().__init__()
        
        assert config.d_model % config.head_size == 0, "d_model must divide with head_size."
        
        self.d_model = config.d_model
        self.head_size = config.head_size
        self.d_k = self.d_model // self.head_size
        
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self.apply(self._init_weigths)
        
    def _init_weigths(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
        
    @staticmethod
    def attention_product(q:Tensor, k:Tensor, v:Tensor, dropout:nn.Dropout, mask:Tensor=None):
        d_k = q.shape[-1]
        
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            score.masked_fill_(mask==0, float("-inf"))
        score = dropout(torch.softmax(score, dim=-1))
        out = torch.matmul(score, v)
        return out, score
    
    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask:Tensor = None) -> Tensor:
        batch_size = q.shape[0]
        q_len, k_len, v_len = q.shape[1], k.shape[1], v.shape[1]
        
        query = self.W_q(q).view(batch_size, q_len, self.head_size, self.d_k)
        key = self.W_k(k).view(batch_size, k_len, self.head_size, self.d_k)
        value = self.W_v(v).view(batch_size, v_len, self.head_size, self.d_k)
        
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        attention, self.attention_score = self.attention_product(query, key, value, self.dropout, mask=mask)
        
        out = attention.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        
        return self.W_o(out)
    
class Encoder(nn.Module):
    def __init__(self, config:TransConfig):
        super().__init__()
        
        self.attention_norm = LayerNorm(config)
        self.attention = Attention(config)
        
        self.ff_norm = LayerNorm(config)
        self.ff = FeedForwardGELU(config)
        
    def forward(self, x:Tensor) -> Tensor:
        
        res = x
        x = self.attention_norm(x)
        x = res + self.attention(x, x, x)
        
        res = x
        x = self.ff_norm(x)
        x = res + self.ff(x)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, config:TransConfig):
        super().__init__()
        
        self.maskattention_norm = LayerNorm(config)
        self.maskattention = Attention(config)
        
        self.crossattention_norm = LayerNorm(config)
        self.crossattention = Attention(config)
        
        self.ff_norm = LayerNorm(config)
        self.ff = FeedForwardGELU(config)
        
    def forward(self, x:Tensor, encoder_input:Tensor, mask:Tensor=None) -> Tensor:
        
        res = x
        x = self.maskattention_norm(x)
        x = res + self.maskattention(x, x, x, mask)
        
        res = x
        x = self.crossattention_norm(x)
        x = res + self.crossattention(x, encoder_input, encoder_input)
        
        res = x
        x = self.ff_norm(x)
        x = self.ff(x)
        
        return x
    
class Transformer(nn.Module):
    def __init__(self, config:TransConfig):
        super().__init__()
        
        self.src_embedding = nn.Embedding(config.src_vocab_size, config.d_model)
        self.tgt_embedding = nn.Embedding(config.tgt_vocab_size, config.d_model)
        
        self.src_pos_embedding = PositionalEmbedding(config.d_model, config.src_seq_len, config.dropout)
        self.tgt_pos_embedding = PositionalEmbedding(config.d_model, config.tgt_seq_len, config.dropout)
        
        self.encoder_layers = nn.ModuleList([Encoder(config) for _ in range(config.n_blocks)])
        self.decoder_layers = nn.ModuleList([Decoder(config) for _ in range(config.n_blocks)])
        
        self.proj = nn.Linear(config.d_model, config.tgt_vocab_size)
        
    def forward(self, src:Tensor, tgt:Tensor, mask:Tensor):
        
        src = self.src_embedding(src)
        src = self.src_pos_embedding(src)
        
        encoders = []
        for layer in self.encoder_layers:
            src = layer(src)
            encoders.append(src)
            
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos_embedding(tgt)
        for idx, layer in enumerate(self.decoder_layers):
            tgt = layer(tgt, encoders[idx], mask)
        
        out = self.proj(tgt)
        
        return out
    

        