import torch
from torch import Tensor
from torch import nn

from .config import T6Config
from .utils import FeedForwardSwiGLU, RMSNorm
from .embedding import Rotary, apply_rotary_emb

class T6Attention(nn.Module):
    """
        ---
        Tensor Product Attention
        ---
        An new Attention Named Tensor Product Attention from 
        from Tensor Product Attention Is All You Need 
        https://arxiv.org/pdf/2501.06425
    """
    def __init__(self, config:T6Config) -> None:
        super().__init__()
        
        assert config.d_model % config.head_size == 0, "d_model is must divisible head_size."
        self.config = config
        self.d_model = config.d_model
        self.d_k = config.d_model // config.head_size
        self.head_size = config.head_size
        self.q_rank = config.q_rank
        self.rank = config.rank
        
        self.W_Aq = nn.Linear(self.d_model, self.q_rank * self.head_size, bias=False)
        self.W_Ak = nn.Linear(self.d_model, self.rank * self.head_size, bias=False)
        self.W_Av = nn.Linear(self.d_model, self.rank * self.head_size, bias=False)
        
        self.W_Bq = nn.Linear(self.d_model, self.q_rank * self.d_k, bias=False)
        self.W_Bk = nn.Linear(self.d_model, self.rank * self.d_k, bias=False)
        self.W_Bv = nn.Linear(self.d_model, self.rank * self.d_k, bias=False)
        
        self.rotary_cos_sin = Rotary(self.d_k)
        self.dropout = nn.Dropout(config.dropout)
        
        self.Wo = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.apply(self._init_weigths)
        
    def _init_weigths(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        
    @staticmethod
    def attention_product(q:Tensor, k:Tensor, v:Tensor, dropout:nn.Dropout, mask:Tensor=None) -> Tensor:
        batch_size, seq_len, d_k = q.shape[0], q.shape[1], q.shape[-1]
        
        score = torch.matmul(q, k.transpose(-2, -1)) / d_k
        if mask is not None:
            # mask (1, 1, seq_len, seq_len)
            score.masked_fill_(mask==0, float("-inf"))
            
        score = dropout(score)
        score = score.softmax(dim=-1) # batch_size, head_size, seq_len, seq_len
        out = torch.matmul(score, v)
        return out, score
        
    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask:Tensor = None) -> Tensor:
        
        batch_size, seq_len, d_model = q.shape
        
        # A - Latent factor metrics
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, head_size, q_rank)
        A_q = self.W_Aq(q).view(batch_size, seq_len, self.head_size, self.q_rank)
        print(A_q.shape)
        print(self.W_Ak(k).shape)
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, head_size, rank)
        A_k = self.W_Ak(k).view(batch_size, seq_len, self.head_size, self.rank)
        A_v = self.W_Av(v).view(batch_size, seq_len, self.head_size, self.rank)
        
        # B - Latent factor metrics
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, q_rank, d_k)
        B_q = self.W_Bq(q).view(batch_size, seq_len, self.q_rank, self.d_k)
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, rank, d_k)
        B_k = self.W_Bk(k).view(batch_size, seq_len, self.rank, self.d_k)
        B_v = self.W_Bv(v).view(batch_size, seq_len, self.rank, self.d_k)
        
        #apply the rotary embedding 
        cos, sin = self.rotary_cos_sin(B_q)
        B_q, B_k = apply_rotary_emb(B_q, cos, sin), apply_rotary_emb(B_k, cos, sin)
        
        #(batch_size, seq_len, d_k, q_rank) -> (batch_size * seq_len, head_size,  q_rank)
        A_q = A_q.view(batch_size * seq_len, self.head_size, self.q_rank)
        #(batch_size, seq_len, d_k, rank) -> (batch_size * seq_len, head_size,  rank)
        A_k = A_k.view(batch_size * seq_len, self.head_size, self.rank)
        A_v = A_v.view(batch_size * seq_len, self.head_size, self.rank)
        
        #(batch_size, seq_len, q_rank, d_k) -> (batch_size * seq_len,  q_rank, d_k)
        B_q = B_q.view(batch_size * seq_len, self.q_rank, self.d_k)
        #(batch_size, seq_len, rank, d_k) -> (batch_size * seq_len,  rank, d_k)
        B_k = B_k.view(batch_size * seq_len, self.rank, self.d_k)
        B_v = B_v.view(batch_size * seq_len, self.rank, self.d_k)
        
        # (batch_size, seq_len, head_size, d_k)
        q = torch.bmm(A_q, B_q).div_(self.q_rank).view(batch_size, seq_len, self.head_size, self.d_k)
        k = torch.bmm(A_k, B_k).div_(self.rank).view(batch_size, seq_len, self.head_size, self.d_k)
        v = torch.bmm(A_v, B_v).div_(self.rank).view(batch_size, seq_len, self.head_size, self.d_k)
        
        # (batch_size, head_size, seq_len, d_k)
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)
        
        #(batch_size, head_size, seq_len, d_k)
        attention, self.attetnion_score = T6Attention.attention_product(q, k, v, self.dropout, mask=mask)
        
        out = attention.transpose(-3, -2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.Wo(out)

class T6Encoder(nn.Module):
    def __init__(self, config:T6Config):
        super().__init__()
        
        self.attetnion_norm = RMSNorm(config)
        self.attetnion = T6Attention(config)
        
        self.ff_norm = RMSNorm(config)
        self.ff = FeedForwardSwiGLU(config)
        
    def forward(self, x:Tensor):
        
        res = x
        x = self.attetnion_norm(x)
        x = res + self.attetnion(x, x, x)
        
        res = x
        x = self.ff_norm(x)
        x = res + self.ff(x)
        
        return x
    
class T6Decoder(nn.Module):
    def __init__(self, config:T6Config):
        super().__init__()
        
        self.maskattetnion_norm = RMSNorm(config)
        self.maskattetnion = T6Attention(config)
        
        self.crossattetion_norm = RMSNorm(config)
        self.crossattetion = T6Attention(config)
        
        self.ff_norm = RMSNorm(config)
        self.ff = FeedForwardSwiGLU(config)
        
    def forward(self, x:Tensor, encoder_input:Tensor, mask:Tensor=None):
        
        res = x
        x = self.maskattetnion_norm(x)
        x = res + self.maskattetnion(x, x, x, mask)
        
        res = x
        x = self.crossattetion_norm(x)
        x = res + self.crossattetion(x, encoder_input, encoder_input)
        
        res = x
        x = self.ff_norm(x)
        x = res + self.ff(x)
        
        return x
    
class T6Transformer(nn.Module):
    """
        ---
        Tensor Product Transformer
        ---
        An new Attention Named Tensor Product Attention from 
        Tensor Product Attention Is All You Need 
        https://arxiv.org/pdf/2501.06425
    """
    def __init__(self, config:T6Config):
        super().__init__()
        
        self.encoder_embedding = nn.Embedding(config.src_vocab_size, config.d_model)
        self.decoder_embedding = nn.Embedding(config.tgt_vocab_size, config.d_model)
        
        self.encoder_layers = nn.ModuleList([T6Encoder(config) for _ in range(config.n_blocks)])
        self.decoder_layers = nn.ModuleList([T6Decoder(config) for _ in range(config.n_blocks)])
        
        self.proj = nn.Linear(config.d_model, config.tgt_vocab_size)
        
    def forward(self, src:Tensor, tgt:Tensor, mask:Tensor):
        
        encoders = []
        src = self.encoder_embedding(src)
        
        for layer in self.encoder_layers:
            src = layer(src)
            encoders.append(src)
            
        tgt = self.decoder_embedding(tgt)
        for idx, layer in enumerate(self.decoder_layers):
            tgt = layer(tgt, encoders[idx], mask)
        
        out = self.proj(tgt)
        return out
    
