from dataclasses import dataclass

@dataclass
class TransConfig:
    
    d_model:int = 512
    d_ff:int = 1024
    head_size:int = 8
    
    src_vocab_size:int = 300
    tgt_vocab_size:int = 300
    
    n_blocks:int = 8
    dropout:float = 0.2
    eps:float = 1e-6
    
    src_seq_len:int = 540
    tgt_seq_len:int = 170
    tokenizer_file:str = "tokenizer_{}.json"
    
    batch_size:int = 8
    workers:int = 0
    lr:float = 3e-4


    
    
    