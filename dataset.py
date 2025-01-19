import torch
import torch.torch_version
from torch.utils.data import DataLoader, Dataset, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from model import TransConfig

# REFERENCE: https://github.com/hkproj/pytorch-transformer/blob/main/dataset.py
class BilingualDataset(Dataset):
    def __init__(self, src_dataset, tgt_dataset, tokenizer_src, tokenizer_tgt, src_seq_len, tgt_seq_len, type="train"):
        super().__init__()
        
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.type = type
        
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        
    def __len__(self):
        return len(self.src_dataset[self.type]["text"])
    
    def __getitem__(self, idx):
        src_lang = self.src_dataset[self.type]["text"][idx]
        tgt_lang = self.tgt_dataset[self.type]["text"][idx]
        
        src_token = self.tokenizer_src.encode(src_lang).ids
        tgt_token = self.tokenizer_tgt.encode(tgt_lang).ids
        
        pad_src = self.src_seq_len - len(src_token) - 2
        pad_tgt = self.tgt_seq_len - len(tgt_token) - 1
        
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(src_token, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * pad_src, dtype=torch.int64)
        ], dim=0)
        
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(tgt_token, dtype=torch.int64),
            torch.tensor([self.pad_token] * pad_tgt, dtype=torch.int64)
        ], dim=0)
        
        label = torch.cat([
            torch.tensor(tgt_token, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * pad_tgt, dtype=torch.int64)
        ], dim=0)
        
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0))
        
        return {
            "encoder_input":encoder_input,
            "decoder_input":decoder_input,
            "decoder_mask":decoder_mask,
            "label":label,
            "src_text":src_lang,
            "tgt_text":tgt_lang
        }
    
causal_mask = lambda size: torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int) == 0

def get_all_sentences(dataset):
    for item in dataset:
        yield item['text']

def get_or_build_tokenizer(config:TransConfig, dataset, lang, type):
    
    tokenizer_path = Path(config.tokenizer_file.format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
        tokenizer.train_from_iterator(get_all_sentences(dataset[type]), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataloader(config:TransConfig):
    en_ds = load_dataset("kartikagg98/HINMIX_hi-en", "lcsalign-en")
    hi_ds = load_dataset("kartikagg98/HINMIX_hi-en", "lcsalign-hicmrom")
    
    src_tokenizer = get_or_build_tokenizer(config, en_ds, "en", "train")
    tgt_tokenizer = get_or_build_tokenizer(config, hi_ds, "hi", "train")
    
    train_ds = BilingualDataset(en_ds, hi_ds, src_tokenizer, tgt_tokenizer, config.src_seq_len, config.tgt_seq_len)    
    val_ds = BilingualDataset(en_ds, hi_ds, src_tokenizer, tgt_tokenizer, config.src_seq_len, config.tgt_seq_len, type="valid")    
    
    # max_len_src = 0
    # max_len_tgt = 0
    
    # for i, j in zip(en_ds['train'], hi_ds['train']):
    #     src_ids = src_tokenizer.encode(i["text"])
    #     tgt_ids = tgt_tokenizer.encode(j["text"])
    #     max_len_src = max(max_len_src, len(src_ids))
    #     max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    # print("Max src len", max_len_src)
    # print("Max tgt len", max_len_tgt)
    
    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, num_workers=config.workers, pin_memory=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, num_workers=config.workers, pin_memory=True)
    
    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer