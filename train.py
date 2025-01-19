import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from typing import List
from tqdm import tqdm

from model import (Transformer, 
                   TransConfig)

from dataset import get_dataloader

def train(config:TransConfig, 
          model:nn.Module, 
          dataloader:DataLoader, 
          optimizer:optim, 
          loss_fn:nn, 
          device:str) -> List[float | int]:
    
    model.train()
    losses = []
    
    for batch in (pbar := tqdm(dataloader, desc="Training")):
        
        #load to device
        encoder_input = batch["encoder_input"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        decoder_mask = batch["decoder_mask"].to(device)
        label = batch["label"].to(device)
        
        pred = model(encoder_input, decoder_input, decoder_mask)
        
        loss = loss_fn(pred.view(-1, config.tgt_vocab_size), label.view(-1))
        
        pbar.set_postfix(Loss=loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        
    return sum(losses) / len(dataloader)

if __name__ == "__main__":
    
    config = TransConfig()
    train_dataloader, test_dataloader, src_tokenizer, tgt_tokenizer = get_dataloader(config)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    config.src_vocab_size = src_tokenizer.get_vocab_size()
    config.tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    
    model = Transformer(config)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    train(config, model, train_dataloader, optimizer, loss_fn, device)
    
    


    
    

