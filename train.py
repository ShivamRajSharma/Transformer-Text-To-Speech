import warnings
warnings.filterwarnings('ignore')

import config
import dataloader
import engine

import sys

import os
import gc
import transformers
import torch 
import torch.nn as nn 
import torchaudio
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
from Transformer_tts_model.TransformerTTSModel import TransformerTTS

def preprocess(data):
    text_data, audio_files_name = [], []
    for d in data:
        audio_name, text = d.split('|')[:2]
        text_data.append(text.lower())
        audio_files_name.append(audio_name)
    return text_data, audio_files_name

def train():
    data = open(config.Metadata).read().strip().split('\n')[:10]
    text_data, audio_file_name = preprocess(data)
    del data
    gc.collect()
    transforms = [
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35)
    ]
    
    train_text_data, val_text_data, train_audio_file_name, val_audio_file_name = train_test_split(
        text_data, 
        audio_file_name, 
        test_size=0.2
        )

    train_data = dataloader.TransformerLoader(
        files_name=train_audio_file_name,
        text_data=train_text_data,
        mel_transforms=transforms,
        normalize=True
    )

    val_data = dataloader.TransformerLoader(
        files_name=val_audio_file_name,
        text_data=val_text_data,
        normalize=True
    )

    pad_idx = 0


    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.Batch_Size,
        num_workers=1,
        pin_memory=True,
        collate_fn=dataloader.MyCollate(
            pad_idx=pad_idx, 
            spect_pad=-config.scaling_factor
        )
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=config.Batch_Size,
        num_workers=1,
        pin_memory=True,
        collate_fn=dataloader.MyCollate(
            pad_idx=pad_idx, 
            spect_pad=-config.scaling_factor
        )
    )

    vocab_size = len(train_data.char_to_idx) + 1

    model = TransformerTTS(
        vocab_size=vocab_size,
        embed_dims=config.embed_dims,
        hidden_dims=config.hidden_dims, 
        heads=config.heads,
        forward_expansion=config.forward_expansion,
        num_layers=config.num_layers,
        dropout=config.dropout,
        mel_dims=config.n_mels,
        max_len=config.max_len,
        pad_idx=config.pad_idx
    )
    # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    # torch.backends.cudnn.benchmark = True
    device = torch.device('cpu')
    model = model.to(device)

    optimizer = transformers.AdamW(model.parameters(), lr=config.LR)

    num_training_steps = config.Epochs*len(train_data)//config.Batch_Size

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps*num_training_steps,
        num_training_steps=num_training_steps
    )

    epoch_start = 0

    if os.path.exists(config.checkpoint):
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch']
        print(f'---------[INFO] Restarting Training from Epoch {epoch_start} -----------\n')

         

    best_loss = 1e10 
    best_model = model.state_dict()
    print('--------- [INFO] STARTING TRAINING ---------\n')
    for epoch in range(epoch_start, config.Epochs):
        train_loss = engine.train_fn(model, train_loader, optimizer, scheduler, device)
        val_loss = engine.eval_fn(model, val_loader, device)
        print(f'EPOCH -> {epoch+1}/{config.Epochs} | TRAIN LOSS = {train_loss} | VAL LOSS = {val_loss} | LR = {scheduler.get_lr()[0]} \n')
        
        torch.save({
            'epoch'                 : epoch,
            'model_state_dict'      : model.state_dict(),
            'optimizer_state_dict'  : optimizer.state_dict(),
            'scheduler_state_dict'  : scheduler.state_dict(),
            'loss': val_loss,
            }, config.checkpoint)

        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            torch.save(best_model, config.Model_Path)


if __name__ == "__main__":
    train()