import config
import dataloader
import engine

import gc
import torch 
import torch.nn as nn 
from tqdm import tqdm 

def preprocess(data):
    text_data, audio_files_name = [], []
    for d in data:
        audio_name, text = d.split('|')[:2]
        text_data.append(text)
        audio_files_name.append(audio_name)
    return text_data, audio_files_name

def train():
    data = open(config.Metadata).read().strip().split('\n')
    text_data, audio_file_name = preprocess(data)
    del data
    gc.collect()
    transforms  = 

