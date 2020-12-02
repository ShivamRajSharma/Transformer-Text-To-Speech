import config

import os
import pickle
import torch 
import torch.nn as  nn
import numpy as np
import librosa
from librosa.feature import melspectrogram
from torch.nn.utils.rnn import pad_sequence


class TransformerLoader(torch.utils.data.Dataset):
    def __init__(self, files_name, text_data, mel_transforms=None, normalize=False):
        self.files_name = files_name
        self.text_data = text_data
        self.transforms = mel_transforms
        self.normalize = normalize
        self.char_to_idx = pickle.load(open('input/char_to_idx.pickle', 'rb'))
    
    def __len__(self):
        return len(self.text_data)
    

    def data_preprocess_(self, text):
        char_idx = [self.char_to_idx[char] for char in text if char in self.char_to_idx]
        return char_idx
    

    def normalize_(self, mel):
        #Normalizing data between -4 and 4 
        #Converges even more faster
        mel = np.clip(
            (config.scaling_factor)*((mel - config.min_db_level)/-config.min_db_level) - config.scaling_factor, 
            -config.scaling_factor, config.scaling_factor
        )
        return mel

    
    def __getitem__(self, idx):
        file_name = self.files_name[idx]
        text = self.text_data[idx]
        text_idx = self.data_preprocess_(text)
        audio_path = os.path.join(config.Audio_file_path + file_name + '.wav')

        audio_file, _ = librosa.load(
            audio_path,
            sr=config.sample_rate
            )

        audio_file, _ = librosa.effects.trim(audio_file)

        mel_spect = melspectrogram(
            audio_file,
            sr=config.sample_rate,
            n_mels=config.n_mels,
            hop_length=config.hop_length,
            win_length=config.win_length
        )
        
        pre_mel_spect = np.zeros((1, config.n_mels))
        mel_spect = librosa.power_to_db(mel_spect).T
        mel_spect = np.concatenate((pre_mel_spect, mel_spect), axis=0)

        if self.normalize:
            mel_spect = self.normalize_(mel_spect)

        mel_spect = torch.tensor(mel_spect, dtype=torch.float)
        mel_mask = [1]*mel_spect.shape[0]

        end_logits = [0]*(len(mel_spect) - 1)
        end_logits += [1]

        if self.transforms:
            for transform in self.transforms:
                if np.random.randint(0, 11) == 10:
                    mel_spect = transform(mel_spect).squeeze(0)

        return {
            'original_text'  : text,
            'mel_spect'     : mel_spect,
            'mel_mask'      : torch.tensor(mel_mask, dtype=torch.long),
            'text_idx'      : torch.tensor(text_idx, dtype=torch.long),
            'end_logits'    : torch.tensor(end_logits, dtype=torch.float),
        }


class MyCollate:
    def __init__(self, pad_idx, spect_pad):
        self.pad_idx = pad_idx
        self.spect_pad =spect_pad
    
    def __call__(self, batch):
        text_idx = [item['text_idx'] for item in batch]
        padded_text_idx = pad_sequence(
            text_idx,
            batch_first=True,
            padding_value=self.pad_idx
        )
        end_logits = [item['end_logits'] for item in batch]
        padded_end_logits  = pad_sequence(
            end_logits,
            batch_first=True,
            padding_value=0
        )
        original_text = [item['original_text'] for item in batch]
        mel_mask = [item['mel_mask'] for item in batch]
        padded_mel_mask = pad_sequence(
            mel_mask,
            batch_first=True,
            padding_value=0
        )
        mel_spects = [item['mel_spect'] for item in batch]

        batch_size, max_len = padded_mel_mask.shape

        padded_mel_spect = torch.zeros(batch_size, max_len, mel_spects[0].shape[-1])

        for num,mel_spect in enumerate(mel_spects):
            padded_mel_spect[num, :mel_spect.shape[0]] = mel_spect
        
        return {
            'original_text'  : original_text,
            'mel_spect'     : padded_mel_spect,
            'mel_mask'      : padded_mel_mask,
            'text_idx'      : padded_text_idx,
            'end_logits'    : padded_end_logits
        }


