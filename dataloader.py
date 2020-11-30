import config

import torch 
import torch.nn as  nn
import librosa
from librosa.feature import melspectrogram

class TransformerLoader(torch.utils.data.Dataset):
    def __init__(self, files_name, text_data, mel_transforms=None):
        self.files_name = files_name
        self.text_data = text_data
        self.mel_transforms = mel_transforms
        self.char_to_idx = pickle.load(open('input/char_to_idx.pickle', 'rb'))
    
    def __len__(self):
        return len(text_data)
    
    @staticmethod
    def data_preprocess(text):
        char_idx = [self.char_to_idx[char] for char in text if char in self.char_to_idx]
        return char_idx
    
    def __getitem__(self, idx):
        file_name = self.files_name[idx]
        text = self.text_data[idx]
        text_idx = data_preprocess(text)
        audio_path = os.path.join(CONFIG.INPUT_PATH + file_name + '.wav')
        audio_file = librosa.load(audio_path)
        mel_spect = melspectrogram(audio_file[0])
        mel_spect = librosa.power_to_db(mel_spect)
        if self.mel_transforms:
            mel_spect = self.mel_transforms(mel_spect)

        return {
            'orignal_text'  : text,
            'text_idx'      : torch.tensor(text, dtype=torch.long),
            'mel_spect'     : torch.tensor(mel_spect, dtype=torch.float) 
        }