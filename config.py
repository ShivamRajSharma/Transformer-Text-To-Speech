sample_rate=16000 #Provided in the paper
n_mels = 80
frame_rate = 80 #Provided in the paper
frame_length = 0.05
hop_length = int(sample_rate/frame_rate) 
win_length = int(sample_rate*frame_length)

scaling_factor = 4
min_db_level = -100

bce_weights = 7

embed_dims = 512
hidden_dims = 256 
heads = 4
forward_expansion = 4
num_layers = 4
dropout = 0.15
max_len = 1024
pad_idx = 0

Metadata = 'input/LJSpeech-1.1/metadata.csv'
Audio_file_path = 'input/LJSpeech-1.1/wavs/'

Model_Path = 'model/model.bin'
checkpoint = 'model/checkpoint.bin'

Batch_Size = 2
Epochs = 40
LR = 3e-4
warmup_steps = 0.2