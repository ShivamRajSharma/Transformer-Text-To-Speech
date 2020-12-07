# Transformer Text To Speech

A text-to-speech (TTS) system converts normal language text into speech; other systems render symbolic linguistic representations like phonetic transcriptions into speech. Now with recent development in deep learning, it's possible to convert text into a human-understandable voice. For this, the text is fed into an Encoder-Decoder type Neural Network to output a Mel-Spectrogram. This Mel-Spectrogram can now be used to generate audio using the ["Griffin-Lim Algorithm"](https://paperswithcode.com/method/griffin-lim-algorithm). But due to its disadvantage that it is not able to produce human-like speech quality, another neural net named [WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio) is employed, which is fed by Mel-Spectrogram to produce audio that even a human is not able to differentiate apart.

## Model Architecture

### 1. Transformer TTS
  <img src="https://github.com/ShivamRajSharma/Transformer-Text-To-Speech/blob/main/Transformer_tts_model/model.png" height="400" width="400"/>

  * An Encoder-Decoder transformer architecture for parallel training instead for Seq2Seq training incase of [Tacotron-2](https://github.com/NVIDIA/tacotron2).
  * Text are sent as input and the model outputs a Mel-Spectrogram.
  * Multi-headed attention is employed, with causal masking only on the decoder side.
  * Paper : [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895).

### 2. Wavenet
  <img src="https://i.stack.imgur.com/t7qkv.png">*
  
  * Output of the Transformer tts (Mel-Spectrogram) is fed into the Wavenet to generate audio samples.
  * Unlike Seq2Seq models wavenet also allows parallel training.
  * Paper : [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499).
  



## Dataset Information
The model was trained on a subset of WMT-2014 English-German Dataset. Preprocessing was carried out before training the model.</br>
Dataset : https://keithito.com/LJ-Speech-Dataset/
