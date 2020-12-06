# Transformer Text To Speech

<p align="center">
  <img src="https://miro.medium.com/max/3330/1*7HOERatJ83E1KkKr1SVAug.png" height="140" width="500" />
</p>

A text-to-speech (TTS) system converts normal language text into speech; other systems render symbolic linguistic representations like phonetic transcriptions into speech. Now with recent development in deep learning, it's possible to convert text into a human-understandable voice. For this, the text is fed into an Encoder-Decoder type Neural Network to output a Mel-Spectrogram. This Mel-Spectrogram can now be used to generate audio using the "Griffin-Lim Algorithm". But due to its disadvantage that it is not able to produce human-like speech quality, another neural net named "WaveNet" is employed, which is fed by Mel-Spectrogram to produce audio that even a human is not able to differentiate apart.
