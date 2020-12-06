# Transformer Text To Speech

<p align="center">
  <img src="https://miro.medium.com/max/3330/1*7HOERatJ83E1KkKr1SVAug.png" height="140" width="500" />
</p>

A text-to-speech (TTS) system converts normal language text into speech; other systems render symbolic linguistic representations like phonetic transcriptions into speech. Now with recent development in deep learning, its now possible to convert text into human understandable voice. For this the text is fed into an Encoder-Decoder type Nueral Network to output an Mel-Spectrogram. This mel-spectrogram can now be used to generate audio using the "Griffin-Lim Algorithm". The disadvantage of Griffin-Lim is that it is not able to produce huaman like speech quality. Therefore another nueral net named as "WaveNet" is employed which generates audio that a even a human is not able to differentiate between.
