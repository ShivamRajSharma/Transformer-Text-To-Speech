import numpy as np

def mulaw_encode(audio_samples):
    # Outputs values ranging from 0-255
    audio_converted_samples = []
    mu = 255
    for audio_sample in audio_samples:
        audio_sample = np.sign(audio_sample) * (np.log1p(mu * np.abs(audio_sample)) / np.log1p(mu))
        audio_sample = ((audio_sample + 1) / 2 * mu + 0.5)
        audio_converted_samples.append(audio_sample.astype(int))
    audio_converted_samples = np.array(audio_converted_samples)
    return audio_converted_samples


def mulaw_decode(audio_samples):
    audio_converted_samples = []
    mu = 255
    for audio_sample in audio_samples:
        audio_sample = audio_sample.astype(np.float32)
        audio_sample = 2*(audio_sample/mu) - 1
        audio_sample = np.sign(audio_sample)*(1.0 / mu)*((1.0 + mu)**(np.abs(audio_sample)) - 1.0)
        audio_converted_samples.append(audio_sample)
    audio_converted_samples = np.array(audio_converted_samples)
    return audio_converted_samples


def normalize_(mel):
        #Normalizing data between -4 and 4
        #Converges even more faster
        mel = np.clip(
            (config.scaling_factor)*((mel - config.min_db_level)/-config.min_db_level) - config.scaling_factor, 
            -config.scaling_factor, config.scaling_factor
        )
        return mel





if __name__ == "__main__":
    import librosa
    import numpy as np
    a = np.random.uniform(-1, 1, (2, 10))
    a = mulaw_encode(a)
    print(a)
    # from sklearn.preprocessing import StandardScaler
    # audio, sr = librosa.load("../Downloads/WhatsApp Ptt 2021-07-05 at 3.18.15 PM.ogg", sr=16000)
    # audio_samples = audio[None, :]
    # print(min(audio_samples[0]), max(audio_samples[0]))
    # audio_samples = mulaw_encode(audio_samples)
    # print(min(audio_samples[0]), max(audio_samples[0]))
    # audio_samples = mulaw_decode(audio_samples)
    # print(min(audio_samples[0]), max(audio_samples[0]))