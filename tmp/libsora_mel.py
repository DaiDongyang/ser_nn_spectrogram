import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display


def get_mel_spectrogram(y, sr, n_fft, win_length, hop_length, power=2, window='hamming',
                            n_mels=128):
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                      window=window)) ** power
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=n_mels)
    return np.dot(mel_basis, spectrogram)


def draw_log_mel_spectrogram():
    filepath = "/Users/d/Desktop/wavs2/Ses01F_impro01_F014.wav"
    y, sr = librosa.load(filepath)
    n_fft = 1024
    win_length = int(0.04 * sr)
    hop_length = int(0.01 * sr)
    print('sr:', sr)
    print('win_length:', win_length)
    print('hop_length:', hop_length)
    mel_spec1 = get_mel_spectrogram(y, sr, n_fft=1024, win_length=win_length, hop_length=hop_length)
    mel_spec2 = get_mel_spectrogram(y, sr, n_fft=1600, win_length=win_length, hop_length=hop_length)
    log_S1 = librosa.core.power_to_db(mel_spec1)
    log_S2 = librosa.core.power_to_db(mel_spec2)
    print('max log_s1', np.max(log_S1), 'min log_s1', np.min(log_S1))
    print('max log_s2', np.max(log_S2), 'min log_s2', np.min(log_S2))
    print(np.max(log_S1-log_S2))
    print(np.min(log_S1-log_S2))
    # mel_spectrogram = get_mel_spectrogram(y, sr, n_fft, win_length, hop_length)
    # log_S = librosa.core.power_to_db(mel_spectrogram)
    ## Make a new figure
    # plt.figure(figsize=(12, 4))
    #
    # # Display the spectrogram on a mel scale
    # # sample rate and hop length parameters are used to render the time axis
    # display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    #
    # # Put a descriptive title on the plot
    # plt.title('mel power spectrogram')
    #
    # # draw a color bar
    # plt.colorbar(format='%+02.0f dB')
    #
    # # Make the figure layout compact
    # plt.tight_layout()
    #
    # plt.show()


if __name__ == '__main__':
    draw_log_mel_spectrogram()
