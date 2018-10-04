# Mostly taken from: http://nbviewer.ipython.org/github/bmcfee/librosa/blob/master/examples/LibROSA%20demo.ipynb

import librosa
import matplotlib.pyplot as plt
from librosa import display
import numpy as np

# Load sound file
y, sr = librosa.load("/Users/d/Desktop/tmp_wav/b.wav")

# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power as reference.
log_S = librosa.power_to_db(S)
print("calc log s")

# Make a new figure
plt.figure(figsize=(12, 4))

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()

plt.show()
