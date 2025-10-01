import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, filtfilt

def bandpass_filter(X, lowcut, highcut, sampling_frequency=30., order=2):
    b, a = butter(order, [lowcut, highcut], fs=sampling_frequency, btype='band')
    return filtfilt(b, a, X)


def generate_white_noise(output_file, sample_rate=44100, duration=1):

    # Generate white noise
    noise = np.random.normal(0, 1, sample_rate * duration)

    # Filter White Noise
    noise = bandpass_filter(noise, lowcut=2, highcut=20000, sampling_frequency=sample_rate)

    # Normalize the white noise
    noise = noise / np.max(np.abs(noise))

    # Convert the white noise to a 16-bit format
    noise = (noise * 2**15).astype(np.int16)

    # Save the white noise as a .wav file
    write(output_file + '.wav', sample_rate, noise)


output_file = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\White_Noise_Sample"
generate_white_noise(output_file)