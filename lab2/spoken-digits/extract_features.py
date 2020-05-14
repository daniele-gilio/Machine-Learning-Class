import numpy as np
import scipy.io.wavfile
import scipy.signal
import os
# import matplotlib.pyplot as plt


# Each wav is padded/truncated to 4096 samples and then converted in a
# 16x64 spectrogram (16 frequencies x 64 segments).  Actually,
# spectrograms of 33 frequencies are first computed.  The first 15
# frequencies are retained, and the remaining are added together.
# Finally, spectrograms are flattened in a 1024-dimensional vector
# (the first 64 components corresponds to the first frequency, etc.).

# https://github.com/Jakobovski/free-spoken-digit-dataset

WAV_DIR = "recordings"
FILES = [f for f in os.listdir(WAV_DIR) if f.lower().endswith(".wav")]
SAMPLES = 4096
SEGMENTS = 64
FREQUENCIES = 16


def load_wav(filename):
    rate, wav = scipy.io.wavfile.read(filename)
    if wav.ndim == 2:
        wav = wav.mean(1)
    wav = wav[:SAMPLES]
    if wav.shape[0] < SAMPLES:
        wav = np.pad(wav, (0, SAMPLES - wav.shape[0]))
    return wav


def spectrogram(x):
    s = scipy.signal.spectrogram(x, nperseg=SAMPLES // SEGMENTS, noverlap=0)
    freqs, segs, y = s
    y[FREQUENCIES - 1, :] = y[FREQUENCIES - 1:, :].sum(0)
    y = y[:FREQUENCIES, :].reshape(-1)
    return y


def load_data():
    sgrams = []
    labels = []
    progressive = []
    names = []
    for f in sorted(FILES):
        x = load_wav(os.path.join(WAV_DIR, f))
        s = spectrogram(x)
        sgrams.append(s)
        # plt.imsave(f[:-4] + ".png", s.reshape(FREQUENCIES, -1), cmap="hot")
        labels.append(int(f[0]))
        progressive.append(int(f[:-4].split("_")[2]))
        names.append(f)
    X = np.stack(sgrams, 0)
    Y = np.array(labels)
    P = np.array(progressive)
    N = np.array(names)
    return X, Y, P, N


if __name__ == "__main__":
    X, Y, P,names = load_data()
    data = np.concatenate([X, Y[:, None]], 1)
    fmt = ["%.6e"] * X.shape[1] + ["%d"]

    test = data[P < 3, :]
    np.savetxt("test.txt.gz", test, fmt=fmt)
    np.savetxt("test-names.txt", names[P < 3], fmt="%s")

    idx = np.logical_and(P >= 3, P < 6)
    valid = data[idx, :]
    np.savetxt("validation.txt.gz", valid, fmt=fmt)
    np.savetxt("validation-names.txt", names[idx], fmt="%s")

    idx = (P >= 6).nonzero()[0]
    np.random.shuffle(idx)    
    train = data[idx, :]
    np.savetxt("train.txt.gz", train, fmt=fmt)
    np.savetxt("train-names.txt", names[idx], fmt="%s")
