This archive contains the Free Spoken Digits Dataset:

     https://github.com/Jakobovski/free-spoken-digit-dataset

It includes 2000 recordings of the ten digits uttered by four
different speakers.  The audio clips in the wav format are in the
`recordings' directory.  File names reports the class (0, 1,..., 9),
the name of the speaker and a numerical index (0, 1, ..., 49).

Clips of indices 0, 1 and 2 form the test set, those of indices 3, 4
and 5 form the validation set, and the others are in the training set.

Feature extraction has already been performed.  The features are
spectrograms extracted from the recorded waveforms (a spectrogram
encodes the power distribution of the signal in a given time period,
and over a set of frequencies).  The figure here below shows one
example of waveform with the corresponding spectrogram.

Features are stored in the *.txt.gz files, while the files *-names.txt
list the content of the training, validation and test sets, in the
correct order.  The feature extraction script extract_features.py is
provided as reerence.
