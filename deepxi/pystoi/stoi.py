# This code is from https://github.com/mpariente/pystoi and modified by Jason.

import numpy as np
from resampy import resample
from . import utils

# Constant definition
FS = 10000                          # Sampling frequency
N_FRAME = 256                       # Window support
NFFT = 512                          # FFT Size
NUMBAND = 15                        # Number of 13 octave band
MINFREQ = 150                       # Center frequency of 1st octave band (Hz)
OBM, CF = utils.thirdoct(FS, NFFT, NUMBAND, MINFREQ)  # Get 1/3 octave band matrix
N = 30                              # N. frames for intermediate intelligibility
BETA = -15.                         # Lower SDR bound
DYN_RANGE = 40                      # Speech dynamic range


def stoi(x, y, fs_sig, extended=False):
    """ Short term objective intelligibility
    Computes the STOI (See [1][2]) of a denoised signal compared to a
    clean signal, The output is expected to have a monotonic
    relation with the subjective speech-intelligibility, where a higher d
    denotes better speech intelligibility
    # Arguments
        x : clean original speech
        y : denoised speech
        fs_sig : sampling rate of x and y
        extended : Boolean, whether to use the extended STOI described in [3]
    # Returns
        Short time objective intelligibility measure between clean and denoised
        speech
    # Raises
        AssertionError : if x and y have different lengths
    # Reference
        [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
            Objective Intelligibility Measure for Time-Frequency Weighted Noisy
            Speech', ICASSP 2010, Texas, Dallas.
        [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
            Intelligibility Prediction of Time-Frequency Weighted Noisy Speech',
            IEEE Transactions on Audio, Speech, and Language Processing, 2011.
        [3] Jesper Jensen and Cees H. Taal, 'An Algorithm for Predicting the
            Intelligibility of Speech Masked by Modulated Noise Maskers',
            IEEE Transactions on Audio, Speech and Language Processing, 2016.
    """
    if x.shape != y.shape:
        raise Exception('x and y should have the same length,' +
                        'found {} and {}'.format(len(x), len(y)))

    # Resample is fs_sig is different than fs
    if fs_sig != FS:
        x = resample(x, fs_sig, FS)
        y = resample(y, fs_sig, FS)

    # Remove silent frames
    x, y = utils.remove_silent_frames(x, y, DYN_RANGE, N_FRAME, int(N_FRAME/2))

    # Take STFT
    x_spec = utils.stft(x, N_FRAME, NFFT, overlap=2).transpose()
    y_spec = utils.stft(y, N_FRAME, NFFT, overlap=2).transpose()

    # Apply OB matrix to the spectrograms as in Eq. (1)
    x_tob = np.sqrt(np.matmul(OBM, np.square(np.abs(x_spec))))
    y_tob = np.sqrt(np.matmul(OBM, np.square(np.abs(y_spec))))

    # Take segments of x_tob, y_tob
    x_segments = np.array(
        [x_tob[:, m - N:m] for m in range(N, x_tob.shape[1] + 1)])
    y_segments = np.array(
        [y_tob[:, m - N:m] for m in range(N, x_tob.shape[1] + 1)])

    if extended:
        x_n = utils.row_col_normalize(x_segments)
        y_n = utils.row_col_normalize(y_segments)
        return np.sum(x_n * y_n / N) / x_n.shape[0]

    else:
        # Find normalization constants and normalize
        normalization_consts = (
            np.linalg.norm(x_segments, axis=2, keepdims=True) /
            (np.linalg.norm(y_segments, axis=2, keepdims=True) + utils.EPS))
        y_segments_normalized = y_segments * normalization_consts

        # Clip as described in [1]
        clip_value = 10 ** (-BETA / 20)
        y_primes = np.minimum(
            y_segments_normalized, x_segments * (1 + clip_value))

        # Subtract mean vectors
        y_primes = y_primes - np.mean(y_primes, axis=2, keepdims=True)
        x_segments = x_segments - np.mean(x_segments, axis=2, keepdims=True)

        # Divide by their norms
        y_primes /= (np.linalg.norm(y_primes, axis=2, keepdims=True) + utils.EPS)
        x_segments /= (np.linalg.norm(x_segments, axis=2, keepdims=True) + utils.EPS)
        # Find a matrix with entries summing to sum of correlations of vectors
        correlations_components = y_primes * x_segments

        # J, M as in [1], eq.6
        J = x_segments.shape[0]
        M = x_segments.shape[1]

        # Find the mean of all correlations
        d = np.sum(correlations_components) / (J * M)
        return d
