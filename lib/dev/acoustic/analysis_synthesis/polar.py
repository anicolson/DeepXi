## FILE:           polar.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Analysis and synthesis with the polar representation in the acoustic-domain.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import tensorflow as tf

def analysis(x, N_w, N_s, NFFT):
    '''
    Polar form acoustic-domain analysis.

    Input/s:
        x - noisy speech.
        N_w - time-domain window length (samples).
        N_s - time-domain window shift (samples).
        NFFT - acoustic-domain DFT components.

    Output/s:
        Magnitude and phase spectrums.
    '''

    ## MAGNITUDE & PHASE SPECTRUMS (ACOUSTIC DOMAIN)
    x_DFT = tf.signal.stft(x, N_w, N_s, NFFT, pad_end=True)
    x_MAG = tf.abs(x_DFT); x_PHA = tf.angle(x_DFT)
    return x_MAG, x_PHA

def synthesis(y_MAG, x_PHA, N_w, N_s, NFFT):
    '''
    Polar form acoustic-domain synthesis.

    Input/s:
        y_MAG - modified nagnitude spectrum.
        x_PHA - unmodified phase spectrum.
        N_w - time-domain window length (samples).
        N_s - time-domain window shift (samples).
        NFFT - acoustic-domain DFT components.

    Output/s:
        synthesised signal.
    '''

    ## SYNTHESISED SIGNAL
    y_DFT = tf.cast(y_MAG, tf.complex64)*tf.exp(1j*tf.cast(x_PHA, tf.complex64))
    return tf.signal.inverse_stft(y_DFT, N_w, N_s, NFFT, tf.signal.inverse_stft_window_fn(N_s))