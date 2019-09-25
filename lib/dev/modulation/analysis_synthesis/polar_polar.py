## FILE:           polar_polar.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Analysis and synthesis with the polar representation in the acoustic-domain and the polar representation in the modulation-domain.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import tensorflow as tf

def analysis(x, N_w, N_s, NFFT, K_w, K_s, NFFT_m):
    '''
    Polar form acoustic-domain and polar form modulation-domain analysis.

    Input/s:
        x - noisy speech.
        N_w - time-domain window length (samples).
        N_s - time-domain window shift (samples).
        NFFT - acoustic-domain DFT components.
        K_w - acoustic-domain window length (samples).
        K_s - acoustic-domain window shift (samples).
        NFFT_m - modulation-domain DFT components.

    Output/s:
        MM, P, & MP spectrums.
    '''

    ## M & P SPECTRUMS (ACOUSTIC DOMAIN)
    x_DFT = tf.transpose(tf.signal.stft(x, N_w, N_s, NFFT, pad_end=True), perm=[0,2,1])
    x_MAG = tf.abs(x_DFT); x_PHA = tf.angle(x_DFT)

    ## MM & MP SPECTRUMS (MODULATION DOMAIN)
    x_MAG_DFT = tf.transpose(tf.signal.stft(x_MAG, K_w, K_s, NFFT_m, pad_end=True), perm=[0,2,1,3])
    x_MAG_MAG = tf.abs(x_MAG_DFT); x_MAG_PHA = tf.angle(x_MAG_DFT)

    ## MM, MP, & P SPECTRUMS
    return x_MAG_MAG, x_MAG_PHA, x_PHA

def synthesis(y_MAG_MAG, x_MAG_PHA, x_PHA, N_w, N_s, NFFT, K_w, K_s, NFFT_m):
    '''
    Polar form acoustic-domain and polar form modulation-domain synthesis.

    Input/s:
        y_MAG_MAG - modified MM spectrum.
        x_MAG_PHA - unmodified MM spectrum.
        x_PHA - unmodified P spectrum.
        N_w - time-domain window length (samples).
        N_s - time-domain window shift (samples).
        NFFT - acoustic-domain DFT components.
        K_w - acoustic-domain window length (samples).
        K_s - acoustic-domain window shift (samples).
        NFFT_m - modulation-domain DFT components.

    Output/s:
        synthesised signal.
    '''

    ## MAGNITUDE SPECTRUM (ACOUSTIC DOMAIN)
    y_MAG_DFT = tf.transpose(tf.cast(y_MAG_MAG, tf.complex64)*tf.exp(1j*tf.cast(x_MAG_PHA, tf.complex64)), perm=[0,2,1,3])
    y_MAG = tf.signal.inverse_stft(y_MAG_DFT, K_w, K_s, NFFT_m, tf.signal.inverse_stft_window_fn(K_s))

    ## SYNTHESISED SIGNAL
    y_DFT = tf.transpose(tf.cast(y_MAG[:,:,0:tf.shape(x_PHA)[-1]], tf.complex64)*tf.exp(1j*tf.cast(x_PHA, tf.complex64)), perm=[0,2,1])
    return tf.signal.inverse_stft(y_DFT, N_w, N_s, NFFT, tf.signal.inverse_stft_window_fn(N_s))