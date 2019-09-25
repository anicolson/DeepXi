## FILE:           cart_polar.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Analysis and synthesis with the cartesian representation in the acoustic-domain and the polar representation in the modulation-domain.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import tensorflow as tf

def analysis(x, N_w, N_s, NFFT, K_w, K_s, NFFT_m):
    '''
    Cartesian form acoustic-domain and polar form modulation-domain analysis.

    Input/s:
        x - noisy speech.
        N_w - time-domain window length (samples).
        N_s - time-domain window shift (samples).
        NFFT - acoustic-domain DFT components.
        K_w - acoustic-domain window length (samples).
        K_s - acoustic-domain window shift (samples).
        NFFT_m - modulation-domain DFT components.

    Output/s:
        RR, RI, IR, and II spectrums.
    '''

    ## R & I SPECTRUMS (ACOUSTIC DOMAIN)
    x_DFT = tf.transpose(tf.signal.stft(x, N_w, N_s, NFFT, pad_end=True), perm=[0,2,1])
    x_REAL = tf.math.real(x_DFT); x_IMAG = tf.math.imag(x_DFT)

    ## RM & RP SPECTRUMS (MODULATION DOMAIN)
    x_REAL_DFT = tf.transpose(tf.signal.stft(x_REAL, K_w, K_s, NFFT_m, pad_end=True), perm=[0,2,1,3])
    x_REAL_MAG = tf.abs(x_REAL_DFT); x_REAL_PHA = tf.angle(x_REAL_DFT)

    ## IM & IP SPECTRUMS (MODULATION DOMAIN)
    x_IMAG_DFT = tf.transpose(tf.signal.stft(x_IMAG, K_w, K_s, NFFT_m, pad_end=True), perm=[0,2,1,3])    
    x_IMAG_MAG = tf.abs(x_IMAG_DFT); x_IMAG_PHA = tf.angle(x_IMAG_DFT)

    ## RM, RP, IM, & IP SPECTRUMS
    return x_REAL_MAG, x_REAL_PHA, x_IMAG_MAG, x_IMAG_PHA

def synthesis(y_REAL_MAG, x_REAL_PHA, y_IMAG_MAG, x_IMAG_PHA, N_w, N_s, NFFT, K_w, K_s, NFFT_m):
    '''
    Cartesian form acoustic-domain and polar form modulation-domain synthesis.

    Input/s:
        y_REAL_MAG - modified RM spectrum.
        x_REAL_PHA - unmodified RP spectrum.
        y_IMAG_MAG - modified IM spectrum.
        x_IMAG_PHA - unmodified IP spectrum.
        N_w - time-domain window length (samples).
        N_s - time-domain window shift (samples).
        NFFT - acoustic-domain DFT components.
        K_w - acoustic-domain window length (samples).
        K_s - acoustic-domain window shift (samples).
        NFFT_m - modulation-domain DFT components.

    Output/s:
        synthesised signal.
    '''

    ## REAL SPECTRUM (ACOUSTIC DOMAIN)
    y_REAL_DFT = tf.transpose(tf.cast(y_REAL_MAG, tf.complex64)*tf.exp(1j*tf.cast(x_REAL_PHA, tf.complex64)), perm=[0,2,1,3])
    y_REAL = tf.signal.inverse_stft(y_REAL_DFT, K_w, K_s, NFFT_m, tf.signal.inverse_stft_window_fn(K_s))

    ## IMAGINARY SPECTRUM (ACOUSTIC DOMAIN)
    y_IMAG_DFT = tf.transpose(tf.cast(y_IMAG_MAG, tf.complex64)*tf.exp(1j*tf.cast(x_IMAG_PHA, tf.complex64)), perm=[0,2,1,3])
    y_IMAG = tf.signal.inverse_stft(y_IMAG_DFT, K_w, K_s, NFFT_m, tf.signal.inverse_stft_window_fn(K_s))

    ## SYNTHESISED SIGNAL
    y_DFT = tf.transpose(tf.dtypes.complex(y_REAL, y_IMAG), perm=[0,2,1])
    return tf.signal.inverse_stft(y_DFT, N_w, N_s, NFFT, tf.signal.inverse_stft_window_fn(N_s))