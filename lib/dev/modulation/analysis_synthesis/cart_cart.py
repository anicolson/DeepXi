## FILE:           cart_cart.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Analysis and synthesis with the cartesian representation in the acoustic-domain and the cartesian representation in the modulation-domain.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import tensorflow as tf

def analysis(x, N_w, N_s, NFFT, K_w, K_s, NFFT_m):
    '''
    Cartesian form acoustic-domain and cartesian form modulation-domain analysis.

    Input/s:
        x - noisy speech.
        N_w - time-domain window length (samples).
        N_s - time-domain window shift (samples).
        NFFT - acoustic-domain DFT components.
        K_w - acoustic-domain window length (samples).
        K_s - acoustic-domain window shift (samples).
        NFFT_m - modulation-domain DFT components.

    Output/s:
        stacked RR, RI, IR, and II spectrums.
    '''

    ## R & I SPECTRUMS (ACOUSTIC DOMAIN)
    x_DFT = tf.transpose(tf.signal.stft(x, N_w, N_s, NFFT, pad_end=True), perm=[0,2,1])
    x_REAL = tf.math.real(x_DFT); x_IMAG = tf.math.imag(x_DFT)

    ## RR & RI SPECTRUMS (MODULATION DOMAIN)
    x_REAL_DFT = tf.transpose(tf.signal.stft(x_REAL, K_w, K_s, NFFT_m, pad_end=True), perm=[0,2,1,3])
    x_REAL_REAL = tf.math.real(x_REAL_DFT); x_REAL_IMAG = tf.math.imag(x_REAL_DFT)

    ## IR & II SPECTRUMS (MODULATION DOMAIN)
    x_IMAG_DFT = tf.transpose(tf.signal.stft(x_IMAG, K_w, K_s, NFFT_m, pad_end=True), perm=[0,2,1,3])    
    x_IMAG_REAL = tf.math.real(x_IMAG_DFT); x_IMAG_IMAG = tf.math.imag(x_IMAG_DFT)

    ## RR, RI, IR, & II SPECTRUMS
    return x_REAL_REAL, x_REAL_IMAG, x_IMAG_REAL, x_IMAG_IMAG

def synthesis(y_REAL_REAL, y_REAL_IMAG, y_IMAG_REAL, y_IMAG_IMAG, N_w, N_s, NFFT, K_w, K_s, NFFT_m):
    '''
    Cartesian form acoustic-domain and cartesian form modulation-domain synthesis.

    Input/s:
        y_REAL_REAL - modified RR spectrum.
        y_REAL_IMAG - modified RI spectrum.
        y_IMAG_REAL - modified IR spectrum.
        y_IMAG_IMAG - modified II spectrum.
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
    y_REAL_DFT = tf.transpose(tf.dtypes.complex(y_REAL_REAL, y_REAL_IMAG), perm=[0,2,1,3])
    y_REAL = tf.signal.inverse_stft(y_REAL_DFT, K_w, K_s, NFFT_m, tf.signal.inverse_stft_window_fn(K_s))

    ## IMAGINARY SPECTRUM (ACOUSTIC DOMAIN)
    y_IMAG_DFT = tf.transpose(tf.dtypes.complex(y_IMAG_REAL, y_IMAG_IMAG), perm=[0,2,1,3])
    y_IMAG = tf.signal.inverse_stft(y_IMAG_DFT, K_w, K_s, NFFT_m, tf.signal.inverse_stft_window_fn(K_s))

    ## SYNTHESISED SIGNAL
    y_DFT = tf.transpose(tf.dtypes.complex(y_REAL, y_IMAG), perm=[0,2,1])
    return tf.signal.inverse_stft(y_DFT, N_w, N_s, NFFT, tf.signal.inverse_stft_window_fn(N_s))