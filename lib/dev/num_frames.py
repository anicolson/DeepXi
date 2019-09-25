## FILE:           nframes.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Detirmines number of frames in a signal.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import tensorflow as tf

def num_frames(N, N_s):
	'''
	Returns the number of frames for a given sequence length, and
	frame shift.

	Inputs:
		N - sequence length (samples).
		N_s - frame shift (samples).

	Output:
		number of frames
	'''
	return tf.cast(tf.ceil(tf.truediv(tf.cast(N, tf.float32),tf.cast(N_s, tf.float32))), tf.int32) # number of frames.

def acou_num_frames(N, N_s, K_s):
    '''
    Returns the number of acoustic-domain frames for a given sequence length, and
    frame shift.

    Inputs:
        N - time-domain sequence length (samples).
        N_s - time-domain frame shift (samples).
        K_s - acoustic-domain frame shift (samples).

    Output:
        number of modulation-domain frames
    '''
    N = tf.cast(N, tf.float32)
    N_s = tf.cast(N_s, tf.float32)
    K_s = tf.cast(K_s, tf.float32)
    return tf.cast(tf.ceil(tf.truediv(tf.truediv(N, N_s), K_s)), tf.int32) # number of frames.
