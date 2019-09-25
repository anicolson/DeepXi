## FILE:           add_noise.py
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
## BRIEF:          Add noise to clean speech at set SNR level.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import tensorflow as tf

def add_noise_batch(s, d, s_len, d_len, SNR):
	'''
	Creates noisy speech batch from clean speech, noise, and SNR batches.

	Input/s:
		s - clean waveforms (dtype=tf.int32).
		d - noisy waveforms (dtype=tf.int32).
		s_len - clean waveform lengths without padding (samples).
		d_len - noise waveform lengths without padding (samples).
		SNR - SNR levels.

	Output/s:
		tuple consisting of clean speech, noisy speech, and noise (x, s, d).
	'''
	return tf.map_fn(lambda z: add_noise_pad(z[0], z[1], z[2], z[3], z[4],
		tf.reduce_max(s_len)), (s, d, s_len, d_len, SNR), dtype=(tf.float32, tf.float32,
		tf.float32))

def add_noise_pad(s, d, s_len, d_len, SNR, P):
	'''
	Calls addnoise() and pads the waveforms to the length given by P.
	Also normalises the waveforms.

	Inputs:
		s - clean speech waveform.
		d - noise waveform.
		s_len - length of s.
		d_len - length of d.
		SNR - SNR level.
		P - padded length.

	Outputs:
		s - padded clean speech waveform.
		x - padded noisy speech waveform.
		d - truncated, scaled, and padded noise waveform.
	'''
	s = tf.truediv(tf.cast(tf.slice(s, [0], [s_len]), tf.float32), 32768.0)
	d = tf.truediv(tf.cast(tf.slice(d, [0], [d_len]), tf.float32), 32768.0)
	(x, d) = add_noise(s, d, SNR)
	total_zeros = tf.subtract(P, tf.shape(s)[0])
	x = tf.pad(x, [[0, total_zeros]], "CONSTANT")
	s = tf.pad(s, [[0, total_zeros]], "CONSTANT")
	d = tf.pad(d, [[0, total_zeros]], "CONSTANT")
	return (x, s, d)

def add_noise(s, d, SNR):
	'''
	Adds noise to the clean waveform at a specific SNR value. A random section 
	of the noise waveform is used.

	Inputs:
		s - clean waveform.
		d - noise waveform.
		SNR - SNR level.

	Outputs:
		x - noisy speech waveform.
		d - truncated and scaled noise waveform.
	'''
	s_len = tf.shape(s)[0]
	d_len = tf.shape(d)[0]
	i = tf.random_uniform([1], 0, tf.add(1, tf.subtract(d_len, s_len)), tf.int32)
	d = tf.slice(d, [i[0]], [s_len])
	d = tf.multiply(tf.truediv(d, tf.norm(d)), tf.truediv(tf.norm(s), 
		tf.pow(10.0, tf.multiply(0.05, SNR))))
	x = tf.add(s, d)
	return (x, d)