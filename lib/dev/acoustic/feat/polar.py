## FILE:           polar.py
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
## BRIEF:          Feature and target generation for polar representation in the acoustic-domain.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from dev.acoustic.analysis_synthesis import polar
import tensorflow as tf
from dev.add_noise import add_noise_batch
from dev.num_frames import num_frames
from dev.utils import log10

def input(z, z_len, N_w, N_s, NFFT, f_s):
	'''
    Input features for polar form acoustic-domain.

	Input/s:
		z - speech (dtype=tf.int32).
		z_len - speech length without padding (samples).
		N_w - time-domain window length (samples).
		N_s - time-domain window shift (samples).
		NFFT - number of acoustic-domain DFT components.
		f_s - sampling frequency (Hz).

	Output/s:
		z_MAG - speech magnitude spectrum.
		z_PHA - speech phase spectrum.
		L - number of time-domain frames for each sequence.
	'''
	z = tf.truediv(tf.cast(z, tf.float32), 32768.0)
	L = num_frames(z_len, N_s)
	z_MAG, z_PHA = polar.analysis(z, N_w, N_s, NFFT)
	return z_MAG, L, z_PHA

def input_target_spec(s, d, s_len, d_len, SNR, N_w, N_s, NFFT, f_s):
	'''
    Input features and target (spectrum) for polar form acoustic-domain.

	Inputs:
		s - clean speech (dtype=tf.int32).
		d - noise (dtype=tf.int32).
		s_len - clean speech length without padding (samples).
		d_len - noise length without padding (samples).
		SNR - SNR level.
		N_w - time-domain window length (samples).
		N_s - time-domain window shift (samples).
		NFFT - number of acoustic-domain DFT components.
		f_s - sampling frequency (Hz).

	Outputs:
		x_MAG - noisy speech magnitude spectrum.
		s_MAG - clean speech magnitude spectrum (target).
		L - number of time-domain frames for each sequence.
	'''
	(x, s, _) = add_noise_batch(s, d, s_len, d_len, SNR)
	L = num_frames(s_len, N_s) # number of time-domain frames for each sequence (uppercase eta).
	x_MAG, _ = polar.analysis(x, N_w, N_s, NFFT)	
	s_MAG, _ = polar.analysis(s, N_w, N_s, NFFT)
	s_MAG = tf.boolean_mask(s_MAG, tf.sequence_mask(L))
	return x_MAG, s_MAG, L

def input_target_xi(s, d, s_len, d_len, SNR, N_w, N_s, NFFT, f_s, mu, sigma):
	'''
    Input features and target (mapped a priori SNR) for polar form acoustic-domain.

	Inputs:
		s - clean speech (dtype=tf.int32).
		d - noise (dtype=tf.int32).
		s_len - clean speech length without padding (samples).
		d_len - noise length without padding (samples).
		SNR - SNR level.
		N_w - time-domain window length (samples).
		N_s - time-domain window shift (samples).
		NFFT - number of acoustic-domain DFT components.
		f_s - sampling frequency (Hz).
		mu - sample mean.
		sigma - sample standard deviation.
	
	Outputs:
		x_MAG - noisy speech magnitude spectrum.
		xi_mapped - mapped a priori SNR (target).
		L - number of time-domain frames for each sequence.
	'''
	(x, s, d) = add_noise_batch(s, d, s_len, d_len, SNR)
	L = num_frames(s_len, N_s) # number of acoustic-domain frames for each sequence (uppercase eta).
	x_MAG, _ = polar.analysis(x, N_w, N_s, NFFT)	
	s_MAG, _ = polar.analysis(s, N_w, N_s, NFFT)
	s_MAG = tf.boolean_mask(s_MAG, tf.sequence_mask(L))
	d_MAG, _ = polar.analysis(d, N_w, N_s, NFFT)
	d_MAG = tf.boolean_mask(d_MAG, tf.sequence_mask(L))
	xi = tf.truediv(tf.square(tf.maximum(s_MAG, 1e-12)), tf.square(tf.maximum(d_MAG, 1e-12))) # a priori SNR.
	xi_dB = tf.multiply(10.0, log10(xi)) # a priori SNR in dB.
	xi_mapped = tf.multiply(0.5, tf.add(1.0, tf.erf(tf.truediv(tf.subtract(xi_dB, mu), 
		tf.multiply(sigma, tf.sqrt(2.0)))))) # mapped a priori SNR.
	return x_MAG, xi_mapped, L

def target_xi(s, d, s_len, d_len, SNR, N_w, N_s, NFFT, f_s):
	'''
    Target (a priori SNR) for polar form acoustic-domain.

	Inputs:
		s - clean speech (dtype=tf.int32).
		d - noise (dtype=tf.int32).
		s_len - clean speech length without padding (samples).
		d_len - noise length without padding (samples).
		SNR - SNR level.
		N_w - time-domain window length (samples).
		N_s - time-domain window shift (samples).
		NFFT - number of acoustic-domain DFT components.
		f_s - sampling frequency (Hz).

	Outputs:
		xi_dB - a priori SNR in dB (target).
		L - number of time-domain frames for each sequence.
	'''
	(_, s, d) = add_noise_batch(s, d, s_len, d_len, SNR)
	L = num_frames(s_len, N_s) # number of acoustic-domain frames for each sequence (uppercase eta).
	s_MAG, _ = polar.analysis(s, N_w, N_s, NFFT)
	d_MAG, _ = polar.analysis(d, N_w, N_s, NFFT)
	s_MAG = tf.boolean_mask(s_MAG, tf.sequence_mask(L))
	d_MAG = tf.boolean_mask(d_MAG, tf.sequence_mask(L))
	xi = tf.truediv(tf.square(tf.maximum(s_MAG, 1e-12)), tf.square(tf.maximum(d_MAG, 1e-12))) # a priori SNR.
	xi_dB = tf.multiply(10.0, log10(xi)) # a priori SNR in dB.
	return xi_dB, L