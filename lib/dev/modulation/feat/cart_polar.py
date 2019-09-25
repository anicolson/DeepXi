## FILE:           cart_polar_spec.py
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
## BRIEF:          Feature and target generation for cartesian representation in the acoustic-domain and the polar representation in the modulation-domain.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from dev.modulation.analysis_synthesis import cart_polar
import tensorflow as tf
from dev.add_noise import add_noise_batch
from dev.num_frames import acou_num_frames
from dev.utils import log10

def input(z, z_len, N_w, N_s, NFFT, K_w, K_s, NFFT_m, f_s):
	'''
    Input features for cartesian form acoustic-domain and polar form modulation-domain.

	Input/s:
		z - speech (dtype=tf.int32).
		z_len - speech length without padding (samples).
		N_w - time-domain window length (samples).
		N_s - time-domain window shift (samples).
		NFFT - number of acoustic-domain DFT components.
		K_w - acoustic-domain window length (samples).
		K_s - acoustic-domain window shift (samples).
		NFFT_m - number of modulation-domain DFT components.
		f_s - sampling frequency (Hz).

	Output/s:
		z_RM_IM - speech RM and IM spectrums.
		Eta - number of acoustic-domain frames for each sequence.
		z_REAL_PHA - speech RP spectrum.
		z_IMAG_PHA - speech IP spectrum.
	'''
	z = tf.truediv(tf.cast(z, tf.float32), 32768.0)
	Eta = acou_num_frames(z_len, N_s, K_s)
	z_REAL_MAG, z_REAL_PHA, z_IMAG_MAG, z_IMAG_PHA = cart_polar.analysis(z, N_w, N_s, NFFT, K_w, K_s, NFFT_m)
	return tf.stack([z_REAL_MAG, z_IMAG_MAG], 4), Eta, z_REAL_PHA, z_IMAG_PHA

def input_target_spec(s, d, s_len, d_len, SNR, N_w, N_s, NFFT, K_w, K_s, NFFT_m, f_s):
	'''
    Input features and target (spectrum) for cartesian form acoustic-domain and polar form modulation-domain.

	Inputs:
		s - clean speech (dtype=tf.int32).
		d - noise (dtype=tf.int32).
		s_len - clean speech length without padding (samples).
		d_len - noise length without padding (samples).
		SNR - SNR level.
		N_w - time-domain window length (samples).
		N_s - time-domain window shift (samples).
		NFFT - number of acoustic-domain DFT components.
		K_w - acoustic-domain window length (samples).
		K_s - acoustic-domain window shift (samples).
		NFFT_m - number of modulation-domain DFT components.
		f_s - sampling frequency (Hz).

	Outputs:
		x_RM_IM - noisy speech RM and IM spectrums.
		s_RM_IM - clean speech RM and IM spectrums (target).
		Eta - number of acoustic-domain frames for each sequence.
	'''
	(x, s, _) = add_noise_batch(s, d, s_len, d_len, SNR)
	Eta = acou_num_frames(s_len, N_s, K_s) # number of acoustic-domain frames for each sequence (uppercase eta).
	x_REAL_MAG, _, x_IMAG_MAG, _ = cart_polar.analysis(x, N_w, N_s, NFFT, K_w, K_s, NFFT_m)	
	x_RM_IM = tf.stack([x_REAL_MAG, x_IMAG_MAG], 4)
	s_REAL_MAG, _, s_IMAG_MAG, _ = cart_polar.analysis(s, N_w, N_s, NFFT, K_w, K_s, NFFT_m)	
	s_RM_IM = tf.boolean_mask(tf.stack([s_REAL_MAG, s_IMAG_MAG], 4), tf.sequence_mask(Eta))
	return x_RM_IM, s_RM_IM, Eta

def input_target_xi(s, d, s_len, d_len, SNR, N_w, N_s, NFFT, K_w, K_s, NFFT_m, f_s, mu, sigma):
	'''
    Input features and target (mapped a priori SNR) for cartesian form acoustic-domain and polar form modulation-domain.

	Inputs:
		s - clean speech (dtype=tf.int32).
		d - noise (dtype=tf.int32).
		s_len - clean speech length without padding (samples).
		d_len - noise length without padding (samples).
		SNR - SNR level.
		N_w - time-domain window length (samples).
		N_s - time-domain window shift (samples).
		NFFT - number of acoustic-domain DFT components.
		K_w - acoustic-domain window length (samples).
		K_s - acoustic-domain window shift (samples).
		NFFT_m - number of modulation-domain DFT components.
		f_s - sampling frequency (Hz).
		mu - sample mean.
		sigma - sample standard deviation.

	Outputs:
		x_RM_IM - noisy speech RM and IM spectrums.
		xi_mapped - mapped a priori SNR (target).
		Eta - number of acoustic-domain frames for each sequence.
	'''
	(x, s, d) = add_noise_batch(s, d, s_len, d_len, SNR)
	Eta = acou_num_frames(s_len, N_s, K_s) # number of acoustic-domain frames for each sequence (uppercase eta).
	x_REAL_MAG, _, x_IMAG_MAG, _ = cart_polar.analysis(x, N_w, N_s, NFFT, K_w, K_s, NFFT_m)	
	x_RM_IM = tf.stack([x_REAL_MAG, x_IMAG_MAG], 4)
	s_REAL_MAG, _, s_IMAG_MAG, _ = cart_polar.analysis(s, N_w, N_s, NFFT, K_w, K_s, NFFT_m)	
	s_RM_IM = tf.boolean_mask(tf.stack([s_REAL_MAG, s_IMAG_MAG], 4), tf.sequence_mask(Eta))
	d_REAL_MAG, _, d_IMAG_MAG, _ = cart_polar.analysis(d, N_w, N_s, NFFT, K_w, K_s, NFFT_m)	
	d_RM_IM = tf.boolean_mask(tf.stack([d_REAL_MAG, d_IMAG_MAG], 4), tf.sequence_mask(Eta))
	xi = tf.truediv(tf.square(tf.maximum(s_RM_IM, 1e-12)), tf.square(tf.maximum(d_RM_IM, 1e-12))) # a priori SNR.
	xi_dB = tf.multiply(10.0, log10(xi)) # a priori SNR in dB.
	xi_mapped = tf.multiply(0.5, tf.add(1.0, tf.erf(tf.truediv(tf.subtract(xi_dB, mu), 
		tf.multiply(sigma, tf.sqrt(2.0)))))) # mapped a priori SNR.
	xi_mapped = tf.where(tf.is_nan(xi_mapped), tf.fill(tf.shape(xi_mapped), 0.5), xi_mapped)
	return x_RM_IM, xi_mapped, Eta

def target_xi(s, d, s_len, d_len, SNR, N_w, N_s, NFFT, K_w, K_s, NFFT_m, f_s):
	'''
    Target (a priori SNR) for cartesian form acoustic-domain and polar form modulation-domain.

	Inputs:
		s - clean speech (dtype=tf.int32).
		d - noise (dtype=tf.int32).
		s_len - clean speech length without padding (samples).
		d_len - noise length without padding (samples).
		SNR - SNR level.
		N_w - time-domain window length (samples).
		N_s - time-domain window shift (samples).
		NFFT - number of acoustic-domain DFT components.
		K_w - acoustic-domain window length (samples).
		K_s - acoustic-domain window shift (samples).
		NFFT_m - number of modulation-domain DFT components.
		f_s - sampling frequency (Hz).

	Outputs:
		xi_dB - a priori SNR in dB (target).
		Eta - number of acoustic-domain frames for each sequence.
	'''
	(_, s, d) = add_noise_batch(s, d, s_len, d_len, SNR)
	Eta = acou_num_frames(s_len, N_s, K_s) # number of acoustic-domain frames for each sequence (uppercase eta).
	s_REAL_MAG, _, s_IMAG_MAG, _ = cart_polar.analysis(s, N_w, N_s, NFFT, K_w, K_s, NFFT_m)	
	s_RM_IM = tf.boolean_mask(tf.stack([s_REAL_MAG, s_IMAG_MAG], 4), tf.sequence_mask(Eta))
	d_REAL_MAG, _, d_IMAG_MAG, _ = cart_polar.analysis(d, N_w, N_s, NFFT, K_w, K_s, NFFT_m)	
	d_RM_IM = tf.boolean_mask(tf.stack([d_REAL_MAG, d_IMAG_MAG], 4), tf.sequence_mask(Eta))
	xi = tf.truediv(tf.square(tf.maximum(s_RM_IM, 1e-12)), tf.square(tf.maximum(d_RM_IM, 1e-12))) # a priori SNR.
	xi_dB = tf.multiply(10.0, log10(xi)) # a priori SNR in dB.
	return xi_dB, Eta