## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from tensorflow.python.ops.signal import window_ops
import functools
import numpy as np
import scipy.special as spsp
import tensorflow as tf

class STFT
	"""
	Short-time Fourier transform.
	"""
	def __init__(self, N_w, N_s, NFFT, f_s):
		"""
		Argument/s
			Nw - window length (samples).
			Ns - window shift (samples).
			NFFT - number of DFT bins.
			f_s - sampling frequency.
		"""
		self.N_w = N_w
		self.N_s = N_s
		self.NFFT = NFFT
		self.f_s = f_s
		self.W = functools.partial(window_ops.hamming_window, periodic=False)

	def polar_analysis(self, x):
		"""
		Polar-form acoustic-domain analysis.

		Argument/s:
			x - waveform.

		Output/s:
			Short-time magnitude and phase spectrums.
		"""
		STFT = tf.signal.stft(x, self.N_w, self.N_s, self.NFFT, window_fn=self.W, pad_end=True)
		return tf.abs(x_DFT), tf.angle(x_DFT)

	def polar_synthesis(self, STMS, STPS):
		"""
		Polar-form acoustic-domain synthesis.

		Argument/s:
			STMS - short-time magnitude spectrum.
			STPS - short-time phase spectrum.

		Output/s:
			Waveform.
		"""
		STFT = tf.cast(STMS, tf.complex64)*tf.exp(1j*tf.cast(STPS, tf.complex64))
		return tf.signal.inverse_stft(STFT, self.N_w, self.N_s, self.NFFT, tf.signal.inverse_stft_window_fn(self.N_s, self.W))

class DeepXiInput(STFT):
	"""
	Input for Deep Xi.
	"""
	def __init__(self, N_w, N_s, NFFT, f_s, mu=None, sigma=None):
		"""
		Argument/s
			Nw - window length (samples).
			Ns - window shift (samples).
			NFFT - number of DFT bins.
			f_s - sampling frequency.
			mu - sample mean of each instantaneous a priori SNR in dB frequency component.
			sigma - sample standard deviation of each instantaneous a priori SNR in dB frequency component.
		"""
		super().__init__(N_w, N_s, NFFT, f_s)
		self.mu = mu
		self.sigma = sigma

	def training_example(self, s, d, s_len, d_len, SNR):
		"""
		Compute training example for Deep Xi, i.e. observation (noisy-speech STMS) 
		and target (mapped a priori SNR).

		Argument/s:
			s - clean speech (dtype=tf.int32).
			d - noise (dtype=tf.int32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			SNR - SNR level.

		Output/s:
			x_STMS - noisy-speech short-time magnitude spectrum.
			xi_bar - mapped a priori SNR.
			n_frames - number of time-domain frames.
		"""
		s_STMS, d_STMS, x_STMS, n_frames = self.mix(s, d, s_len, d_len, SNR)
		xi_bar = self.xi_bar(s_STMS, d_STMS, self.mu, self.sigma)
		return x_STMS, xi_bar, n_frames

	def instantaneous_a_priori_snr_db(self, s, d, s_len, d_len, SNR):
		"""
	    Instantaneous a priori SNR in dB.

		Argument/s:
			s - clean speech (dtype=tf.int32).
			d - noise (dtype=tf.int32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			SNR - SNR level.

		Output/s:
			xi_dB - instantaneous a priori SNR in dB.
			L - number of time-domain frames for each sequence.
		"""
		s_STMS, d_STMS, _, n_frames = self.mix(s, d, SNR)
		xi_dB = self.xi_dB(s_STMS, d_STMS)
		return xi_dB, n_frames

	def observation(self, x, x_len):
		"""
	    An observation for Deep Xi (noisy-speech STMS).

		Argument/s:
			x - noisy speech (dtype=tf.int32).
			x_len - noisy speech length without padding (samples).

		Output/s:
			x_STMS - speech magnitude spectrum.
			n_frames - number of time-domain frames.
			x_STPS - speech phase spectrum.
		"""
		x = self.normalisation(x)
		n_frames = self.n_frames(x_len)
		x_STMS, x_STPS = self.polar_analysis(z)
		return x_STMS, n_frames, x_STPS

	def add_noise_batch(self, s, d, s_len, d_len, SNR):
		"""
		Creates noisy speech batch from clean speech, noise, and SNR batches.

		Argument/s:
			s - clean speech (dtype=tf.float32).
			d - noise (dtype=tf.float32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			SNR - SNR levels.

		Output/s:
			tuple consisting of clean speech, noisy speech, and noise (x, s, d).
		"""
		return tf.map_fn(lambda z: self.add_noise_pad(z[0], z[1], z[2], z[3], z[4],
			tf.reduce_max(s_len)), (s, d, s_len, d_len, SNR), dtype=(tf.float32, tf.float32,
			tf.float32))

	def add_noise_pad(self, s, d, s_len, d_len, SNR, P):
		"""
		Calls addnoise() and pads the waveforms to the length given by P.
		Also normalises the waveforms.

		Inputs:
			s - clean speech (dtype=tf.float32).
			d - noise (dtype=tf.float32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			SNR - SNR level.
			P - padded length.

		Outputs:
			s - padded clean speech waveform.
			x - padded noisy speech waveform.
			d - truncated, scaled, and padded noise waveform.
		"""
		(x, d) = self.add_noise(s, d, SNR)
		total_zeros = tf.subtract(P, tf.shape(s)[0])
		x = tf.pad(x, [[0, total_zeros]], "CONSTANT")
		s = tf.pad(s, [[0, total_zeros]], "CONSTANT")
		d = tf.pad(d, [[0, total_zeros]], "CONSTANT")
		return (x, s, d)

	def add_noise(self, s, d, SNR):
		"""
		Adds noise to the clean waveform at a specific SNR value. A random section 
		of the noise waveform is used.

		Inputs:
			s - clean speech (dtype=tf.float32).
			d - noise (dtype=tf.float32).
			SNR - SNR level.

		Outputs:
			x - noisy speech waveform.
			d - truncated and scaled noise waveform.
		"""
		s_len = tf.shape(s)[0]
		d_len = tf.shape(d)[0]
		i = tf.random_uniform([1], 0, tf.add(1, tf.subtract(d_len, s_len)), tf.int32)
		d = tf.slice(d, [i[0]], [s_len])
		d = tf.multiply(tf.truediv(d, tf.norm(d)), tf.truediv(tf.norm(s), 
			tf.pow(10.0, tf.multiply(0.05, SNR))))
		x = tf.add(s, d)
		return (x, d)

	def normalise(self, x):
		"""
		Convert waveform from int32 to float32 and normalise between [-1.0, 1.0].

		Argument/s:
			x - tf.int32 waveform.

		Output/s:
			tf.float32 waveform between [-1.0, 1.0].
		"""
		return tf.truediv(tf.cast(x, tf.float32), 32768.0)

	def n_frames(self, N):
		"""
		Returns the number of frames for a given sequence length, and
		frame shift.

		Inputs:
			N - sequence length (samples).

		Output/s:
			Number of frames
		"""
		return tf.cast(tf.ceil(tf.truediv(tf.cast(N, tf.float32), tf.cast(self.N_s, tf.float32))), tf.int32)

	def mix(self, s, d, s_len, d_len, SNR):
		"""
		Mix the clean speech and noise at SNR level, and then perform STFT analysis.

		Argument/s:
			s - clean speech (dtype=tf.int32).
			d - noise (dtype=tf.int32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			SNR - SNR level.

		Output/s:
			s_STMS - clean-speech short-time magnitude spectrum.
			d_STMS - noise short-time magnitude spectrum.
			x_STMS - noisy-speech short-time magnitude spectrum.
			n_frames - number of time-domain frames.
		"""
		s, d = self.normalise(s), self.normalise(d)
		n_frames = self.n_frames(x_len)
		(x, s, d) = add_noise_batch(s, d, s_len, d_len, SNR)
		s_STMS, _ = self.polar_analysis(s)
		s_STMS = tf.boolean_mask(s_STMS, tf.sequence_mask(n_frames))
		d_STMS, _ = self.polar_analysis(d)
		d_STMS = tf.boolean_mask(d_STMS, tf.sequence_mask(n_frames))
		x_STMS, _ = self.polar_analysis(x)
		return s_STMS, d_STMS, x_STMS, n_frames

	def log10(self, x):
		"""
		log_10(x).

		Argument/s:
			x - input.

		Output/s:
			log_10(x)
		"""
		return tf.truediv(tf.log(x), tf.constant(np.log(10), dtype=numerator.dtype))

	def xi(self, s_STMS, d_STMS):
		"""
		Instantaneous a priori SNR.

		Argument/s:
			s_STMS - clean-speech short-time magnitude spectrum.
			d_STMS - noise short-time magnitude spectrum.

		Output/s:
			Instantaneous a priori SNR.
		"""
		return tf.truediv(tf.square(s_STMS), tf.maximum(tf.square(d_STMS), 1e-12))

	def xi_dB(self, s_STMS, d_STMS):
		"""
		Instantaneous a priori SNR in dB.

		Argument/s:
			s_STMS - clean-speech short-time magnitude spectrum.
			d_STMS - noise short-time magnitude spectrum.

		Output/s:
			Instantaneous a priori SNR in dB.
		"""
		return tf.multiply(10.0, log10(tf.maximum(self.xi(s_STMS, d_STMS), 1e-12)))

	def xi_bar(self, s_STMS, d_STMS):
		"""
		Mapped a priori SNR in dB.

		Argument/s:
			s_STMS - clean-speech short-time magnitude spectrum.
			d_STMS - noise short-time magnitude spectrum.

		Output/s:
			Mapped a priori SNR in dB.
		"""
		return tf.multiply(0.5, tf.add(1.0, tf.erf(tf.truediv(tf.subtract(xi_dB(s_STMS, d_STMS), self.mu), 
			tf.multiply(self.sigma, tf.sqrt(2.0))))))

	def xi_hat(self, xi_bar_hat):
		"""
		A priori SNR estimate.

		Argument/s:
			xi_bar_hat - mapped a priori SNR estimate.

		Output/s:
			A priori SNR estimate.
		"""
		xi_dB_hat = np.add(np.multiply(np.multiply(self.sigma, np.sqrt(2.0)), 
			spsp.erfinv(np.subtract(np.multiply(2.0, xi_bar_hat), 1))), self.mu)	
		return np.power(10.0, np.divide(xi_dB_hat, 10.0))