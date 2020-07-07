## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from deepxi.gain import gfunc
from tensorflow.python.ops.signal import window_ops
import deepxi.dct as dct
import functools
import numpy as np
import scipy.special as spsp
import tensorflow as tf

"""
[1] Huang, X., Acero, A., Hon, H., 2001. Spoken Language Processing:
	A guide to theory, algorithm, and system development.
	Prentice Hall, Upper Saddle River, NJ, USA (pp. 315).
"""

class AnalysisSynthesis:
	"""
	Analysis and synthesis stages of speech enhacnement.
	"""
	def __init__(self, N_d, N_s, K, f_s):
		"""
		Argument/s:
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
		"""
		self.N_d = N_d
		self.N_s = N_s
		self.K = K
		self.f_s = f_s
		self.W = functools.partial(window_ops.hamming_window,
			periodic=False)
		self.ten = tf.cast(10.0, tf.float32)
		self.one = tf.cast(1.0, tf.float32)

	def polar_analysis(self, x):
		"""
		Polar-form acoustic-domain analysis.

		Argument/s:
			x - waveform.

		Returns:
			Short-time magnitude and phase spectrums.
		"""
		STFT = tf.signal.stft(x, self.N_d, self.N_s, self.K,
			window_fn=self.W, pad_end=True)
		return tf.abs(STFT), tf.math.angle(STFT)

	def polar_synthesis(self, STMS, STPS):
		"""
		Polar-form acoustic-domain synthesis.

		Argument/s:
			STMS - short-time magnitude spectrum.
			STPS - short-time phase spectrum.

		Returns:
			Waveform.
		"""
		STFT = tf.cast(STMS, tf.complex64)*tf.exp(1j*tf.cast(STPS, tf.complex64))
		return tf.signal.inverse_stft(STFT, self.N_d, self.N_s, self.K, tf.signal.inverse_stft_window_fn(self.N_s, self.W))

	def stdct_analysis(self, x):
		"""
		Short-time discrete cosine transform analysis.

		Argument/s:
			x - waveform.

		Returns:
			Short-time discrete cosine transform.
		"""
		return dct.stdct(x, self.N_d, self.N_s, self.K,
			window_fn=self.W, pad_end=True)

	def stdct_synthesis(self, STDCT):
		"""
		Short-time discrete cosine transform synthesis.

		Argument/s:
			STDCT - short-time discrete cosine transform.

		Returns:
			Waveform.
		"""
		return dct.inverse_stdct(STDCT, self.N_d, self.N_s, self.K,
			tf.signal.inverse_stft_window_fn(self.N_s, self.W))

class InputTarget(AnalysisSynthesis):
	"""
	Computes the input and target of Deep Xi.
	"""
	def __init__(self, N_d, N_s, K, f_s):
		super().__init__(N_d, N_s, K, f_s)
		"""
		Argument/s
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
		"""
	def xi(self, S, D):
		"""
		Instantaneous a priori SNR.

		Argument/s:
			S - clean-speech short-time spectrum.
			D - noise short-time spectrum.

		Returns:
			Instantaneous a priori SNR.
		"""
		return tf.truediv(tf.square(S), tf.maximum(tf.square(D), 1e-12))

	# def xi_db(self, S, D):
	# 	"""
	# 	Instantaneous a priori SNR in dB.
	#
	# 	Argument/s:
	# 		S - clean-speech short-time spectrum.
	# 		D - noise short-time spectrum.
	#
	# 	Returns:
	# 		Instantaneous a priori SNR in dB.
	# 	"""
	# 	return tf.multiply(10.0, self.log_10(tf.maximum(self.xi(S, D), 1e-12)))

	# def xi_bar(self, S, D, mu, sigma):
	# 	"""
	# 	Mapped a priori SNR in dB.
	#
	# 	Argument/s:
	# 		S - clean-speech short-time spectrum.
	# 		D - noise short-time spectrum.
	# 		mu - mean of each instantaneous a priori SNR (dB) frequency component.
	# 		sigma - standard deviation of each instantaneous a priori SNR (dB) frequency component.
	#
	# 	Returns:
	# 		Mapped a priori SNR in dB.
	# 	"""
	# 	return
	#
	# def xi_bar_inv(self, xi_bar_hat, mu, sigma):
	# 	"""
	# 	A priori SNR estimate computed from the inverse of the mapped a priori SNR.
	#
	# 	Argument/s:
	# 		xi_bar_hat - mapped a priori SNR estimate.
	# 		mu - mean of each instantaneous a priori SNR (dB) frequency component.
	# 		sigma - standard deviation of each instantaneous a priori SNR (dB) frequency component.
	#
	# 	Returns:
	# 		A priori SNR estimate.
	# 	"""
	# 	xi_db_hat = np.add(np.multiply(np.multiply(sigma, np.sqrt(2.0)),
	# 		spsp.erfinv(np.subtract(np.multiply(2.0, xi_bar_hat), 1))), mu)
	# 	return np.power(10.0, np.divide(xi_db_hat, 10.0))

	def gamma(self, X, D):
		"""
		Instantaneous a posteriori SNR.

		Argument/s:
			X - noisy-speech short-time spectrum.
			D - noise short-time spectrum.

		Returns:
			Instantaneous a posteriori SNR.
		"""
		return tf.truediv(tf.square(X), tf.maximum(tf.square(D), 1e-12))

	# def gamma_db(self, X, D):
	# 	"""
	# 	Instantaneous a posteriori SNR (dB).
	#
	# 	Argument/s:
	# 		X - noisy-speech short-time spectrum.
	# 		D - noise short-time spectrum.
	#
	# 	Returns:
	# 		Instantaneous a posteriori SNR (dB).
	# 	"""
	# 	return tf.multiply(10.0, self.log_10(tf.maximum(self.gamma(X, D), 1e-12)))

	# def gamma_bar(self, X, D, alpha, b):
	# 	"""
	# 	Mapped a posteriori SNR in dB. Assumed truncated double gamma
	# 	distribution. The location parameter is assumed to be 1.0. 'a' is set to
	# 	0.0 as it is the lowest possible value for the a posteriori SNR.
	#
	# 	Argument/s:
	# 		X - noisy-speech short-time spectrum.
	# 		D - noise short-time spectrum.
	# 		alpha - location parameter.
	# 		b - upper bounds of interval.
	#
	# 	Returns:
	# 		Mapped a posteriori SNR.
	# 	"""
	# 	gamma = self.gamma(X, D)
	# 	return self.truncated_double_gamma_cdf(gamma, alpha, 1.0, 0.0, b)

	# def gamma_bar_inv(self, gamma_bar_hat, alpha, b):
	# 	"""
	# 	A posteriori SNR estimate computed from the inverse of the mapped a
	# 	posteriori SNR. The location parameter is assumed to be 1.0. 'a' is set
	# 	to 0.0 as it is the lowest possible value for the a posteriori SNR.
	#
	# 	Argument/s:
	# 		gamma_bar_hat - mapped a posteriori SNR estimate.
	# 		m - location parameter.
	# 		alpha - location parameter.
	# 		b - upper bounds of interval.
	#
	# 	Returns:
	# 		A posteriori SNR estimate.
	# 	"""
	# 	return self.truncated_double_gamma_cdf_inverse(gamma_bar_hat,
	# 		alpha, 1.0, 0.0, b)
	#
	# def gamma_bar_depreciated(self, X, D, m, lambda_, kappa):
	# 	"""
	# 	Mapped a posteriori SNR in dB. Assumed asymmetric Laplace distribution.
	#
	# 	Argument/s:
	# 		X - noisy-speech short-time spectrum.
	# 		D - noise short-time spectrum.
	# 		m - location parameter.
	# 		lambda_ - scale parameter.
	# 		kappa - asymmetry parameter.
	#
	# 	Returns:
	# 		Mapped a posteriori SNR in dB.
	# 	"""
	# 	gamma_db = self.gamma_db(X, D)
	# 	v_1 = tf.math.square(kappa)
	# 	v_2 = tf.math.add(1.0, v_1)
	# 	v_3 = tf.math.truediv(v_1, v_2)
	# 	v_4 = tf.math.truediv(1.0, v_2)
	# 	v_5 = tf.math.truediv(lambda_, kappa)
	# 	v_6 = tf.math.negative(tf.math.multiply(lambda_, kappa))
	# 	gamma_bar_minus = tf.math.multiply(v_3,
	# 		tf.math.exp(tf.math.multiply(v_5, tf.math.subtract(gamma_db, m))))
	# 	gamma_bar_plus = tf.math.subtract(1.0, tf.math.multiply(v_4,
	# 		tf.math.exp(tf.math.multiply(v_6, tf.math.subtract(gamma_db, m)))))
	# 	return tf.where(tf.math.greater(gamma_db, m), gamma_bar_plus,
	# 		gamma_bar_minus)
	#
	# def gamma_hat_depreciated(self, gamma_bar_hat, m, lambda_, kappa):
	# 	"""
	# 	A posteriori SNR estimate.
	#
	# 	Argument/s:
	# 		gamma_bar_hat - mapped a posteriori SNR estimate.
	# 		m - location parameter.
	# 		lambda_ - scale parameter.
	# 		kappa - asymmetry parameter.
	#
	# 	Returns:
	# 		A posteriori SNR estimate.
	# 	"""
	# 	v_1 = tf.math.square(kappa)
	# 	v_2 = tf.math.add(1.0, v_1)
	# 	v_3 = tf.math.truediv(v_2, v_1)
	# 	v_4 = tf.math.truediv(kappa, lambda_)
	# 	v_5 = tf.math.multiply(kappa, lambda_)
	#
	# 	gamma_hat_minus = tf.math.add(tf.math.multiply(tf.math.log(tf.math.multiply(gamma_bar_hat,
	# 		v_3)), v_4), m)
	# 	gamma_hat_plus = tf.math.multiply(tf.math.subtract(1.0, gamma_bar_hat), v_2)
	# 	gamma_hat_plus = tf.math.subtract(m, tf.math.divide(tf.math.log(gamma_hat_plus), v_5))
	# 	logical = tf.math.greater(gamma_bar_hat, tf.math.truediv(v_1, v_2))
	# 	gamma_db_hat = tf.where(logical, gamma_hat_plus, gamma_hat_minus)
	# 	return np.power(10.0, np.divide(gamma_db_hat, 10.0))

	def cdm(self, S, D):
		"""
		Constructuve-deconstructive mask.

		Argument/s:
			S - clean-speech short-time spectrum.
			D - noise short-time spectrum.

		Returns:
			Instantaneous a priori SNR.
		"""
		return tf.cast(tf.math.greater_equal(tf.multiply(S, D), 0.0), tf.float32)

	# def normal_cdf(self, x, mu, sigma):
	# 	"""
	# 	Normal (Gaussian) cumulative distribution function (CDF).
	#
	# 	Argument/s:
	# 		x - random variable realisations.
	# 		mu - mean.
	# 		sigma - standard deviation.
	#
	# 	Returns:
	# 		CDF.
	# 	"""
	# 	return None
	#
	# def normal_cdf_inverse(self, x, mu, sigma):
	# 	"""
	# 	Inverse of normal (Gaussian) cumulative distribution function (CDF).
	#
	# 	Argument/s:
	# 		x - random variable realisations.
	# 		mu - mean.
	# 		sigma - standard deviation.
	#
	# 	Returns:
	# 		Inverse of CDF.
	# 	"""
	# 	return None
	#
	# def double_gamma_cdf(self, x, alpha, mu):
	# 	"""
	# 	Double gamma cumulative distribution function (CDF).
	#
	# 	Argument/s:
	# 		x - random variable realisations.
	# 		alpha - shape parameter.
	# 		mu - location parameter.
	#
	# 	Returns:
	# 		CDF.
	# 	"""
	# 	v_1 = tf.math.multiply(0.5, tf.math.igamma(alpha,
	# 		tf.math.abs(tf.math.subtract(x, mu))))
	# 	cdf_minus = tf.math.subtract(0.5, v_1)
	# 	cdf_plus = tf.math.add(0.5, v_1)
	# 	return tf.where(tf.math.greater(x, mu), cdf_plus, cdf_minus)
	#
	# def double_gamma_cdf_inverse(self, cdf, alpha, mu):
	# 	"""
	# 	Inverse of double gamma cumulative distribution function (CDF).
	#
	# 	Argument/s:
	# 		cdf - cumulative distribution function value.
	# 		alpha - shape parameter.
	# 		mu - location parameter.
	#
	# 	Returns:
	# 		x - inverse of CDF.
	# 	"""
	# 	x_lower = tf.math.subtract(1.0, tf.math.multiply(2.0, cdf))
	# 	x_lower = spsp.gammaincinv(alpha, x_lower)
	# 	x_lower = tf.math.subtract(mu, x_lower)
	# 	x_upper = tf.math.subtract(tf.math.multiply(2.0, cdf), 1.0)
	# 	x_upper = spsp.gammaincinv(alpha, x_upper)
	# 	x_upper = tf.math.add(mu, x_upper)
	# 	x = tf.where(tf.math.greater(cdf, 0.5), x_upper, x_lower)
	# 	return x
	#
	# def truncated_double_gamma_cdf(self, x, alpha, mu, a, b):
	# 	"""
	# 	Truncated double gamma cumulative distribution function (CDF).
	#
	# 	Argument/s:
	# 		x - random variable realisations.
	# 		alpha - shape parameter.
	# 		mu - location parameter.
	# 		a - lower bounds.
	# 		b - upper bounds.
	#
	# 	Returns:
	# 		CDF.
	# 	"""
	# 	cdf_a = self.double_gamma_cdf(a, alpha, mu)
	# 	cdf_b = self.double_gamma_cdf(b, alpha, mu)
	# 	cdf = self.double_gamma_cdf(x, alpha, mu)
	# 	cdf = tf.math.truediv(tf.math.subtract(cdf,
	# 		cdf_a), tf.math.subtract(cdf_b, cdf_a))
	# 	cdf = tf.where(tf.math.less(x, a), tf.zeros_like(x), cdf)
	# 	cdf = tf.where(tf.math.greater(x, b), tf.ones_like(x), cdf)
	# 	return cdf
	#
	# def truncated_double_gamma_cdf_inverse(self, cdf, alpha, mu, a, b):
	# 	"""
	# 	Inverse of truncated double gamma cumulative distribution function (CDF).
	#
	# 	Argument/s:
	# 		cdf - cumulative distribution function value.
	# 		alpha - shape parameter.
	# 		mu - location parameter.
	# 		a - lower bounds.
	# 		b - upper bounds.
	#
	# 	Returns:
	# 		x - inverse of CDF value.
	# 	"""
	# 	cdf_a = self.double_gamma_cdf(a, alpha, mu)
	# 	cdf_b = self.double_gamma_cdf(b, alpha, mu)
	# 	cdf = tf.math.add(tf.math.multiply(cdf,
	# 		tf.math.subtract(cdf_b, cdf_a)), cdf_a)
	# 	x = self.double_gamma_cdf_inverse(cdf, alpha, mu)
	# 	return x

	def mix(self, s, d, s_len, d_len, snr):
		"""
		Mix the clean speech and noise at SNR level, and then perform STFT analysis.

		Argument/s:
			s - clean speech (dtype=tf.int32).
			d - noise (dtype=tf.int32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level.

		Returns:
			s - clean speech.
			d - scaled random section of the noise.
			x - noisy speech.
			n_frames - number of time-domain frames.
		"""
		s, d = self.normalise(s), self.normalise(d)
		n_frames = self.n_frames(s_len)
		if tf.rank(s) == 2:
			x, s, d = self.add_noise_batch(s, d, s_len, d_len, snr)
		elif tf.rank(s) == 1:
			x, d = self.add_noise(s[:s_len], d[:d_len], s_len, d_len, snr)
			s = s[:s_len]
		else: raise ValueError("Waveforms are of incorrect rank.")
		return s, d, x, n_frames

	def normalise(self, x):
		"""
		Convert waveform from int32 to float32 and normalise between [-1.0, 1.0].

		Argument/s:
			x - tf.int32 waveform.

		Returns:
			tf.float32 waveform between [-1.0, 1.0].
		"""
		return tf.truediv(tf.cast(x, tf.float32), 32768.0)

	def n_frames(self, N):
		"""
		Returns the number of frames for a given sequence length, and
		frame shift.

		Argument/s:
			N - sequence length (samples).

		Returns:
			Number of frames
		"""
		return tf.cast(tf.math.ceil(tf.truediv(tf.cast(N, tf.float32), tf.cast(self.N_s, tf.float32))), tf.int32)

	def add_noise_batch(self, s, d, s_len, d_len, snr):
		"""
		Creates noisy speech batch from clean speech, noise, and SNR batches.

		Argument/s:
			s - clean speech (dtype=tf.float32).
			d - noise (dtype=tf.float32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR levels.

		Returns:
			tuple consisting of clean speech, noisy speech, and noise (x, s, d).
		"""
		return tf.map_fn(lambda z: self.add_noise_pad(z[0], z[1], z[2], z[3], z[4],
			tf.reduce_max(s_len)), (s, d, s_len, d_len, snr), dtype=(tf.float32, tf.float32,
			tf.float32), back_prop=False)

	def add_noise_pad(self, s, d, s_len, d_len, snr, pad_len):
		"""
		Calls add_noise() and pads the waveforms to the length given by 'pad_len'.
		Also normalises the waveforms.

		Argument/s:
			s - clean speech (dtype=tf.float32).
			d - noise (dtype=tf.float32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level.
			pad_len - padded length.

		Returns:
			s - padded clean-speech waveform.
			x - padded noisy-speech waveform.
			d - truncated, scaled, and padded noise waveform.
		"""
		s, d = s[:s_len], d[:d_len]
		(x, d) = self.add_noise(s, d, s_len, d_len, snr)
		total_zeros = tf.subtract(pad_len, s_len)
		x = tf.pad(x, [[0, total_zeros]], "CONSTANT")
		s = tf.pad(s, [[0, total_zeros]], "CONSTANT")
		d = tf.pad(d, [[0, total_zeros]], "CONSTANT")
		return (x, s, d)

	def add_noise(self, s, d, s_len, d_len, snr):
		"""
		Adds noise to the clean speech at a specific SNR value. A random section
		of the noise waveform is used.

		Argument/s:
			s - clean speech (dtype=tf.float32).
			d - noise (dtype=tf.float32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level (dB).

		Returns:
			x - noisy-speech waveform.
			d - truncated and scaled noise waveform.
		"""
		snr = tf.cast(snr, tf.float32)
		snr = tf.pow(self.ten, tf.truediv(snr, self.ten)) # inverse of dB.
		i = tf.random.uniform([1], 0, tf.add(1, tf.subtract(d_len, s_len)), tf.int32)
		d = tf.slice(d, [i[0]], [s_len])
		P_s = tf.reduce_mean(tf.math.square(s), 0) # average power of clean speech.
		P_d = tf.reduce_mean(tf.math.square(d), 0) # average power of noise.
		alpha = tf.math.sqrt(tf.truediv(P_s,
			tf.maximum(tf.multiply(P_d, snr), 1e-12))) # scaling factor.
		d =	tf.multiply(d, alpha)
		x = tf.add(s, d)
		return (x, d)

	def snr_db(self, s, d):
		"""
		Calculates the SNR (dB) between the speech and noise.

		Argument/s:
			s - clean speech (dtype=tf.float32).
			d - noise (dtype=tf.float32).

		Returns:
			SNR level (dB).
		"""
		P_s = tf.reduce_mean(tf.math.square(s), 0) # average power of clean speech.
		P_d = tf.reduce_mean(tf.math.square(d), 0) # average power of noise.
		return tf.multiply(self.ten, self.log_10(tf.truediv(P_s, P_d)))

	def mel_filter_bank(self, M):
		"""
		Created a mel-scaled filter bank using the equations from [1].
		The notation from [1] is also used. For this case, each filter
		sums to unity, so that it can be used to weight the STMS a
		priori SNR to compute the a priori SNR for each subband, i.e.
		each filter bank.

		Argument/s:
			M - number of filters.

		Returns:
			H - triangular mel filterbank matrix.

		"""
		f_l = 0 # lowest frequency (Hz).
		f_h = self.f_s/2 # highest frequency (Hz).
		K = self.K//2 + 1 # number of frequency bins.
		H = np.zeros([M, K], dtype=np.float32) # mel filter bank.
		for m in range(1, M + 1):
			bl = self.bpoint(m - 1, M, f_l, f_h) # lower boundary point, f(m - 1) for m-th filterbank.
			c = self.bpoint(m, M, f_l, f_h) # m-th filterbank centre point, f(m).
			bh = self.bpoint(m + 1, M, f_l, f_h) # higher boundary point f(m + 1) for m-th filterbank.
			for k in range(K):
				if k >= bl and k <= c:
					H[m-1,k] = (2*(k - bl))/((bh - bl)*(c - bl)) # m-th filterbank up-slope.
				if k >= c and k <= bh:
					H[m-1,k] = (2*(bh - k))/((bh - bl)*(bh - c)) # m-th filterbank down-slope.
		return H

	def bpoint(self, m, M, f_l, f_h):
		"""
		Detirmines the frequency bin boundary point for a filterbank.

		Argument/s:
			m - filterbank.
			M - total filterbanks.
			f_l - lowest frequency.
			f_h - highest frequency.

		Returns:
			Frequency bin boundary point.
		"""
		K = self.K//2 + 1 # number of frequency bins.
		return ((2*K)/self.f_s)*self.mel_to_hz(self.hz_to_mel(f_l) + \
			m*((self.hz_to_mel(f_h) - self.hz_to_mel(f_l))/(M + 1))) # boundary point.

	def hz_to_mel(self, f):
		"""
		Converts a value from the Hz scale to a value in the mel scale.

		Argument/s:
			f - Hertz value.

		Returns:
			Mel value.
		"""
		return 2595*np.log10(1 + (f/700))

	def mel_to_hz(self, m):
		"""
		Converts a value from the mel scale to a value in the Hz scale.

		Argument/s:
			m - mel value.

		Returns:
			Hertz value.
		"""
		return 700*((10**(m/2595)) - 1)

	def log_10(self, x):
		"""
		log_10(x).

		Argument/s:
			x - input.

		Returns:
			log_10(x)
		"""
		return tf.truediv(tf.math.log(x), tf.math.log(self.ten))

	def spectral_distortion(self, instantaneous, estimate):
		"""
		Computes the frame wise spectral distortion between the instantaneous
		a priori/posteriori (dB) and its estimate (dB). The function converts
		the values to dB

		Argument/s:
			instantaneous - instantaneous a priori/posteriori SNR.
			estimate - estimate of the a priori/posteriori SNR.

		Returns:
			Frame-wise root-mean-squared difference.
		"""
		instantaneous = tf.multiply(self.ten, self.log_10(instantaneous))
		estimate = tf.multiply(self.ten, self.log_10(instantaneous))
		diff = tf.math.subtract(instantaneous, estimate)
		squared_diff = tf.math.square(diff)
		mean_square_diff = tf.math.reduce_mean(squared_diff, axis=-1)
		root_mean_square_diff = tf.math.sqrt(mean_square_diff)
		return tf.math.reduce_mean(root_mean_square_diff)
