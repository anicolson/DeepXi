## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from deepxi.gain import gfunc
from deepxi.map import map_selector
from deepxi.sig import InputTarget
import math
import tensorflow as tf

def inp_tgt_selector(inp_tgt_type, N_d, N_s, K, f_s, **kwargs):
	"""
	Selects the inp_tgt class.

	Argument/s:
		inp_tgt_type - inp_tgt type.
		N_d - window duration (time samples).
		N_s - window shift (time samples).
		K - number of frequency components.
		f_s - sampling frequency.
		kwargs - keyword arguments.

	Returns:
		inp_tgt class.
	"""
	if inp_tgt_type == "MagXi":
		return MagXi(N_d, N_s, K, f_s, xi_map_type=kwargs['xi_map_type'],
			xi_map_params=kwargs['xi_map_params'])
	elif inp_tgt_type == "MagGamma":
		return MagGamma(N_d, N_s, K, f_s, gamma_map_type=kwargs['gamma_map_type'],
			gamma_map_params=kwargs['gamma_map_params'])
	elif inp_tgt_type == "MagXiGamma":
		return MagXiGamma(N_d, N_s, K, f_s, xi_map_type=kwargs['xi_map_type'],
			xi_map_params=kwargs['xi_map_params'],
			gamma_map_type=kwargs['gamma_map_type'],
			gamma_map_params=kwargs['gamma_map_params'])
	elif inp_tgt_type == "MagGain":
		return MagGain(N_d, N_s, K, f_s, gain=kwargs['gain'])
	elif inp_tgt_type == "MagPhaXiPha":
		return MagPhaXiPha(N_d, N_s, K, f_s, xi_map_type=kwargs['xi_map_type'],
			xi_map_params=kwargs['xi_map_params'],
			s_stps_map_type=kwargs['s_stps_map_type'],
			s_stps_map_params=kwargs['s_stps_map_params'])
	# elif inp_tgt_type == "STDCTXiBarCDM": return STDCTXiBarCDM(N_d,
	# 	N_s, K, f_s, stdct_mu_xi_db=kwargs['stdct_mu_xi_db'],
	# 	stdct_sigma_xi_db=kwargs['stdct_sigma_xi_db'])
	# elif inp_tgt_type == "PowDBXi": return PowDBXiBar(N_d, N_s, K, f_s,
	# 	mu_ps_db=kwargs['mu_ps_db'], sigma_ps_db=kwargs['sigma_ps_db'])
	else: raise ValueError("Invalid inp_tgt type.")

class MagTgt(InputTarget):
	"""
	Base class for magnitude spectrum input and any target.
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

	def observation(self, x):
		"""
	    An observation for Deep Xi (noisy-speech STMS).

		Argument/s:
			x - noisy speech (dtype=tf.int32).
			x_len - noisy speech length without padding (samples).

		Returns:
			x_STMS - short-time magnitude spectrum.
			x_STPS - short-time phase spectrum.
		"""
		x = self.normalise(x)
		x_STMS, x_STPS = self.polar_analysis(x)
		return x_STMS, x_STPS

	def stats(self, s_STMS, d_STMS, x_STMS):
		"""
		The base stats() function is used when no statistics are requied for
			the target.

		Argument/s:
			s_STMS, d_STMS, x_STMS - clean-speech, noise, and noisy-speech
				magnitude spectrum samples.
		"""
		pass

class MagXi(MagTgt):
	"""
	Magnitude spectrum input and mapped a priori SNR target.
	"""
	def __init__(self, N_d, N_s, K, f_s, xi_map_type, xi_map_params):
		super().__init__(N_d, N_s, K, f_s)
		"""
		Argument/s
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
			xi_map_type - instantaneous a priori SNR map type.
			xi_map_params - parameters for the a priori SNR map.
		"""
		self.n_feat = math.ceil(K/2 + 1)
		self.n_outp = self.n_feat
		self.xi_map = map_selector(xi_map_type, xi_map_params)

	def stats(self, s_STMS, d_STMS, x_STMS):
		"""
		Compute statistics for map class.

		Argument/s:
			s_STMS, d_STMS, x_STMS - clean-speech, noise, and noisy-speech
				magnitude spectrum samples.
		"""
		xi = self.xi(s_STMS, d_STMS)
		self.xi_map.stats(xi)

	def example(self, s, d, s_len, d_len, snr):
		"""
		Compute example for Deep Xi, i.e. observation (noisy-speech STMS)
		and target (mapped a priori SNR).

		Argument/s:
			s - clean speech (dtype=tf.int32).
			d - noise (dtype=tf.int32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level.

		Returns:
			x_STMS - noisy-speech short-time magnitude spectrum.
			xi_bar - mapped a priori SNR.
			n_frames - number of time-domain frames.
		"""
		s, d, x, n_frames = self.mix(s, d, s_len, d_len, snr)
		s_STMS, _ = self.polar_analysis(s)
		d_STMS, _ = self.polar_analysis(d)
		x_STMS, _ = self.polar_analysis(x)
		xi = self.xi(s_STMS, d_STMS)
		xi_bar = self.xi_map.map(xi)
		return x_STMS, xi_bar, n_frames

	def enhanced_speech(self, x_STMS, x_STPS, xi_bar_hat, gtype):
		"""
		Compute enhanced speech.

		Argument/s:
			x_STMS - noisy-speech short-time magnitude spectrum.
			x_STPS - noisy-speech short-time phase spectrum.
			xi_bar_hat - mapped a priori SNR estimate.
			gtype - gain function type.

		Returns:
			enhanced speech.
		"""
		xi_hat = self.xi_map.inverse(xi_bar_hat)
		gamma_hat = tf.math.add(xi_hat, self.one)
		y_STMS = tf.math.multiply(x_STMS, gfunc(xi_hat, gamma_hat, gtype))
		return self.polar_synthesis(y_STMS, x_STPS)

	def xi_hat(self, xi_bar_hat):
		"""
		A priori SNR estimate.

		Argument/s:
			xi_bar_hat - mapped a priori SNR estimate.

		Returns:
			xi_hat - a priori SNR estimate.
		"""
		xi_hat = self.xi_map.inverse(xi_bar_hat)
		return xi_hat

	def gamma_hat(self, xi_bar_hat):
		"""
		A posteriori SNR estimate.

		Argument/s:
			xi_bar_hat - mapped a priori SNR estimate.

		Returns:
			gamma_hat - a posteriori SNR estimate.
		"""
		xi_hat = self.xi_map.inverse(xi_bar_hat)
		return tf.math.add(xi_hat, 1.0).numpy()

class MagGamma(MagTgt):
	"""
	Magnitude spectrum input and instantaneous a posteriori SNR target.
	"""
	def __init__(self, N_d, N_s, K, f_s, gamma_map_type, gamma_map_params):
		super().__init__(N_d, N_s, K, f_s)
		"""
		Argument/s
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
			gamma_map_type - instantaneous a posteriori SNR map type.
			gamma_map_params - parameters for the a posteriori SNR map.
		"""
		self.n_feat = math.ceil(K/2 + 1)
		self.n_outp = self.n_feat
		self.gamma_map = map_selector(gamma_map_type, gamma_map_params)

	def stats(self, s_STMS, d_STMS, x_STMS):
		"""
		Compute statistics for map class.

		Argument/s:
			s_STMS, d_STMS, x_STMS - clean-speech, noise, and noisy-speech
				magnitude spectrum samples.
		"""
		gamma = self.gamma(x_STMS, d_STMS)
		self.gamma_map.stats(gamma)

	def example(self, s, d, s_len, d_len, snr):
		"""
		Compute example for Deep Xi, i.e. observation (noisy-speech STMS)
		and target (mapped a posteriori SNR).

		Argument/s:
			s - clean speech (dtype=tf.int32).
			d - noise (dtype=tf.int32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level.

		Returns:
			x_STMS - noisy-speech short-time magnitude spectrum.
			gamma_bar - mapped a priori SNR.
			n_frames - number of time-domain frames.
		"""
		s, d, x, n_frames = self.mix(s, d, s_len, d_len, snr)
		s_STMS, _ = self.polar_analysis(s)
		d_STMS, _ = self.polar_analysis(d)
		x_STMS, _ = self.polar_analysis(x)
		gamma = self.gamma(x_STMS, d_STMS)
		gamma_bar = self.gamma_map.map(gamma)
		return x_STMS, gamma_bar, n_frames

	def enhanced_speech(self, x_STMS, x_STPS_xi_hat_mat, gamma_bar_hat, gtype):
		"""
		Compute enhanced speech.

		Argument/s:
			x_STMS - noisy-speech short-time magnitude spectrum.
			x_STPS_xi_hat_mat - tuple of noisy-speech short-time phase spectrum and
				a priori SNR loaded from .mat file.
			gamma_bar_hat - mapped a priori SNR estimate.
			gtype - gain function type.

		Returns:
			enhanced speech.
		"""
		gamma_hat = self.gamma_map.inverse(gamma_bar_hat)
		x_STPS, xi_hat_mat = x_STPS_xi_hat_mat
		xi_hat = xi_hat_mat['xi_hat']
		y_STMS = tf.math.multiply(x_STMS, gfunc(xi_hat, gamma_hat, gtype))
		return self.polar_synthesis(y_STMS, x_STPS)

	def gamma_hat(self, gamma_bar_hat):
		"""
		A posteriori SNR estimate.

		Argument/s:
			gamma_bar_hat - mapped a posteriori SNR estimate.

		Returns:
			gamma_hat - a posteriori SNR estimate.
		"""
		gamma_hat = self.gamma_map.inverse(gamma_bar_hat)
		return gamma_hat

	def xi_hat(self, gamma_bar_hat):
		"""
		A priori SNR estimate.

		Argument/s:
			gamma_bar_hat - mapped a priori SNR estimate.

		Returns:
			xi_hat - a priori SNR estimate.
		"""
		gamma_hat = self.gamma_map.inverse(gamma_bar_hat)
		return tf.maximum(tf.math.subtract(gamma_hat, 1.0), 1e-12).numpy()

class MagXiGamma(MagTgt):
	"""
	Magnitude spectrum input and mapped a priori and a posteriori SNR target.
	"""
	def __init__(self, N_d, N_s, K, f_s, xi_map_type, xi_map_params,
		gamma_map_type, gamma_map_params):
		super().__init__(N_d, N_s, K, f_s)
		"""
		Argument/s
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
			xi_map_type - instantaneous a priori SNR map type.
			xi_map_params - parameters for the a priori SNR map.
			gamma_map_type - instantaneous a posteriori SNR map type.
			gamma_map_params - parameters for the a posteriori SNR map.
		"""
		self.n_feat = math.ceil(K/2 + 1)
		self.n_outp = self.n_feat*2
		self.xi_map = map_selector(xi_map_type, xi_map_params)
		self.gamma_map = map_selector(gamma_map_type, gamma_map_params)

	def stats(self, s_STMS, d_STMS, x_STMS):
		"""
		Compute statistics for map class.

		Argument/s:
			s_STMS, d_STMS, x_STMS - clean-speech, noise, and noisy-speech
				magnitude spectrum samples.
		"""
		xi = self.xi(s_STMS, d_STMS)
		self.xi_map.stats(xi)
		gamma = self.gamma(x_STMS, d_STMS)
		self.gamma_map.stats(gamma)

	def example(self, s, d, s_len, d_len, snr):
		"""
		Compute example for Deep Xi, i.e. observation (noisy-speech STMS)
		and target (mapped a priori and a posteriori SNR).

		Argument/s:
			s - clean speech (dtype=tf.int32).
			d - noise (dtype=tf.int32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level.

		Returns:
			x_STMS - noisy-speech short-time magnitude spectrum.
			xi_gamma_bar - mapped a priori and a posteriori SNR.
			n_frames - number of time-domain frames.
		"""
		s, d, x, n_frames = self.mix(s, d, s_len, d_len, snr)
		s_STMS, _ = self.polar_analysis(s)
		d_STMS, _ = self.polar_analysis(d)
		x_STMS, _ = self.polar_analysis(x)
		xi = self.xi(s_STMS, d_STMS)
		xi_bar = self.xi_map.map(xi)
		gamma = self.gamma(x_STMS, d_STMS)
		gamma_bar = self.gamma_map.map(gamma)
		xi_gamma_bar = tf.concat([xi_bar, gamma_bar], axis=-1)
		return x_STMS, xi_gamma_bar, n_frames

	def enhanced_speech(self, x_STMS, x_STPS, xi_gamma_bar_hat, gtype):
		"""
		Compute enhanced speech.

		Argument/s:
			x_STMS - noisy-speech short-time magnitude spectrum.
			x_STPS - noisy-speech short-time phase spectrum.
			xi_gamma_bar_hat - mapped a priori and a posteriorir SNR estimate.
			gtype - gain function type.

		Returns:
			enhanced speech.
		"""
		xi_bar_hat, gamma_bar_hat = tf.split(xi_gamma_bar_hat,
			num_or_size_splits=2, axis=-1)
		xi_hat = self.xi_map.inverse(xi_bar_hat)
		gamma_hat = self.gamma_map.inverse(gamma_bar_hat)
		y_STMS = tf.math.multiply(x_STMS, gfunc(xi_hat, gamma_hat, gtype))
		return self.polar_synthesis(y_STMS, x_STPS)

	def xi_hat(self, xi_gamma_bar_hat):
		"""
		A priori SNR estimate.

		Argument/s:
			xi_gamma_bar_hat - mapped a priori and a posteriori SNR estimate.

		Returns:
			xi_hat - a priori SNR estimate.
		"""
		xi_bar_hat, _ = tf.split(xi_gamma_bar_hat, num_or_size_splits=2, axis=-1)
		xi_hat = self.xi_map.inverse(xi_bar_hat)
		return xi_hat

	def gamma_hat(self, xi_gamma_bar_hat):
		"""
		A posteriori SNR estimate.

		Argument/s:
			xi_gamma_bar_hat - mapped a priori and a posteriori SNR estimate.

		Returns:
			gamma_hat - a posteriori SNR estimate.
		"""
		_, gamma_bar_hat = tf.split(xi_gamma_bar_hat, num_or_size_splits=2, axis=-1)
		gamma_hat = self.gamma_map.inverse(gamma_bar_hat)
		return gamma_hat

class MagGain(MagTgt):
	"""
	Magnitude spectrum input and gain target.
	"""
	def __init__(self, N_d, N_s, K, f_s, gain):
		super().__init__(N_d, N_s, K, f_s)
		"""
		Argument/s
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
		"""
		self.n_feat = math.ceil(K/2 + 1)
		self.n_outp = self.n_feat
		self.gain = gain

	def example(self, s, d, s_len, d_len, snr):
		"""
		Compute example for Deep Xi, i.e. observation (noisy-speech STMS)
		and target (gain).

		Argument/s:
			s - clean speech (dtype=tf.int32).
			d - noise (dtype=tf.int32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level.

		Returns:
			x_STMS - noisy-speech short-time magnitude spectrum.
			gain - gain.
			n_frames - number of time-domain frames.
		"""
		s, d, x, n_frames = self.mix(s, d, s_len, d_len, snr)
		s_STMS, _ = self.polar_analysis(s)
		d_STMS, _ = self.polar_analysis(d)
		x_STMS, _ = self.polar_analysis(x)
		xi = self.xi(s_STMS, d_STMS) # instantaneous a priori SNR.
		gamma = self.gamma(x_STMS, d_STMS) # instantaneous a posteriori SNR.
		G = gfunc(xi=xi, gamma=gamma, gtype=self.gain)
		# IRM = tf.math.sqrt(tf.math.truediv(xi, tf.math.add(xi, self.one)))
		return x_STMS, G, n_frames

	def enhanced_speech(self, x_STMS, x_STPS, G_hat, gtype):
		"""
		Compute enhanced speech.

		Argument/s:
			x_STMS - noisy-speech short-time magnitude spectrum.
			x_STPS - noisy-speech short-time phase spectrum.
			G_hat - Gain estimate.
			gtype - gain function type.

		Returns:
			enhanced speech.
		"""
		y_STMS = tf.math.multiply(x_STMS, G_hat)
		return self.polar_synthesis(y_STMS, x_STPS)

class MagPhaXiPha(MagTgt):
	"""
	Magnitude spectrum and phase input and mapped a priori SNR and mapped phase
	target.
	"""
	def __init__(self, N_d, N_s, K, f_s, xi_map_type, xi_map_params,
		s_stps_map_type, s_stps_map_params):
		super().__init__(N_d, N_s, K, f_s)
		"""
		Argument/s
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
			xi_map_type - instantaneous a priori SNR map type.
			xi_map_params - parameters for the a priori SNR map.
			s_stps_map_type - phase map type.
			s_stps_map_params - parameters for the clean-speech STPS map.
		"""
		self.n_feat = math.ceil(K/2 + 1)*2
		self.n_outp = self.n_feat
		self.xi_map = map_selector(xi_map_type, xi_map_params)
		self.s_stps_map = map_selector(s_stps_map_type, s_stps_map_params)

	def observation(self, x):
		"""
	    An observation for Deep Xi (noisy-speech STMS and STPS).

		Argument/s:
			x - noisy speech (dtype=tf.int32).
			x_len - noisy speech length without padding (samples).

		Returns:
			x_STMS_STPS - short-time magnitude and phase spectrum.
			x_STMS_STPS - dummy variable.
		"""
		x = self.normalise(x)
		x_STMS, x_STPS = self.polar_analysis(x)
		x_STMS_STPS = tf.concat([x_STMS, x_STPS], axis=-1)
		return x_STMS_STPS, x_STMS_STPS

	def stats(self, s_STMS, d_STMS, x_STMS):
		"""
		Compute statistics for map class.

		Argument/s:
			s_STMS, d_STMS, x_STMS - clean-speech, noise, and noisy-speech
				magnitude spectrum samples.
		"""
		xi = self.xi(s_STMS, d_STMS)
		self.xi_map.stats(xi)

	def example(self, s, d, s_len, d_len, snr):
		"""
		Compute example for Deep Xi, i.e. observation (noisy-speech STMS)
		and target (mapped a priori SNR and clean-speech STPS).

		Argument/s:
			s - clean speech (dtype=tf.int32).
			d - noise (dtype=tf.int32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level.

		Returns:
			x_STMS - noisy-speech short-time magnitude spectrum.
			xi_s_stps_bar - mapped a priori SNR and clean-speech STPS.
			n_frames - number of time-domain frames.
		"""
		s, d, x, n_frames = self.mix(s, d, s_len, d_len, snr)
		s_STMS, s_STPS = self.polar_analysis(s)
		d_STMS, _ = self.polar_analysis(d)
		x_STMS, x_STPS = self.polar_analysis(x)
		x_STMS_STPS = tf.concat([x_STMS, x_STPS], axis=-1)
		s_stps_bar = self.s_stps_map.map(s_STPS)
		xi = self.xi(s_STMS, d_STMS)
		xi_bar = self.xi_map.map(xi)
		xi_s_stps_bar = tf.concat([xi_bar, s_stps_bar], axis=-1)
		return x_STMS_STPS, xi_s_stps_bar, n_frames

	def enhanced_speech(self, x_STMS_STPS, dummy, xi_s_stps_bar_hat, gtype):
		"""
		Compute enhanced speech.

		Argument/s:
			x_STMS_STPS - noisy-speech short-time magnitude and phase spectrum.
			dummy - dummy variable.
			xi_s_stps_bar_hat - mapped a priori SNR and clean-speech STPS estimate.
			gtype - gain function type.

		Returns:
			enhanced speech.
		"""
		x_STMS, _ = tf.split(x_STMS_STPS,
			num_or_size_splits=2, axis=-1)
		xi_bar_hat, s_stps_bar_hat = tf.split(xi_s_stps_bar_hat,
			num_or_size_splits=2, axis=-1)
		xi_hat = self.xi_map.inverse(xi_bar_hat)
		gamma_hat = tf.math.add(xi_hat, self.one)
		y_STPS = self.s_stps_map.inverse(s_stps_bar_hat)
		y_STMS = tf.math.multiply(x_STMS, gfunc(xi_hat, gamma_hat, gtype))
		return self.polar_synthesis(y_STMS, y_STPS)

	def xi_hat(self, xi_s_stps_bar_hat):
		"""
		A priori SNR estimate.

		Argument/s:
			xi_s_stps_bar_hat - mapped a priori SNR and clean-speech STPS estimate.

		Returns:
			xi_hat - a priori SNR estimate.
		"""
		xi_bar_hat, _ = tf.split(xi_s_stps_bar_hat, num_or_size_splits=2, axis=-1)
		xi_hat = self.xi_map.inverse(xi_bar_hat)
		return xi_hat

	def s_stps_hat(self, xi_s_stps_bar_hat):
		"""
		Clean-speech STPS estimate.

		Argument/s:
			xi_s_stps_bar_hat - mapped a priori SNR and clean-speech STPS estimate.

		Returns:
			xi_hat - a priori SNR estimate.
		"""
		_, s_STPS_bar_hat = tf.split(xi_s_stps_bar_hat, num_or_size_splits=2, axis=-1)
		s_STPS_hat = self.s_stps_map.inverse(s_STPS_bar_hat)
		return s_STPS_hat

# class STDCTXiBarCDM(InputTarget):
# 	"""
# 	Magnitude spectrum input. Mapped a priori SNR and CDM target.
# 	"""
# 	def __init__(self, N_d, N_s, K, f_s, stdct_mu_xi_db=None, stdct_sigma_xi_db=None):
# 		super().__init__(N_d, N_s, K, f_s)
# 		"""
# 		Argument/s
# 			N_d - window duration (samples).
# 			N_s - window shift (samples).
# 			K - number of frequency bins.
# 			f_s - sampling frequency.
# 			stdct_mu_xi_db - mean of instantaneous a priori SNR (dB) (STDCT).
# 			stdct_sigma_xi_db - standard deviation of instantaneous a priori SNR
# 				(dB) (STDCT).
# 		"""
# 		self.n_feat = K
# 		self.n_outp = self.n_feat*2
# 		self.stdct_mu_xi_db = stdct_mu_xi_db
# 		self.stdct_sigma_xi_db = stdct_sigma_xi_db
#
# 	def observation(self, x):
# 		"""
# 	    An observation for Deep Xi (noisy-speech STMS).
#
# 		Argument/s:
# 			x - noisy speech (dtype=tf.int32).
# 			x_len - noisy speech length without padding (samples).
#
# 		Returns:
# 			x_STMS - short-time magnitude spectrum.
# 			x_STPS - short-time phase spectrum.
# 		"""
# 		x = self.normalise(x)
# 		x_STDCT = self.stdct_analysis(x)
# 		return x_STDCT, None
#
# 	def example(self, s, d, s_len, d_len, snr):
# 		"""
# 		Compute example for Deep Xi, i.e. observation (noisy-speech STMS)
# 		and target (mapped a priori SNR).
#
# 		Argument/s:
# 			s - clean speech (dtype=tf.int32).
# 			d - noise (dtype=tf.int32).
# 			s_len - clean-speech length without padding (samples).
# 			d_len - noise length without padding (samples).
# 			snr - SNR level.
#
# 		Returns:
# 			x_STMS - noisy-speech short-time magnitude spectrum.
# 			xi_bar - mapped a priori SNR.
# 			n_frames - number of time-domain frames.
# 		"""
# 		s, d, x, n_frames = self.mix(s, d, s_len, d_len, snr)
# 		s_STDCT = self.stdct_analysis(s)
# 		d_STDCT = self.stdct_analysis(d)
# 		x_STDCT = self.stdct_analysis(x)
# 		xi_bar = self.xi_bar(s_STDCT, d_STDCT, self.stdct_mu_xi_db,
# 			self.stdct_sigma_xi_db)
# 		cdm = self.cdm(s_STDCT, d_STDCT)
# 		xi_bar_cdm = tf.concat([xi_bar, cdm], axis=-1)
# 		return x_STDCT, xi_bar_cdm, n_frames
#
# 	def enhanced_speech(self, x_STDCT, dummy, xi_bar_cdm_hat, gtype):
# 		"""
# 		Compute enhanced speech.
#
# 		Argument/s:
# 			x_STDCT - noisy-speech short-time magnitude spectrum.
# 			dummy - dummy variable (not used).
# 			xi_bar_hat - mapped a priori SNR estimate.
# 			gtype - gain function type.
#
# 		Returns:
# 			enhanced speech.
# 		"""
# 		xi_bar_hat, cdm_hat = tf.split(xi_bar_cdm_hat,
# 			num_or_size_splits=2, axis=-1)
# 		xi_hat = self.xi_bar_inv(xi_bar_hat, self.stdct_mu_xi_db,
# 			self.stdct_sigma_xi_db)
# 		cdm_hat = tf.math.greater(cdm_hat, 0.5)
# 		y_STDCT= tf.math.multiply(x_STDCT, gfunc(xi_hat, xi_hat+1.0, gtype, cdm_hat))
# 		return self.stdct_synthesis(y_STDCT)
#
# class PowDBXiBar(InputTarget):
# 	"""
# 	Power spectrum (dB) input (normalised) and mapped a priori SNR target.
# 	"""
# 	def __init__(self, N_d, N_s, K, f_s, mu=None, sigma=None, mu_ps_db=0.0,
# 		sigma_ps_db=1.0):
# 		super().__init__(N_d, N_s, K, f_s)
# 		"""
# 		Argument/s
# 			N_d - window duration (samples).
# 			N_s - window shift (samples).
# 			K - number of frequency bins.
# 			f_s - sampling frequency.
# 			mu - mean of each instantaneous a priori SNR (dB) frequency component.
# 			sigma - standard deviation of each instantaneous a priori SNR (dB) frequency component.
# 			mu_ps_db - mean of each power spectrum (dB) frequency component.
# 			sigma_ps_db - standard deviation of each power spectrum (dB) frequency component.
# 		"""
# 		self.n_feat = math.ceil(K/2 + 1)
# 		self.n_outp = self.n_feat
# 		self.mu = mu
# 		self.sigma = sigma
# 		self.mu_ps_db = mu_ps_db
# 		self.sigma_ps_db = sigma_ps_db
#
# 	def observation(self, x):
# 		"""
# 	    An observation for Deep Xi (noisy-speech STMS).
#
# 		Argument/s:
# 			x - noisy speech (dtype=tf.int32).
# 			x_len - noisy speech length without padding (samples).
#
# 		Returns:
# 			x_PS_dB_bar - mapped short-time power spectrum (dB).
# 			x_STPS - short-time phase spectrum.
# 		"""
# 		x = self.normalise(x)
# 		x_STMS, x_STPS = self.polar_analysis(x)
# 		x_PS_dB = tf.multiply(10.0, self.log_10(tf.math.square(x_STMS)))
# 		x_PS_dB_bar = tf.math.truediv(tf.math.subtract(x_PS_dB, self.mu_ps_db), self.sigma_ps_db)
# 		return x_PS_dB_bar, x_STPS
#
# 	def example(self, s, d, s_len, d_len, snr):
# 		"""
# 		Compute example for Deep Xi, i.e. observation (noisy-speech STMS)
# 		and target (mapped a priori SNR).
#
# 		Argument/s:
# 			s - clean speech (dtype=tf.int32).
# 			d - noise (dtype=tf.int32).
# 			s_len - clean-speech length without padding (samples).
# 			d_len - noise length without padding (samples).
# 			snr - SNR level.
#
# 		Returns:
# 			x_STMS - noisy-speech short-time magnitude spectrum.
# 			xi_bar - mapped a priori SNR.
# 			n_frames - number of time-domain frames.
# 		"""
# 		s, d, x, n_frames = self.mix(s, d, s_len, d_len, snr)
# 		s_STMS, _ = self.polar_analysis(s)
# 		d_STMS, _ = self.polar_analysis(d)
# 		x_STMS, _ = self.polar_analysis(x)
# 		# mask = tf.expand_dims(tf.cast(tf.sequence_mask(n_frames), tf.float32), 2)
# 		x_PS_dB = tf.multiply(10.0, self.log_10(tf.math.square(x_STMS)))
# 		x_PS_dB_bar = tf.math.truediv(tf.math.subtract(x_PS_dB, self.mu_ps_db), self.sigma_ps_db)
# 		# x_PS_dB_bar = tf.multiply(x_PS_dB_bar, mask)
# 		xi_bar = self.xi_bar(s_STMS, d_STMS, m, lambda_, kappa)
# 		# xi_bar = tf.multiply(xi_bar, mask)
# 		return x_PS_dB_bar, xi_bar, n_frames
#
