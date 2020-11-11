## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from deepxi.gain import gfunc
from deepxi.map import map_selector
from deepxi.sig import InputTarget
from deepxi.utils import save_mat
from tqdm import tqdm
import math
import numpy as np
import tensorflow as tf
"""
[1] Wang, Y., Narayanan, A. and Wang, D., 2014. On training targets for
	supervised speech separation. IEEE/ACM transactions on audio, speech, and
	language processing, 22(12), pp.1849-1858.
"""

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
		return MagXi(N_d, N_s, K, f_s, xi_map_type=kwargs['map_type'],
			xi_map_params=kwargs['map_params'])
	elif inp_tgt_type == "MagGamma":
		return MagGamma(N_d, N_s, K, f_s, gamma_map_type=kwargs['map_type'],
			gamma_map_params=kwargs['map_params'])
	elif inp_tgt_type == "MagXiGamma":
		return MagXiGamma(N_d, N_s, K, f_s, xi_map_type=kwargs['map_type'][0],
			xi_map_params=kwargs['map_params'][0],
			gamma_map_type=kwargs['map_type'][1],
			gamma_map_params=kwargs['map_params'][1])
	elif inp_tgt_type == "MagGain":
		return MagGain(N_d, N_s, K, f_s, gain=kwargs['gain'])
	elif inp_tgt_type == "MagMag":
		return MagMag(N_d, N_s, K, f_s, mag_map_type=kwargs['map_type'],
			mag_map_params=kwargs['map_params'])
	elif inp_tgt_type == "MagSMM":
		return MagSMM(N_d, N_s, K, f_s, smm_map_type=kwargs['map_type'],
			smm_map_params=kwargs['map_params'])
	elif inp_tgt_type == "MagPhaXiPha":
		return MagPhaXiPha(N_d, N_s, K, f_s, xi_map_type=kwargs['map_type'][0],
			xi_map_params=kwargs['map_params'][0],
			s_stps_map_type=kwargs['map_type'][1],
			s_stps_map_params=kwargs['map_params'][1])
	elif inp_tgt_type == "STDCTXiCD":
		return STDCTXiCD(N_d, N_s, K, f_s, xi_map_type=kwargs['map_type'][0],
			xi_map_params=kwargs['map_params'][0],
			cd_map_type=kwargs['map_type'][1],
			cd_map_params=kwargs['map_params'][1])
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

	def stats(self, s_sample, d_sample, x_sample, wav_len):
		"""
		The base stats() function is used when no statistics are requied for
			the target.

		Argument/s:
			s_sample, d_sample, x_sample, wav_len - clean speech, noise, noisy speech
				samples and their lengths.
		"""
		pass

	def transfrom_stats(self, s_sample, d_sample, x_sample, wav_len):
		"""
		Transforms time-domain sample to short-time magnitude spectrum sample.

		Argument/s:
			s_sample, d_sample, x_sample, wav_len - clean speech, noise,
				noisy speech samples and their lengths.

		Returns:
			s_STMS, d_STMS, x_STMS - cleanspeech, noise and noisy-speech
				short-time magnitude spectrum samples.
		"""
		s_STMS_sample = []
		d_STMS_sample = []
		x_STMS_sample = []
		for i in tqdm(range(s_sample.shape[0])):
			s_STMS, _ = self.polar_analysis(s_sample[i,0:wav_len[i]])
			d_STMS, _ = self.polar_analysis(d_sample[i,0:wav_len[i]])
			x_STMS, _ = self.polar_analysis(x_sample[i,0:wav_len[i]])
			s_STMS_sample.append(np.squeeze(s_STMS.numpy()))
			d_STMS_sample.append(np.squeeze(d_STMS.numpy()))
			x_STMS_sample.append(np.squeeze(x_STMS.numpy()))
		s_STMS_sample = np.vstack(s_STMS_sample)
		d_STMS_sample = np.vstack(d_STMS_sample)
		x_STMS_sample = np.vstack(x_STMS_sample)
		return s_STMS_sample, d_STMS_sample, x_STMS_sample

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

	def stats(self, s_sample, d_sample, x_sample, wav_len):
		"""
		Compute statistics for map class.

		Argument/s:
			s_sample, d_sample, x_sample, wav_len - clean speech, noise, noisy speech
				samples and their lengths.
		"""
		s_STMS_sample, d_STMS_sample, x_STMS_sample = self.transfrom_stats(s_sample,
			d_sample, x_sample, wav_len)
		xi_sample = self.xi(s_STMS_sample, d_STMS_sample)
		self.xi_map.stats(xi_sample)

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
		Maximum likelihood a posteriori SNR estimate.

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

	def stats(self, s_sample, d_sample, x_sample, wav_len):
		"""
		Compute statistics for map class.

		Argument/s:
			s_sample, d_sample, x_sample, wav_len - clean speech, noise, noisy speech
				samples and their lengths.
		"""
		s_STMS_sample, d_STMS_sample, x_STMS_sample = self.transfrom_stats(s_sample,
			d_sample, x_sample, wav_len)
		gamma_sample = self.gamma(x_STMS_sample, d_STMS_sample)
		self.gamma_map.stats(gamma_sample)

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

	def stats(self, s_sample, d_sample, x_sample, wav_len):
		"""
		Compute statistics for map class.

		Argument/s:
			s_sample, d_sample, x_sample, wav_len - clean speech, noise, noisy speech
				samples and their lengths.
		"""
		s_STMS_sample, d_STMS_sample, x_STMS_sample = self.transfrom_stats(s_sample,
			d_sample, x_sample, wav_len)
		xi_sample = self.xi(s_STMS_sample, d_STMS_sample)
		self.xi_map.stats(xi_sample)
		gamma_sample = self.gamma(x_STMS_sample, d_STMS_sample)
		self.gamma_map.stats(gamma_sample)

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
		if self.gain == 'ibm':
			G_hat = tf.cast(tf.math.greater(G_hat, 0.5), tf.float32)
		y_STMS = tf.math.multiply(x_STMS, G_hat)
		return self.polar_synthesis(y_STMS, x_STPS)

class MagMag(MagTgt):
	"""
	Magnitude spectrum input and target.
	"""
	def __init__(self, N_d, N_s, K, f_s, mag_map_type, mag_map_params):
		super().__init__(N_d, N_s, K, f_s)
		"""
		Argument/s
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
			mag_map_type - clean-speech STMS map type.
			mag_map_params - parameters for the clean-speech STMS map.
		"""
		self.n_feat = math.ceil(K/2 + 1)
		self.n_outp = self.n_feat
		self.mag_map = map_selector(mag_map_type, mag_map_params)

	def stats(self, s_sample, d_sample, x_sample, wav_len):
		"""
		Compute statistics for map class.

		Argument/s:
			s_sample, d_sample, x_sample, wav_len - clean speech, noise, noisy speech
				samples and their lengths.
		"""
		s_STMS_sample, d_STMS_sample, x_STMS_sample = self.transfrom_stats(s_sample,
			d_sample, x_sample, wav_len)
		self.mag_map.stats(s_STMS_sample)

	def example(self, s, d, s_len, d_len, snr):
		"""
		Compute example for Deep Xi, i.e. observation (noisy-speech STMS)
		and target (mapped clean-speech STMS).

		Argument/s:
			s - clean speech (dtype=tf.int32).
			d - noise (dtype=tf.int32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level.

		Returns:
			x_STMS - noisy-speech short-time magnitude spectrum.
			s_STMS_bar - mapped clean-speech short-time magnitude spectrum.
			n_frames - number of time-domain frames.
		"""
		s, d, x, n_frames = self.mix(s, d, s_len, d_len, snr)
		s_STMS, _ = self.polar_analysis(s)
		x_STMS, _ = self.polar_analysis(x)
		s_STMS_bar = self.mag_map.map(s_STMS)
		return x_STMS, s_STMS_bar, n_frames

	def enhanced_speech(self, x_STMS, x_STPS, s_STMS_bar_hat, gtype):
		"""
		Compute enhanced speech.

		Argument/s:
			x_STMS - noisy-speech short-time magnitude spectrum.
			x_STPS - noisy-speech short-time phase spectrum.
			s_STMS_bar - mapped clean-speech short-time magnitude spectrum estimate.
			gtype - gain function type.

		Returns:
			enhanced speech.
		"""
		s_STMS_hat = self.mag_map.inverse(s_STMS_bar_hat)
		return self.polar_synthesis(s_STMS_hat, x_STPS)

	def mag_hat(self, s_STMS_bar_hat):
		"""
		Clean-speech magnitude spectrum estimate.

		Argument/s:
			s_STMS_bar_hat - mapped clean-speech magnitude spectrum estimate.

		Returns:
			s_STMS_hat - clean-speech magnitude spectrum estimate.
		"""
		s_STMS_hat = self.mag_map.inverse(s_STMS_bar_hat)
		return s_STMS_hat

class MagSMM(MagTgt):
	"""
	Magnitude spectrum input and spectral magnitude mask (SMM) target.
	"""
	def __init__(self, N_d, N_s, K, f_s, smm_map_type, smm_map_params):
		super().__init__(N_d, N_s, K, f_s)
		"""
		Argument/s
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
			smm_map_type - clean-speech STMS map type.
			smm_map_params - parameters for the clean-speech STMS map.
		"""
		self.n_feat = math.ceil(K/2 + 1)
		self.n_outp = self.n_feat
		# self.smm_map = map_selector(smm_map_type, smm_map_params)

	def stats(self, s_sample, d_sample, x_sample, wav_len):
		"""
		Compute statistics for map class.

		Argument/s:
			s_sample, d_sample, x_sample, wav_len - clean speech, noise, noisy speech
				samples and their lengths.
		"""
		pass
		# s_STMS_sample, d_STMS_sample, x_STMS_sample = self.transfrom_stats(s_sample,
		# 	d_sample, x_sample, wav_len)
		# smm_sample = tf.math.truediv(s_STMS_sample, x_STMS_sample)
		# self.smm_map.stats(smm_sample)

	def example(self, s, d, s_len, d_len, snr):
		"""
		Compute example for Deep Xi, i.e. observation (noisy-speech STMS)
		and target (mapped SMM).

		Argument/s:
			s - clean speech (dtype=tf.int32).
			d - noise (dtype=tf.int32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level.

		Returns:
			x_STMS - noisy-speech short-time magnitude spectrum.
			smm_bar - mapped SMM.
			n_frames - number of time-domain frames.
		"""
		s, d, x, n_frames = self.mix(s, d, s_len, d_len, snr)
		s_STMS, _ = self.polar_analysis(s)
		x_STMS, _ = self.polar_analysis(x)
		smm = tf.math.truediv(s_STMS, x_STMS)
		smm_bar = tf.clip_by_value(smm, 0.0, 5.0)
		return x_STMS, smm_bar, n_frames

	def enhanced_speech(self, x_STMS, x_STPS, smm_bar_hat, gtype):
		"""
		Compute enhanced speech.

		Argument/s:
			x_STMS - noisy-speech short-time magnitude spectrum.
			x_STPS - noisy-speech short-time phase spectrum.
			smm_bar_hat - mapped SMM estimate.
			gtype - gain function type.

		Returns:
			enhanced speech.
		"""
		# smm_hat = self.smm_map.inverse(smm_bar_hat)
		smm_hat = smm_bar_hat
		s_STMS_hat = tf.math.multiply(smm_hat, x_STMS)
		return self.polar_synthesis(s_STMS_hat, x_STPS)

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

	def stats(self, s_sample, d_sample, x_sample, wav_len):
		"""
		Compute statistics for map class.

		Argument/s:
			s_sample, d_sample, x_sample, wav_len - clean speech, noise, noisy speech
				samples and their lengths.
		"""
		s_STMS_sample, d_STMS_sample, x_STMS_sample = self.transfrom_stats(s_sample,
			d_sample, x_sample, wav_len)
		xi_sample = self.xi(s_STMS_sample, d_STMS_sample)
		self.xi_map.stats(xi_sample)

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

class STDCTXiCD(InputTarget):
	"""
	__________________________.
	"""
	def __init__(self, N_d, N_s, K, f_s, xi_map_type, xi_map_params,
		cd_map_type, cd_map_params):
		super().__init__(N_d, N_s, K, f_s)
		"""
		Argument/s
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
			_ - _____________.
		"""
		self.n_feat = K
		self.n_outp = self.n_feat*2
		self.xi_map = map_selector(xi_map_type, xi_map_params)
		self.cd_map = map_selector(cd_map_type, cd_map_params)

	def observation(self, x):
		"""
	    An observation for Deep Xi (noisy-speech __________).

		Argument/s:
			x - noisy speech (dtype=tf.int32).
			x_len - noisy speech length without padding (samples).

		Returns:
			x_STDCT - noisy-speech short-time discrete cosine transform.
		"""
		x = self.normalise(x)
		x_STDCT = self.stdct_analysis(x)
		return x_STDCT, None

	def stats(self, s_sample, d_sample, x_sample, wav_len):
		"""
		Compute statistics for map class.

		Argument/s:
			s_sample, d_sample, x_sample, wav_len - clean speech, noise, noisy speech
				samples and their lengths.
		"""
		s_STDCT_sample, d_STDCT_sample, x_STDCT_sample = self.transfrom_stats(s_sample,
			d_sample, x_sample, wav_len)
		xi_sample = self.xi(s_STDCT_sample, d_STDCT_sample)
		self.xi_map.stats(xi_sample)
		cd_sample = self.cd(s_STDCT_sample, d_STDCT_sample)
		self.cd_map.stats(cd_sample)

	def transfrom_stats(self, s_sample, d_sample, x_sample, wav_len):
		"""
		Transforms time-domain sample to short-time discrete cosine transform
		sample.

		Argument/s:
			s_sample, d_sample, x_sample, wav_len - clean speech, noise,
				noisy speech samples and their lengths.

		Returns:
			s_STDCT, d_STDCT, x_STDCT - clean-speech, noise and noisy-speech
				short-time discrete cosine transform samples.
		"""
		s_STDCT_sample = []
		d_STDCT_sample = []
		x_STDCT_sample = []
		for i in tqdm(range(s_sample.shape[0])):
			s_STDCT = self.stdct_analysis(s_sample[i,0:wav_len[i]])
			d_STDCT = self.stdct_analysis(d_sample[i,0:wav_len[i]])
			x_STDCT = self.stdct_analysis(x_sample[i,0:wav_len[i]])
			s_STDCT_sample.append(np.squeeze(s_STDCT.numpy()))
			d_STDCT_sample.append(np.squeeze(d_STDCT.numpy()))
			x_STDCT_sample.append(np.squeeze(x_STDCT.numpy()))
		s_STDCT_sample = np.vstack(s_STDCT_sample)
		d_STDCT_sample = np.vstack(d_STDCT_sample)
		x_STDCT_sample = np.vstack(x_STDCT_sample)
		return s_STDCT_sample, d_STDCT_sample, x_STDCT_sample

	def example(self, s, d, s_len, d_len, snr):
		"""
		Compute example for Deep Xi, i.e. observation (noisy-speech STDCT)
		and target (________).

		Argument/s:
			s - clean speech (dtype=tf.int32).
			d - noise (dtype=tf.int32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level.

		Returns:
			x_STMS - noisy-speech short-time magnitude spectrum.
			___ - _______________
			n_frames - number of time-domain frames.
		"""
		s, d, x, n_frames = self.mix(s, d, s_len, d_len, snr)
		s_STDCT = self.stdct_analysis(s)
		d_STDCT = self.stdct_analysis(d)
		x_STDCT = self.stdct_analysis(x)
		xi = self.xi(s_STDCT, d_STDCT)
		xi_bar = self.xi_map.map(xi)
		cd = self.cd(s_STDCT, d_STDCT)
		cd_bar = self.cd_map.map(cd)
		xi_cd_map = tf.concat([xi_bar, cd_bar], axis=-1)
		return x_STDCT, xi_cd_map, n_frames

	def enhanced_speech(self, x_STDCT, dummy, xi_cd_bar_hat, gtype):
		"""
		Compute enhanced speech.

		Argument/s:
			x_STDCT - noisy-speech short-time magnitude spectrum.
			dummy - dummy variable (not used).
			____ - _____________________.
			gtype - gain function type.

		Returns:
			enhanced speech.
		"""
		xi_bar_hat, cd_bar_hat = tf.split(xi_cd_bar_hat,
			num_or_size_splits=2, axis=-1)
		xi_hat = self.xi_map.inverse(xi_bar_hat)
		gamma_hat = tf.math.add(xi_hat, self.one)
		cd_hat = self.cd_map.inverse(cd_bar_hat)
		cdm_hat = tf.math.greater(cd_hat, 0.0)
		y_STDCT = tf.math.multiply(x_STDCT, gfunc(xi_hat, gamma_hat, gtype, cdm_hat))
		return self.stdct_synthesis(y_STDCT)

	def xi_hat(self, xi_cd_bar_hat):
		"""
		A priori SNR estimate.

		Argument/s:
			xi_cd_bar_hat - mapped a priori SNR and ______________ estimate.

		Returns:
			xi_hat - a priori SNR estimate.
		"""
		xi_bar_hat, _ = tf.split(xi_cd_bar_hat, num_or_size_splits=2, axis=-1)
		xi_hat = self.xi_map.inverse(xi_bar_hat)
		return xi_hat

	def cd_hat(self, xi_cd_bar_hat):
		"""
		___________ estimate.

		Argument/s:
			xi_cd_bar_hat - mapped a priori SNR and ______________ estimate.

		Returns:
			cd_hat - ________________ estimate.
		"""
		_, cd_bar_hat = tf.split(xi_cd_bar_hat, num_or_size_splits=2, axis=-1)
		cd_hat = self.cd_map.inverse(cd_bar_hat)
		return cd_hat
