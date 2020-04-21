## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from deepxi.gain import gfunc
from deepxi.network.cnn import TCN
from deepxi.network.rnn import ResLSTM
from deepxi.sig import DeepXiInput
from deepxi.utils import read_wav, save_wav, save_mat
from pesq import pesq
from pystoi import stoi
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.lib.io import file_io
# from tensorflow.python.util.compat import collections_abc
from tqdm import tqdm
import deepxi.se_batch as batch
import csv, math, os, random # collections, io, six
import numpy as np
import tensorflow as tf

# [1] Nicolson, A. and Paliwal, K.K., 2019. Deep learning for
# 	  minimum mean-square error approaches to speech enhancement.
# 	  Speech Communication, 111, pp.44-55.

class DeepXi(DeepXiInput):
	"""
	Deep Xi model from [1].
	"""
	def __init__(
		self,
		N_d,
		N_s,
		NFFT,
		f_s,
		network,
		min_snr,
		max_snr,
		ver='VERSION_NAME',
		**kwargs
		):
		"""
		Argument/s
			N_d - window duration (samples).
			N_s - window shift (samples).
			NFFT - number of DFT bins.
			f_s - sampling frequency.
			network - network type.
			min_snr - minimum SNR level for training.
			max_snr - maximum SNR level for training.
			ver - version name.
		"""
		super().__init__(N_d, N_s, NFFT, f_s)
		self.min_snr = min_snr
		self.max_snr = max_snr
		self.ver = ver
		self.n_feat = math.ceil(self.NFFT/2 + 1)
		self.n_outp = self.n_feat
		self.inp = Input(name='inp', shape=[None, self.n_feat], dtype='float32')
		self.mask = tf.keras.layers.Masking(mask_value=0.0)(self.inp)
		if network == 'TCN': self.network = TCN(
			inp=self.mask,
			n_outp=self.n_outp,
			n_blocks=kwargs['n_blocks'],
			d_model=kwargs['d_model'],
			d_f=kwargs['d_f'],
			k=kwargs['k'],
			max_d_rate=kwargs['max_d_rate'],
			)
		elif network == 'ResLSTM': self.network = ResLSTM(
			inp=self.mask,
			n_outp=self.n_outp,
			n_blocks=kwargs['n_blocks'],
			d_model=kwargs['d_model'],
			)
		else: raise ValueError('Invalid network type.')
		self.model = Model(inputs=self.inp, outputs=self.network.outp)
		self.model.summary()
		if not os.path.exists("log/summary"):
			os.makedirs("log/summary")
		with open("log/summary/" + self.ver + ".txt", "w") as f:
		    self.model.summary(print_fn=lambda x: f.write(x + '\n'))

	def train(
		self,
		train_s_list,
		train_d_list,
		model_path='model',
		val_s=None,
		val_d=None,
		val_s_len=None,
		val_d_len=None,
		val_snr=None,
		val_flag=True,
		val_save_path=None,
		mbatch_size=8,
		max_epochs=200,
		resume_epoch=0,
		stats_path=None,
		sample_size=None,
		eval_example=False,
		save_model=True,
		log_iter=False,
		):
		"""
		Deep Xi training.

		Argument/s:
			train_s_list - clean-speech training list.
			train_d_list - noise training list.
			model_path - model save path.
			val_s - clean-speech validation batch.
			val_d - noise validation batch.
			val_s_len - clean-speech validation sequence length batch.
			val_d_len - noise validation sequence length batch.
			val_snr - SNR validation batch.
			val_flag - perform validation.
			val_save_path - validation batch save path.
			mbatch_size - mini-batch size.
			max_epochs - maximum number of epochs.
			resume_epoch - epoch to resume training from.
			stats_path - path to save sample statistics.
			sample_size - sample size.
			eval_example - evaluate a mini-batch of training examples.
			save_model - save architecture, weights, and training configuration.
			log_iter - log training loss for each training iteration.
		"""
		self.train_s_list = train_s_list
		self.train_d_list = train_d_list
		self.mbatch_size = mbatch_size
		self.n_examples = len(self.train_s_list)
		self.n_iter = math.ceil(self.n_examples/mbatch_size)

		self.sample_stats(stats_path, sample_size, train_s_list, train_d_list)

		train_dataset = self.dataset(max_epochs-resume_epoch)

		if val_flag:
			val_set = self.val_batch(val_save_path, val_s, val_d, val_s_len, val_d_len, val_snr)
			val_steps = len(val_set[0])
		else: val_set, val_steps = None, None

		if eval_example:
			print("Saving a mini-batch of training examples in .mat files...")
			x_STMS_batch, xi_bar_batch, seq_mask_batch = list(train_dataset.take(1).as_numpy_iterator())[0]
			save_mat('./x_STMS_batch.mat', x_STMS_batch, 'x_STMS_batch')
			save_mat('./xi_bar_batch.mat', xi_bar_batch, 'xi_bar_batch')
			save_mat('./seq_mask_batch.mat', seq_mask_batch, 'seq_mask_batch')
			print("Testing if add_noise() works correctly...")
			s, d, s_len, d_len, snr_tgt = self.wav_batch(train_s_list[0:mbatch_size], train_d_list[0:mbatch_size])
			(_, s, d) = self.add_noise_batch(self.normalise(s), self.normalise(d), s_len, d_len, snr_tgt)
			for (i, _) in enumerate(s):
				snr_act = self.snr_db(s[i][0:s_len[i]], d[i][0:d_len[i]])
				print('SNR target|actual: {:.2f}|{:.2f} (dB).'.format(snr_tgt[i], snr_act))

		if not os.path.exists(model_path): os.makedirs(model_path)
		if not os.path.exists("log"): os.makedirs("log")
		if not os.path.exists("log/iter"): os.makedirs("log/iter")

		callbacks = []
		callbacks.append(CSVLogger("log/" + self.ver + ".csv", separator=',', append=True))
		if save_model: callbacks.append(SaveWeights(model_path))
		# if log_iter: callbacks.append(CSVLoggerIter("log/iter/" + self.ver + ".csv", separator=',', append=True))

		if resume_epoch > 0: self.model.load_weights(model_path + "/epoch-" +
			str(resume_epoch-1) + "/variables/variables" )

		self.model.compile(
			sample_weight_mode="temporal",
			loss="binary_crossentropy",
			optimizer=Adam(lr=0.001, clipvalue=1.0)
			)

		self.model.fit(
			x=train_dataset,
			initial_epoch=resume_epoch,
			epochs=max_epochs,
			steps_per_epoch=self.n_iter,
			callbacks=callbacks,
			validation_data=val_set,
			validation_steps=val_steps
			)

	def infer( ## NEED TO ADD DeepMMSE
		self,
		test_x,
		test_x_len,
		test_x_base_names,
		test_epoch,
		model_path='model',
		out_type='y',
		gain='mmse-lsa',
		out_path='out',
		stats_path=None
		):
		"""
		Deep Xi inference. The specified 'out_type' is saved.

		Argument/s:
			test_x - noisy-speech test batch.
			test_x_len - noisy-speech test batch lengths.
			test_x_base_names - noisy-speech base names.
			test_epoch - epoch to test.
			model_path - path to model directory.
			out_type - output type (see deepxi/args.py).
			gain - gain function (see deepxi/args.py).
			out_path - path to save output files.
			stats_path - path to the saved statistics.
		"""
		if out_type == 'xi_hat': out_path = out_path + '/xi_hat'
		elif out_type == 'y': out_path = out_path + '/y/' + gain
		elif out_type == 'ibm_hat': out_path = out_path + '/ibm_hat'
		elif out_type == 'deepmmse': out_path = out_path + '/deepmmse'
		else: raise ValueError('Invalid output type.')
		if not os.path.exists(out_path): os.makedirs(out_path)

		if test_epoch < 1: raise ValueError("test_epoch must be greater than 0.")

		self.sample_stats(stats_path)
		self.model.load_weights(model_path + '/epoch-' + str(test_epoch-1) +
			'/variables/variables' )

		print("Processing observations...")
		x_STMS_batch, x_STPS_batch, n_frames = self.observation_batch(test_x, test_x_len)
		print("Performing inference...")
		xi_bar_hat_batch = self.model.predict(x_STMS_batch, batch_size=1, verbose=1)

		print("Performing synthesis...")
		batch_size = len(test_x_len)
		for i in tqdm(range(batch_size)):
			base_name = test_x_base_names[i]
			x_STMS = x_STMS_batch[i,:n_frames[i],:]
			x_STPS = x_STPS_batch[i,:n_frames[i],:]
			xi_bar_hat = xi_bar_hat_batch[i,:n_frames[i],:]
			xi_hat = self.xi_hat(xi_bar_hat)
			if out_type == 'xi_hat': save_mat(args.out_path + '/' + base_name + '.mat', xi_hat, 'xi_hat')
			elif out_type == 'y':
				y_STMS = np.multiply(x_STMS, gfunc(xi_hat, xi_hat+1, gtype=gain))
				y = self.polar_synthesis(y_STMS, x_STPS).numpy()
				save_wav(out_path + '/' + base_name + '.wav', y, self.f_s)
			elif out_type == 'ibm_hat':
				ibm_hat = np.greater(xi_hat, 1.0).astype(bool)
				save_mat(out_path + '/' + base_name + '.mat', ibm_hat, 'ibm_hat')
			elif out_type == 'deepmmse':
				d_PSD_hat = np.multiply(np.square(x_STMS), gfunc(xi_hat, xi_hat+1, gtype='deepmmse'))
				save_mat(out_path + '/' + base_name + '.mat', d_PSD_hat, 'd_psd_hat')
			else: raise ValueError('Invalid output type.')

	def test(
		self,
		test_x,
		test_x_len,
		test_x_base_names,
		test_s,
		test_s_len,
		test_s_base_names,
		test_epoch,
		model_path='model',
		gain='mmse-lsa',
		stats_path=None
		):
		"""
		Deep Xi testing. Objective measures are used to evaluate the performance
		of Deep Xi.

		Argument/s:
			test_x - noisy-speech test batch.
			test_x_len - noisy-speech test batch lengths.
			test_x_base_names - noisy-speech base names.
			test_s - clean-speech test batch.
			test_s_len - clean-speech test batch lengths.
			test_s_base_names - clean-speech base names.
			test_epoch - epoch to test.
			model_path - path to model directory.
			gain - gain function (see deepxi/args.py).
			stats_path - path to the saved statistics.

		"""
		if not isinstance(test_epoch, list): test_epoch = [test_epoch]
		if not isinstance(gain, list): gain = [gain]
		for e in test_epoch:
			for g in gain:

				if e < 1: raise ValueError("test_epoch must be greater than 0.")

				self.sample_stats(stats_path)
				self.model.load_weights(model_path + '/epoch-' + str(e-1) +
					'/variables/variables' )

				print("Processing observations...")
				x_STMS_batch, x_STPS_batch, n_frames = self.observation_batch(test_x, test_x_len)
				print("Performing inference...")
				xi_bar_hat_batch = self.model.predict(x_STMS_batch, batch_size=1, verbose=1)

				print("Performing synthesis and objective scoring...")
				results = {}
				batch_size = len(test_x_len)
				for i in tqdm(range(batch_size)):
					base_name = test_x_base_names[i]
					x_STMS = x_STMS_batch[i,:n_frames[i],:]
					x_STPS = x_STPS_batch[i,:n_frames[i],:]
					xi_bar_hat = xi_bar_hat_batch[i,:n_frames[i],:]
					xi_hat = self.xi_hat(xi_bar_hat)
					y_STMS = np.multiply(x_STMS, gfunc(xi_hat, xi_hat+1, gtype=g))
					y = self.polar_synthesis(y_STMS, x_STPS).numpy()

					for (j, basename) in enumerate(test_s_base_names):
						if basename in test_x_base_names[i]: ref_idx = j

					s = self.normalise(test_s[ref_idx,0:test_s_len[ref_idx]]).numpy()
					y = y[0:len(s)]

					noise_source = test_x_base_names[i].split("_")[-2]
					snr_level = int(test_x_base_names[i].split("_")[-1][:-2])

					results = self.add_score(results, (noise_source, snr_level, 'STOI'),
						100*stoi(s, y, self.f_s, extended=False))
					results = self.add_score(results, (noise_source, snr_level, 'eSTOI'),
						100*stoi(s, y, self.f_s, extended=True))
					results = self.add_score(results, (noise_source, snr_level, 'PESQ'),
						pesq(self.f_s, s, y, 'nb'))
					results = self.add_score(results, (noise_source, snr_level, 'MOS-LQO'),
						pesq(self.f_s, s, y, 'wb'))

				noise_sources, snr_levels, metrics = set(), set(), set()
				for key, value in results.items():
					noise_sources.add(key[0])
					snr_levels.add(key[1])
					metrics.add(key[2])

				if not os.path.exists("log/results"): os.makedirs("log/results")

				with open("log/results/" + self.ver + "_e" + str(e) + '_' + g + ".csv", "w") as f:
					f.write("noise,snr_db")
					for k in sorted(metrics): f.write(',' + k)
					f.write('\n')
					for i in sorted(noise_sources):
						for j in sorted(snr_levels):
							f.write("{},{}".format(i, j))
							for k in sorted(metrics):
								if (i, j, k) in results.keys():
									f.write(",{:.2f}".format(np.mean(results[(i,j,k)])))
							f.write('\n')

				avg_results = {}
				for i in sorted(noise_sources):
					for j in sorted(snr_levels):
						if (j >= self.min_snr) and (j <= self.max_snr):
							for k in sorted(metrics):
								if (i, j, k) in results.keys():
									avg_results = self.add_score(avg_results, k, results[(i,j,k)])

				if not os.path.exists("log/results/average.csv"):
					with open("log/results/average.csv", "w") as f:
						f.write("ver")
						for i in sorted(metrics): f.write("," + i)
						f.write('\n')

				with open("log/results/average.csv", "a") as f:
					f.write(self.ver + "_e" + str(e) + '_' + g)
					for i in sorted(metrics):
						if i in avg_results.keys():
							f.write(",{:.2f}".format(np.mean(avg_results[i])))
					f.write('\n')

	def sample_stats(
		self,
		stats_path='data',
		sample_size=1000,
		train_s_list=None,
		train_d_list=None
		):
		"""
		Computes statistics for each frequency component of the instantaneous a priori SNR
		in dB over a sample of the training set. The statistics are then used to map the
		instantaneous a priori SNR in dB between 0 and 1 using its cumulative distribution
		function. This forms the mapped a priori SNR (the training target).

		Argument/s:
			stats_path - path to the saved statistics.
			sample_size - number of training examples to compute the statistics from.
			train_s_list - train clean speech list.
			train_d_list - train noise list.
		"""
		if os.path.exists(stats_path + '/stats.npz'):
			print('Loading sample statistics...')
			with np.load(stats_path + '/stats.npz') as stats:
				self.mu = stats['mu_hat']
				self.sigma = stats['sigma_hat']
		elif train_s_list == None:
			raise ValueError('No stats.npz file exists. data/stats.p is available here: https://github.com/anicolson/DeepXi/blob/master/data/stats.npz.')
		else:
			print('Finding sample statistics...')
			s_sample_list = random.sample(self.train_s_list, sample_size)
			d_sample_list = random.sample(self.train_d_list, sample_size)
			s_sample, d_sample, s_sample_len, d_sample_len, snr_sample = self.wav_batch(s_sample_list, d_sample_list)
			snr_sample = np.random.randint(self.min_snr, self.max_snr + 1, sample_size)
			samples = []
			for i in tqdm(range(s_sample.shape[0])):
				s_STMS, d_STMS, _, _ = self.mix(s_sample[i:i+1], d_sample[i:i+1], s_sample_len[i:i+1],
					d_sample_len[i:i+1], snr_sample[i:i+1])
				xi_db = self.xi_db(s_STMS, d_STMS) # instantaneous a priori SNR (dB).
				samples.append(np.squeeze(xi_db.numpy()))
			samples = np.vstack(samples)
			if len(samples.shape) != 2: raise ValueError('Incorrect shape for sample.')
			stats = {'mu_hat': np.mean(samples, axis=0), 'sigma_hat': np.std(samples, axis=0)}
			self.mu, self.sigma = stats['mu_hat'], stats['sigma_hat']
			if not os.path.exists(stats_path): os.makedirs(stats_path)
			np.savez(stats_path + '/stats.npz', mu_hat=stats['mu_hat'], sigma_hat=stats['sigma_hat'])
			save_mat(stats_path + '/stats.mat', stats, 'stats')
			print('Sample statistics saved.')

	def dataset(self, n_epochs, buffer_size=16):
		"""
		Used to create a tf.data.Dataset for training.

		Argument/s:
			n_epochs - number of epochs to generate.
			buffer_size - number of mini-batches to keep in buffer.

		Returns:
			dataset - tf.data.Dataset
		"""
		dataset = tf.data.Dataset.from_generator(
			self.mbatch_gen,
			(tf.float32, tf.float32, tf.float32),
			(tf.TensorShape([None, None, self.n_feat]),
				tf.TensorShape([None, None, self.n_outp]),
				tf.TensorShape([None, None])),
			[tf.constant(n_epochs)]
			)
		dataset = dataset.prefetch(buffer_size)
		return dataset

	def mbatch_gen(self, n_epochs):
		"""
		A generator that yields a mini-batch of training examples.

		Argument/s:
			n_epochs - number of epochs to generate.

		Returns:
			x_STMS_mbatch - mini-batch of observations (noisy speech short-time magnitude spectum).
			xi_bar_mbatch - mini-batch of targets (mapped a priori SNR).
			seq_mask_mbatch - mini-batch of sequence masks.
		"""
		for _ in range(n_epochs):
			random.shuffle(self.train_s_list)
			start_idx, end_idx = 0, self.mbatch_size
			for _ in range(self.n_iter):
				s_mbatch_list = self.train_s_list[start_idx:end_idx]
				d_mbatch_list = random.sample(self.train_d_list, end_idx-start_idx)
				s_mbatch, d_mbatch, s_mbatch_len, d_mbatch_len, snr_mbatch = \
					self.wav_batch(s_mbatch_list, d_mbatch_list)
				x_STMS_mbatch, xi_bar_mbatch, n_frames_mbatch = \
					self.example(s_mbatch, d_mbatch, s_mbatch_len,
					d_mbatch_len, snr_mbatch)
				seq_mask_mbatch = tf.cast(tf.sequence_mask(n_frames_mbatch), tf.float32)
				start_idx += self.mbatch_size; end_idx += self.mbatch_size
				if end_idx > self.n_examples: end_idx = self.n_examples
				yield x_STMS_mbatch, xi_bar_mbatch, seq_mask_mbatch

	def val_batch(
		self,
		save_path,
		val_s,
		val_d,
		val_s_len,
		val_d_len,
		val_snr
		):
		"""
		Creates and saves the examples for the validation set. If
		already saved, the function will load the batch of examples.

		Argument/s:
			save_path - path to save the validation batch.
			val_s - validation clean speech waveforms.
			val_d - validation noise waveforms.
			val_s_len - validation clean speech waveform lengths.
			val_d_len - validation noise waveform lengths.
			val_snr - validation SNR levels.

		Returns:
			x_STMS_batch - batch of observations (noisy speech short-time magnitude spectum).
			xi_bar_batch - batch of targets (mapped a priori SNR).
			seq_mask_batch - batch of sequence masks.
		"""
		if not os.path.exists(save_path): os.makedirs(save_path)
		if os.path.exists(save_path + '/val_batch.npz'):
			print('Loading validation batch...')
			with np.load(save_path + '/val_batch.npz') as data:
				x_STMS_batch = data['val_inp']
				xi_bar_batch = data['val_tgt']
				seq_mask_batch =  data['val_seq_mask']
		else:
			print('Creating validation batch...')
			batch_size = len(val_s)
			max_n_frames = self.n_frames(max(val_s_len))
			x_STMS_batch = np.zeros([batch_size, max_n_frames, self.n_feat], np.float32)
			xi_bar_batch = np.zeros([batch_size, max_n_frames, self.n_feat], np.float32)
			seq_mask_batch = np.zeros([batch_size, max_n_frames], np.float32)
			for i in tqdm(range(batch_size)):
				x_STMS, xi_bar, _ = self.example(val_s[i:i+1], val_d[i:i+1],
					val_s_len[i:i+1], val_d_len[i:i+1], val_snr[i:i+1])
				n_frames = self.n_frames(val_s_len[i])
				x_STMS_batch[i,:n_frames,:] = x_STMS.numpy()
				xi_bar_batch[i,:n_frames,:] = xi_bar.numpy()
				seq_mask_batch[i,:n_frames] = tf.cast(tf.sequence_mask(n_frames), tf.float32)
			np.savez(save_path + '/val_batch.npz', val_inp=x_STMS_batch,
				val_tgt=xi_bar_batch, val_seq_mask=seq_mask_batch)
		return x_STMS_batch, xi_bar_batch, seq_mask_batch

	def observation_batch(self, x_batch, x_batch_len):
		"""
		Computes observations (noisy-speech STMS) from noisy speech recordings.

		Argument/s:
			x_batch - noisy-speech batch.
			x_batch_len - noisy-speech batch lengths.

		Returns:
			x_STMS_batch - batch of observations (noisy-speech short-time magnitude spectrums).
			x_STPS_batch - batch of noisy-speech short-time phase spectrums.
			n_frames_batch - number of frames in each observation.
		"""
		batch_size = len(x_batch)
		max_n_frames = self.n_frames(max(x_batch_len))
		x_STMS_batch = np.zeros([batch_size, max_n_frames, self.n_feat], np.float32)
		x_STPS_batch = np.zeros([batch_size, max_n_frames, self.n_feat], np.float32)
		n_frames_batch = [self.n_frames(i) for i in x_batch_len]
		for i in tqdm(range(batch_size)):
			x_STMS, x_STPS = self.observation(x_batch[i,:x_batch_len[i]])
			x_STMS_batch[i,:n_frames_batch[i],:] = x_STMS.numpy()
			x_STPS_batch[i,:n_frames_batch[i],:] = x_STPS.numpy()
		return x_STMS_batch, x_STPS_batch, n_frames_batch

	def wav_batch(self, s_list, d_list):
		"""
		Loads .wav files into batches.

		Argument/s:
			s_list - clean-speech list.
			d_list - noise list.

		Returns:
			s_batch - batch of clean speech.
			d_batch - batch of noise.
			s_batch_len - sequence length of each clean speech waveform.
			d_batch_len - sequence length of each noise waveform.
			snr_batch - batch of SNR levels.
		"""
		batch_size = len(s_list)
		max_len = max([dic['wav_len'] for dic in s_list])
		s_batch = np.zeros([batch_size, max_len], np.int16)
		d_batch = np.zeros([batch_size, max_len], np.int16)
		s_batch_len = np.zeros(batch_size, np.int32)
		for i in range(batch_size):
			(wav, _) = read_wav(s_list[i]['file_path'])
			s_batch[i,:s_list[i]['wav_len']] = wav
			s_batch_len[i] = s_list[i]['wav_len']
			flag = True
			while flag:
				if d_list[i]['wav_len'] < s_batch_len[i]: d_list[i] = random.choice(self.train_d_list)
				else: flag = False
			(wav, _) = read_wav(d_list[i]['file_path'])
			rand_idx = np.random.randint(0, 1+d_list[i]['wav_len']-s_batch_len[i])
			d_batch[i,:s_batch_len[i]] = wav[rand_idx:rand_idx+s_batch_len[i]]
		d_batch_len = s_batch_len
		snr_batch = np.random.randint(self.min_snr, self.max_snr+1, batch_size)
		return s_batch, d_batch, s_batch_len, d_batch_len, snr_batch

	def add_score(self, dict, key, score):
		"""
		Adds score/s to the list for the given key.

		Argument/s:
			dict - dictionary with condition as keys and a list of objective
				scores as values.
			key - noisy-speech conditions.
			score - objective score.

		Returns:
			dict - updated dictionary.
		"""
		if isinstance(score, list):
			if key in dict.keys(): dict[key].extend(score)
			else: dict[key] = score
		else:
			if key in dict.keys(): dict[key].append(score)
			else: dict[key] = [score]
		return dict

#############################################################
## CREATE deepxi/callbacks.py AND MOVE THE CALLBACKS THERE ##
#############################################################

class SaveWeights(Callback):  ### RENAME TO SaveModel
	"""
	"""
	def __init__(self, model_path):
		"""
		"""
		super(SaveWeights, self).__init__()
		self.model_path = model_path

	def on_epoch_end(self, epoch, logs=None):
		"""
		"""
		self.model.save(self.model_path + "/epoch-" + str(epoch))

# class CSVLoggerIter(Callback):
# 	"""
# 	for each training iteration
# 	"""
# 	def __init__(self, filename, separator=',', append=False):
# 		"""
# 		"""
# 		self.sep = separator
# 		self.filename = filename
# 		self.append = append
# 		self.writer = None
# 		self.keys = None
# 		self.append_header = True
# 		if six.PY2:
# 			self.file_flags = 'b'
# 			self._open_args = {}
# 		else:
# 			self.file_flags = ''
# 			self._open_args = {'newline': '\n'}
# 		super(CSVLoggerIter, self).__init__()
#
# 	def on_train_begin(self, logs=None):
# 		"""
# 		"""
# 		if self.append:
# 			if file_io.file_exists(self.filename):
# 				with open(self.filename, 'r' + self.file_flags) as f:
# 					self.append_header = not bool(len(f.readline()))
# 			mode = 'a'
# 		else:
# 			mode = 'w'
# 		self.csv_file = io.open(self.filename, mode + self.file_flags,
# 			**self._open_args)
#
# 	def on_train_batch_end(self, batch, logs=None):
# 		"""
# 		"""
# 		logs = logs or {}
#
# 		def handle_value(k):
# 			is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
# 			if isinstance(k, six.string_types):
# 				return k
# 			elif isinstance(k, collections_abc.Iterable) and not is_zero_dim_ndarray:
# 				return '"[%s]"' % (', '.join(map(str, k)))
# 			else:
# 				return k
#
# 		if self.keys is None:
# 			self.keys = sorted(logs.keys())
#
# 		if self.model.stop_training:
# 			# NA is set so that csv parsers do not fail for the last batch.
# 			logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])
#
# 		if not self.writer:
#
# 			class CustomDialect(csv.excel):
# 				delimiter = self.sep
#
# 			fieldnames = self.keys
# 			if six.PY2:
# 				fieldnames = [unicode(x) for x in fieldnames]
#
# 			self.writer = csv.DictWriter(
# 				self.csv_file,
# 				fieldnames=fieldnames,
# 				dialect=CustomDialect)
# 			if self.append_header:
# 				self.writer.writeheader()
#
# 		row_dict = collections.OrderedDict({'batch': batch})
# 		row_dict.update((key, handle_value(logs[key])) for key in self.keys)
#
# 		self.writer.writerow(row_dict)
# 		self.csv_file.flush()
#
# 	def on_train_end(self, logs=None):
# 		self.csv_file.close()
# 		self.writer = None
