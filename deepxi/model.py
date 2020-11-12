## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from deepxi.gain import gfunc
from deepxi.network.selector import network_selector
from deepxi.inp_tgt import inp_tgt_selector
from deepxi.sig import InputTarget
from deepxi.utils import read_mat, read_wav, save_mat, save_wav
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import Input, Masking
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.lib.io import file_io
from tqdm import tqdm
import csv, math, os, pickle, random
import deepxi.se_batch as batch
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# [1] Nicolson, A. and Paliwal, K.K., 2019. Deep learning for
# 	  minimum mean-square error approaches to speech enhancement.
# 	  Speech Communication, 111, pp.44-55.

class DeepXi():
	"""
	Deep Xi model from [1].
	"""
	def __init__(
		self,
		N_d,
		N_s,
		K,
		f_s,
		inp_tgt_type,
		network_type,
		min_snr,
		max_snr,
		snr_inter,
		log_path,
		sample_dir=None,
		ver='VERSION_NAME',
		train_s_list=None,
		train_d_list=None,
		sample_size=None,
		reset_inp_tgt=False,
		**kwargs
		):
		"""
		Argument/s:
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
			inp_tgt_type - input and target type.
			network_type - network type.
			min_snr - minimum SNR level for training.
			max_snr - maximum SNR level for training.
			stats_dir - path to save sample statistics.
			ver - version name.
			train_s_list - clean-speech training list to compute statistics.
			train_d_list - noise training list to compute statistics.
			sample_size - number of samples to compute the statistics from.
			kwargs - keyword arguments.
		"""
		self.inp_tgt_type = inp_tgt_type
		self.network_type = network_type
		self.min_snr = min_snr
		self.max_snr = max_snr
		self.snr_levels = list(range(self.min_snr, self.max_snr + 1, snr_inter))
		self.ver = ver
		self.train_s_list=train_s_list
		self.train_d_list=train_d_list

		inp_tgt_obj_path = sample_dir + '/' + self.ver + '_inp_tgt.p'
		if os.path.exists(inp_tgt_obj_path) and not reset_inp_tgt:
			with open(inp_tgt_obj_path, 'rb') as f:
				self.inp_tgt = pickle.load(f)
		else:
			self.inp_tgt = inp_tgt_selector(self.inp_tgt_type, N_d, N_s, K, f_s, **kwargs)
			s_sample, d_sample, x_sample, wav_len = self.sample(sample_size, sample_dir)
			self.inp_tgt.stats(s_sample, d_sample, x_sample, wav_len)
			with open(inp_tgt_obj_path, 'wb') as f:
				pickle.dump(self.inp_tgt, f, pickle.HIGHEST_PROTOCOL)

		self.inp = Input(name='inp', shape=[None, self.inp_tgt.n_feat], dtype='float32')
		self.network = network_selector(self.network_type, self.inp,
			self.inp_tgt.n_outp, **kwargs)

		self.model = Model(inputs=self.inp, outputs=self.network.outp)
		self.model.summary()
		if not os.path.exists(log_path + "/summary"):
			os.makedirs(log_path + "/summary")
		with open(log_path + "/summary/" + self.ver + ".txt", "w") as f:
			self.model.summary(print_fn=lambda x: f.write(x + '\n'))

	def train(
		self,
		train_s_list,
		train_d_list,
		mbatch_size,
		max_epochs,
		loss_fnc,
		log_path,
		model_path='model',
		val_s=None,
		val_d=None,
		val_s_len=None,
		val_d_len=None,
		val_snr=None,
		val_flag=True,
		val_save_path=None,
		resume_epoch=0,
		eval_example=False,
		save_model=True,
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
			eval_example - evaluate a mini-batch of training examples.
			save_model - save architecture, weights, and training configuration.
			loss_fnc - loss function.
		"""
		self.train_s_list = train_s_list
		self.train_d_list = train_d_list
		self.mbatch_size = mbatch_size
		self.n_examples = len(self.train_s_list)
		self.n_iter = math.ceil(self.n_examples/mbatch_size)

		train_dataset = self.dataset(max_epochs-resume_epoch)

		if val_flag:
			val_set = self.val_batch(val_save_path, val_s, val_d, val_s_len, val_d_len, val_snr)
			val_steps = len(val_set[0])
		else: val_set, val_steps = None, None

		if not os.path.exists(model_path): os.makedirs(model_path)
		if not os.path.exists(log_path + "/loss"): os.makedirs(log_path + "/loss")

		callbacks = []
		callbacks.append(CSVLogger(log_path + "/loss/" + self.ver + ".csv",
			separator=',', append=True))
		if save_model: callbacks.append(SaveWeights(model_path))

		if resume_epoch > 0: self.model.load_weights(model_path + "/epoch-" +
			str(resume_epoch-1) + "/variables/variables")

		if eval_example:
			print("Saving a mini-batch of training examples in .mat files...")
			inp_batch, tgt_batch, seq_mask_batch = list(train_dataset.take(1).as_numpy_iterator())[0]
			save_mat('./inp_batch.mat', inp_batch, 'inp_batch')
			save_mat('./tgt_batch.mat', tgt_batch, 'tgt_batch')
			save_mat('./seq_mask_batch.mat', seq_mask_batch, 'seq_mask_batch')
			print("Testing if add_noise() works correctly...")
			s, d, s_len, d_len, snr_tgt = self.wav_batch(self.train_s_list[0:mbatch_size],
				self.train_d_list[0:mbatch_size])
			(_, s, d) = self.inp_tgt.add_noise_batch(self.inp_tgt.normalise(s),
				self.inp_tgt.normalise(d), s_len, d_len, snr_tgt)
			for (i, _) in enumerate(s):
				snr_act = self.inp_tgt.snr_db(s[i][0:s_len[i]], d[i][0:d_len[i]])
				print('SNR target|actual: {:.2f}|{:.2f} (dB).'.format(snr_tgt[i], snr_act))

		if "MHA" in self.network_type:
			print("Using Transformer learning rate schedular.")
			lr_schedular = TransformerSchedular(self.network.d_model,
				self.network.warmup_steps)
			opt = Adam(learning_rate=lr_schedular, clipvalue=1.0, beta_1=0.9,
				beta_2=0.98, epsilon=1e-9)
		else: opt = Adam(learning_rate=0.001, clipvalue=1.0)

		if loss_fnc == "BinaryCrossentropy": loss = BinaryCrossentropy()
		elif loss_fnc == "MeanSquaredError": loss = MeanSquaredError()
		else: raise ValueError("Invalid loss function")

		self.model.compile(
			sample_weight_mode="temporal",
			loss=loss,
			optimizer=opt
			)
		print("SNR levels used for training:")
		print(self.snr_levels)
		self.model.fit(
			x=train_dataset,
			initial_epoch=resume_epoch,
			epochs=max_epochs,
			steps_per_epoch=self.n_iter,
			callbacks=callbacks,
			validation_data=val_set,
			validation_steps=val_steps
			)

	def infer(
		self,
		test_x,
		test_x_len,
		test_x_base_names,
		test_epoch,
		model_path='model',
		out_type='y',
		gain='mmse-lsa',
		out_path='out',
		n_filters=40,
		saved_data_path=None,
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
			saved_data_path - path to saved data necessary for enhancement.
		"""
		out_path_base = out_path
		if not isinstance(test_epoch, list): test_epoch = [test_epoch]
		if not isinstance(gain, list): gain = [gain]

		# The mel-scale filter bank is to compute an ideal binary mask (IBM)
		# estimate for log-spectral subband energies (LSSE).
		if out_type == 'subband_ibm_hat':
			mel_filter_bank = self.mel_filter_bank(n_filters)

		for e in test_epoch:
			if e < 1: raise ValueError("test_epoch must be greater than 0.")
			for g in gain:

				out_path = out_path_base + '/' + self.ver + '/' + 'e' + str(e) # output path.
				if out_type == 'xi_hat': out_path = out_path + '/xi_hat'
				elif out_type == 'gamma_hat': out_path = out_path + '/gamma_hat'
				elif out_type == 'mag_hat': out_path = out_path + '/mag_hat'
				elif out_type == 'y':
					if (self.inp_tgt_type == 'MagGain') or (self.inp_tgt_type == 'MagMag'):
						out_path = out_path + '/y'
					else: out_path = out_path + '/y/' + g
				elif out_type == 'deepmmse': out_path = out_path + '/deepmmse'
				elif out_type == 'ibm_hat': out_path = out_path + '/ibm_hat'
				elif out_type == 'subband_ibm_hat': out_path = out_path + '/subband_ibm_hat'
				elif out_type == 'cd_hat': out_path = out_path + '/cd_hat'
				else: raise ValueError('Invalid output type.')
				if not os.path.exists(out_path): os.makedirs(out_path)

				self.model.load_weights(model_path + '/epoch-' + str(e-1) +
					'/variables/variables' )

				print("Processing observations...")
				inp_batch, supplementary_batch, n_frames = self.observation_batch(test_x, test_x_len)

				print("Performing inference...")
				tgt_hat_batch = self.model.predict(inp_batch, batch_size=1, verbose=1)

				print("Saving outputs...")
				batch_size = len(test_x_len)
				for i in tqdm(range(batch_size)):
					base_name = test_x_base_names[i]
					inp = inp_batch[i,:n_frames[i],:]
					tgt_hat = tgt_hat_batch[i,:n_frames[i],:]

					# if tf.is_tensor(supplementary_batch):
					supplementary = supplementary_batch[i,:n_frames[i],:]

					if saved_data_path is not None:
						saved_data = read_mat(saved_data_path + '/' + base_name + '.mat')
						supplementary = (supplementary, saved_data)

					if out_type == 'xi_hat':
						xi_hat = self.inp_tgt.xi_hat(tgt_hat)
						save_mat(out_path + '/' + base_name + '.mat', xi_hat, 'xi_hat')
					elif out_type == 'gamma_hat':
						gamma_hat = self.inp_tgt.gamma_hat(tgt_hat)
						save_mat(out_path + '/' + base_name + '.mat', gamma_hat, 'gamma_hat')
					elif out_type == 'mag_hat':
						mag_hat = self.inp_tgt.mag_hat(tgt_hat)
						save_mat(out_path + '/' + base_name + '.mat', mag_hat, 'mag_hat')
					elif out_type == 'y':
						y = self.inp_tgt.enhanced_speech(inp, supplementary, tgt_hat, g).numpy()
						save_wav(out_path + '/' + base_name + '.wav', y, self.inp_tgt.f_s)
					elif out_type == 'deepmmse':
						xi_hat = self.inp_tgt.xi_hat(tgt_hat)
						d_PSD_hat = np.multiply(np.square(inp), gfunc(xi_hat, xi_hat+1.0,
							gtype='deepmmse'))
						save_mat(out_path + '/' + base_name + '.mat', d_PSD_hat, 'd_psd_hat')
					elif out_type == 'ibm_hat':
						xi_hat = self.inp_tgt.xi_hat(tgt_hat)
						ibm_hat = np.greater(xi_hat, 1.0).astype(bool)
						save_mat(out_path + '/' + base_name + '.mat', ibm_hat, 'ibm_hat')
					elif out_type == 'subband_ibm_hat':
						xi_hat = self.inp_tgt.xi_hat(tgt_hat)
						xi_hat_subband = np.matmul(xi_hat, mel_filter_bank.transpose())
						subband_ibm_hat = np.greater(xi_hat_subband, 1.0).astype(bool)
						save_mat(out_path + '/' + base_name + '.mat', subband_ibm_hat,
							'subband_ibm_hat')
					elif out_type == 'cd_hat':
						cd_hat = self.inp_tgt.cd_hat(tgt_hat)
						save_mat(out_path + '/' + base_name + '.mat', cd_hat, 'cd_hat')
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
		log_path,
		model_path='model',
		gain='mmse-lsa',
		):
		"""
		Deep Xi testing. Objective measures are used to evaluate the performance
		of Deep Xi. Note that the 'supplementary' variable can includes other
		variables necessary for synthesis, like the noisy-speech short-time
		phase spectrum.

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
		"""
		from pesq import pesq
		from pystoi import stoi
		print("Processing observations...")
		inp_batch, supplementary_batch, n_frames = self.observation_batch(test_x, test_x_len)
		if not isinstance(test_epoch, list): test_epoch = [test_epoch]
		if not isinstance(gain, list): gain = [gain]
		for e in test_epoch:
			for g in gain:

				if e < 1: raise ValueError("test_epoch must be greater than 0.")

				self.model.load_weights(model_path + '/epoch-' + str(e-1) +
					'/variables/variables' )

				print("Performing inference...")
				tgt_hat_batch = self.model.predict(inp_batch, batch_size=1, verbose=1)

				print("Performing synthesis and objective scoring...")
				results = {}
				batch_size = len(test_x_len)
				for i in tqdm(range(batch_size)):
					base_name = test_x_base_names[i]
					inp = inp_batch[i,:n_frames[i],:]
					supplementary = supplementary_batch[i,:n_frames[i],:]
					tgt_hat = tgt_hat_batch[i,:n_frames[i],:]

					y = self.inp_tgt.enhanced_speech(inp, supplementary, tgt_hat, g).numpy()

					for (j, basename) in enumerate(test_s_base_names):
						if basename in test_x_base_names[i]: ref_idx = j

					s = self.inp_tgt.normalise(test_s[ref_idx,
						0:test_s_len[ref_idx]]).numpy() # from int16 to float.
					y = y[0:len(s)]

					try: noise_src = test_x_base_names[i].split("_")[-2]
					except IndexError: noise_src = "Null"
					if noise_src == "Null": snr_level = 0
					else: snr_level = int(test_x_base_names[i].split("_")[-1][:-2])

					results = self.add_score(results, (noise_src, snr_level, 'STOI'),
						100*stoi(s, y, self.inp_tgt.f_s, extended=False))
					results = self.add_score(results, (noise_src, snr_level, 'eSTOI'),
						100*stoi(s, y, self.inp_tgt.f_s, extended=True))
					results = self.add_score(results, (noise_src, snr_level, 'PESQ'),
						pesq(self.inp_tgt.f_s, s, y, 'nb'))
					results = self.add_score(results, (noise_src, snr_level, 'MOS-LQO'),
						pesq(self.inp_tgt.f_s, s, y, 'wb'))

				noise_srcs, snr_levels, metrics = set(), set(), set()
				for key, value in results.items():
					noise_srcs.add(key[0])
					snr_levels.add(key[1])
					metrics.add(key[2])

				if not os.path.exists(log_path + "/results"): os.makedirs(log_path + "/results")

				with open(log_path + "/results/" + self.ver + "_e" + str(e) + '_' + g + ".csv", "w") as f:
					f.write("noise,snr_db")
					for k in sorted(metrics): f.write(',' + k)
					f.write('\n')
					for i in sorted(noise_srcs):
						for j in sorted(snr_levels):
							f.write("{},{}".format(i, j))
							for k in sorted(metrics):
								if (i, j, k) in results.keys():
									f.write(",{:.2f}".format(np.mean(results[(i,j,k)])))
							f.write('\n')

				avg_results = {}
				for i in sorted(noise_srcs):
					for j in sorted(snr_levels):
						if (j >= self.min_snr) and (j <= self.max_snr):
							for k in sorted(metrics):
								if (i, j, k) in results.keys():
									avg_results = self.add_score(avg_results, k, results[(i,j,k)])

				if not os.path.exists(log_path + "/results/average.csv"):
					with open(log_path + "/results/average.csv", "w") as f:
						f.write("ver")
						for i in sorted(metrics): f.write("," + i)
						f.write('\n')

				with open(log_path + "/results/average.csv", "a") as f:
					f.write(self.ver + "_e" + str(e) + '_' + g)
					for i in sorted(metrics):
						if i in avg_results.keys():
							f.write(",{:.2f}".format(np.mean(avg_results[i])))
					f.write('\n')

	def sample(
		self,
		sample_size,
		sample_dir='data',
		):
		"""
		Gathers a sample of the training set. The sample can be used to compute
		statistics for mapping functions.

		Argument/s:
			sample_size - number of training examples included in the sample.
			sample_dir - path to the saved sample.
		"""
		sample_path = sample_dir + '/sample'
		if os.path.exists(sample_path + '.npz'):
			print('Loading sample...')
			with np.load(sample_path + '.npz') as sample:
				s_sample = sample['s_sample']
				d_sample = sample['d_sample']
				x_sample = sample['x_sample']
				wav_len = sample['wav_len']
		elif self.train_s_list == None:
			raise ValueError('No sample.npz file exists.')
		else:
			if sample_size == None: raise ValueError("sample_size is not set.")
			print('Gathering a sample of the training set...')
			s_sample_list = random.sample(self.train_s_list, sample_size)
			d_sample_list = random.sample(self.train_d_list, sample_size)
			s_sample_int, d_sample_int, s_sample_len, d_sample_len, snr_sample = self.wav_batch(s_sample_list,
				d_sample_list)
			s_sample = np.zeros_like(s_sample_int, np.float32)
			d_sample = np.zeros_like(s_sample_int, np.float32)
			x_sample = np.zeros_like(s_sample_int, np.float32)
			for i in tqdm(range(s_sample.shape[0])):
				s, d, x, _ = self.inp_tgt.mix(s_sample_int[i:i+1], d_sample_int[i:i+1],
					s_sample_len[i:i+1], d_sample_len[i:i+1], snr_sample[i:i+1])
				s_sample[i, 0:s_sample_len[i]] = s
				d_sample[i, 0:s_sample_len[i]] = d
				x_sample[i, 0:s_sample_len[i]] = x
			wav_len = s_sample_len
			if not os.path.exists(sample_dir): os.makedirs(sample_dir)
			np.savez(sample_path + '.npz', s_sample=s_sample,
				d_sample=d_sample, x_sample=x_sample, wav_len=wav_len)
			sample = {'s_sample': s_sample, 'd_sample': d_sample,
				'x_sample': x_sample, 'wav_len': wav_len}
			save_mat(sample_path + '.mat', sample, 'stats')
			print('Sample of the training set saved.')
		return s_sample, d_sample, x_sample, wav_len

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
			(tf.TensorShape([None, None, self.inp_tgt.n_feat]),
				tf.TensorShape([None, None, self.inp_tgt.n_outp]),
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
			inp_mbatch - mini-batch of observations (input to network).
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
				inp_mbatch, xi_bar_mbatch, n_frames_mbatch = \
					self.inp_tgt.example(s_mbatch, d_mbatch, s_mbatch_len,
					d_mbatch_len, snr_mbatch)
				seq_mask_mbatch = tf.cast(tf.sequence_mask(n_frames_mbatch), tf.float32)
				start_idx += self.mbatch_size; end_idx += self.mbatch_size
				if end_idx > self.n_examples: end_idx = self.n_examples
				yield inp_mbatch, xi_bar_mbatch, seq_mask_mbatch

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
			inp_batch - batch of observations (input to network).
			tgt_batch - batch of targets (mapped a priori SNR).
			seq_mask_batch - batch of sequence masks.
		"""
		print('Processing validation batch...')
		batch_size = len(val_s)
		max_n_frames = self.inp_tgt.n_frames(max(val_s_len))
		inp_batch = np.zeros([batch_size, max_n_frames, self.inp_tgt.n_feat], np.float32)
		tgt_batch = np.zeros([batch_size, max_n_frames, self.inp_tgt.n_outp], np.float32)
		seq_mask_batch = np.zeros([batch_size, max_n_frames], np.float32)
		for i in tqdm(range(batch_size)):
			inp, tgt, _ = self.inp_tgt.example(val_s[i:i+1], val_d[i:i+1],
				val_s_len[i:i+1], val_d_len[i:i+1], val_snr[i:i+1])
			n_frames = self.inp_tgt.n_frames(val_s_len[i])
			inp_batch[i,:n_frames,:] = inp.numpy()
			if tf.is_tensor(tgt): tgt_batch[i,:n_frames,:] = tgt.numpy()
			else: tgt_batch[i,:n_frames,:] = tgt
			seq_mask_batch[i,:n_frames] = tf.cast(tf.sequence_mask(n_frames), tf.float32)
		return inp_batch, tgt_batch, seq_mask_batch

	def observation_batch(self, x_batch, x_batch_len):
		"""
		Computes observations (inp) from noisy speech recordings.

		Argument/s:
			x_batch - noisy-speech batch.
			x_batch_len - noisy-speech batch lengths.

		Returns:
			inp_batch - batch of observations (input to network).
			supplementary_batch - batch of noisy-speech short-time phase spectrums.
			n_frames_batch - number of frames in each observation.
		"""
		batch_size = len(x_batch)
		max_n_frames = self.inp_tgt.n_frames(max(x_batch_len))
		inp_batch = np.zeros([batch_size, max_n_frames, self.inp_tgt.n_feat], np.float32)
		supplementary_batch = np.zeros([batch_size, max_n_frames, self.inp_tgt.n_feat], np.float32)
		n_frames_batch = [self.inp_tgt.n_frames(i) for i in x_batch_len]
		for i in tqdm(range(batch_size)):
			inp, supplementary = self.inp_tgt.observation(x_batch[i,:x_batch_len[i]])
			inp_batch[i,:n_frames_batch[i],:] = inp
			supplementary_batch[i,:n_frames_batch[i],:] = supplementary
		return inp_batch, supplementary_batch, n_frames_batch

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
		# snr_batch = np.random.randint(self.min_snr, self.max_snr+1, batch_size)
		snr_batch = np.array(random.choices(self.snr_levels, k=batch_size))
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

class SaveWeights(Callback):
	def __init__(self, model_path):
		super(SaveWeights, self).__init__()
		self.model_path = model_path

	def on_epoch_end(self, epoch, logs=None):
		self.model.save(self.model_path + "/epoch-" + str(epoch))

class TransformerSchedular(LearningRateSchedule):
	def __init__(self, d_model, warmup_steps):
		super(TransformerSchedular, self).__init__()
		self.d_model = float(d_model)
		self.warmup_steps = warmup_steps

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps ** -1.5)
		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

	def get_config(self):
		config = {'d_model': self.d_model, 'warmup_steps': self.warmup_steps}
		return config
