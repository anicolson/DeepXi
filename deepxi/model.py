## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

# from dev.acoustic.analysis_synthesis.polar import synthesis
# from dev.acoustic.feat import polar
# from dev.ResNet import ResNet
# import dev.optimisation as optimisation
# import numpy as np
# import tensorflow as tf

from deepxi.gain import gfunc
from deepxi.network.cnn import TCN
from deepxi.sig import DeepXiInput
from deepxi.utils import read_wav, save_wav, save_mat
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import deepxi.se_batch as batch
import csv, math, os, random
import numpy as np
import tensorflow as tf

class DeepXi(DeepXiInput):
	"""
	Deep Xi
	"""
	def __init__(
		self, 
		N_w, 
		N_s, 
		NFFT, 
		f_s, 
		mu=None, 
		sigma=None, 
		network='TCN', 
		min_snr=None, 
		max_snr=None, 
		):
		"""
		Argument/s
			Nw - window length (samples).
			Ns - window shift (samples).
			NFFT - number of DFT bins.
			f_s - sampling frequency.
			mu - sample mean of each instantaneous a priori SNR in dB frequency component.
			sigma - sample standard deviation of each instantaneous a priori SNR in dB frequency component.
			network - network type.
			min_snr - minimum SNR level for training.
			max_snr - maximum SNR level for training.
		"""
		super().__init__(N_w, N_s, NFFT, f_s, mu, sigma)
		self.min_snr = min_snr
		self.max_snr = max_snr
		self.n_feat = math.ceil(self.NFFT/2 + 1)
		self.n_outp = self.n_feat
		self.inp = Input(name='inp', shape=[None, self.n_feat], dtype='float32')

		self.mask = tf.keras.layers.Masking(mask_value=0.0)(self.inp)

		if network == 'TCN': self.network = TCN(self.mask, self.n_outp, B=40, d_model=256, d_f=64, k=3, max_d_rate=16)
		else: raise ValueError('Invalid network type.')

		self.opt = Adam()
		self.model = Model(inputs=self.inp, outputs=self.network.outp)
		self.model.summary()

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
		ver='VERSION_NAME',
		stats_path=None, 
		sample_size=None,
		):
		"""

		Argument/s:

		Output/s:
		"""
		self.train_s_list = train_s_list
		self.train_d_list = train_d_list
		self.mbatch_size = mbatch_size
		self.n_examples = len(self.train_s_list)
		self.n_iter = math.ceil(self.n_examples/mbatch_size)

		self.sample_stats(stats_path, sample_size, train_s_list, train_d_list)
		train_dataset = self.dataset(max_epochs-resume_epoch)
		if val_flag: val_set = self.val_batch(val_save_path, val_s, val_d, val_s_len, val_d_len, val_snr)
		else: val_set = None
		
		if not os.path.exists(model_path): os.makedirs(model_path)
		if not os.path.exists("log"): os.makedirs("log")
		if not os.path.exists("log/" + ver + ".csv"):
			with open("log/" + ver + ".csv", "w") as f: 
				f.write("Epoch, Train loss, Val. loss\n")

		callbacks = []
		callbacks.append(CSVLogger("log/" + ver + ".csv", separator=',', append=True))
		callbacks.append(SaveWeights(model_path))

		if resume_epoch > 0: self.model.load_weights(model_path + '/epoch-' + str(resume_epoch-1) + 
			'/variables/variables' )

		self.model.compile(loss='binary_crossentropy', optimizer=self.opt)
		self.model.fit(
			train_dataset, 
			initial_epoch=resume_epoch, 
			epochs=max_epochs, 
			steps_per_epoch=self.n_iter,
			validation_data=val_set, 
			validation_steps=len(val_set[0]),
			callbacks=callbacks
			)

	def infer(
		self,
		test_x,
		test_x_len,
		test_x_base_names,
		epoch,
		model_path='model',
		out_type='y',
		gain='mmse-lsa',
		out_path='out',
		stats_path=None 
		): 
		"""

		Argument/s:

		Output/s:
		"""
		if out_type == 'xi_hat': out_path = out_path + '/xi_hat'
		elif out_type == 'y': out_path = out_path + '/' + gain + '/y'
		elif out_type == 'ibm_hat': out_path = out_path + '/ibm_hat'
		else: raise ValueError('Invalid output type.')
		if not os.path.exists(out_path): os.makedirs(out_path)
		
		self.sample_stats(stats_path)
		self.model.load_weights(model_path + '/epoch-' + str(epoch-1) + 
			'/variables/variables' )

		x_STMS_batch, x_STPS_batch, n_frames = self.observation_batch(test_x, test_x_len)
		xi_bar_hat_batch = self.model.predict(x_STMS_batch, batch_size=1) 

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
				ibm_hat = np.greater(xi_hat, 1.0)
				save_mat(out_path + '/' + base_name + '.mat', ibm_hat, 'ibm_hat')
			else: raise ValueError('Invalid output type.')

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

		Output/s:
		"""
		if os.path.exists(stats_path + '/stats.npz'):
			print('Loading sample statistics...')
			with np.load(stats_path + '/stats.npz') as stats:
				self.mu = stats['mu_hat']
				self.sigma = stats['sigma_hat']
		elif train_s_list == None:
			raise ValueError('No stats.p file exists. data/stats.p is available here: https://github.com/anicolson/DeepXi/blob/master/data/stats.npz.')
		else:
			print('Finding sample statistics...')
			s_sample_list = random.sample(self.train_s_list, sample_size)
			d_sample_list = random.sample(self.train_d_list, sample_size)
			s_sample, d_sample, s_sample_len, d_sample_len, snr_sample = self.wav_batch(s_sample_list, d_sample_list)
			snr_sample = np.random.randint(self.min_snr, self.max_snr + 1, sample_size)
			samples = []
			for i in tqdm(range(s_sample.shape[0])):
				xi, _ = self.instantaneous_a_priori_snr_db(s_sample[i:i+1], d_sample[i:i+1], s_sample_len[i:i+1], 
					d_sample_len[i:i+1], snr_sample[i:i+1])
				samples.append(np.squeeze(xi.numpy()))
			samples = np.vstack(samples)
			if len(samples.shape) != 2: raise ValueError('Incorrect shape for sample.')
			stats = {'mu_hat': np.mean(samples, axis=0), 'sigma_hat': np.std(samples, axis=0)}
			self.mu, self.sigma = stats['mu_hat'], stats['sigma_hat']
			if not os.path.exists(stats_path): os.makedirs(stats_path)
			np.savez(stats_path + '/stats.npz', mu_hat=stats['mu_hat'], sigma_hat=stats['sigma_hat'])
			save_mat(stats_path + '/stats.m', stats, 'stats')
			print('Sample statistics saved.')

	def dataset(self, n_epochs, buffer_size=16):
		"""

		Argument/s:

		Output/s:
		"""
		dataset = tf.data.Dataset.from_generator(
			self.mbatch_gen, 
			(tf.float32, tf.float32), 
			(tf.TensorShape([None, None, self.n_feat]), tf.TensorShape([None, None, self.n_outp])),
			[tf.constant(n_epochs)]
			)
		dataset = dataset.prefetch(buffer_size) 
		return dataset

	def val_batch(self, save_path='data', val_s=None, val_d=None, val_s_len=None, val_d_len=None, val_snr=None):
		"""

		Argument/s:

		Output/s:
		"""
		if not os.path.exists(save_path): os.makedirs(save_path)
		if os.path.exists(save_path + '/val_batch.npz'):
			print('Loading validation batch...')
			with np.load(save_path + '/val_batch.npz') as data:
				val_inp = data['val_inp']
				val_tgt = data['val_tgt']
		else:
			print('Creating validation batch...')
			val_inp, val_tgt = self.example_batch(val_s, val_d, val_s_len, val_d_len, val_snr)
			np.savez(save_path + '/val_batch.npz', val_inp=val_inp, val_tgt=val_tgt)
		return val_inp, val_tgt

	def mbatch_gen(self, n_epochs): 
		"""
		Used to create tf.data.Dataset for training.

		Argument/s:

		Output/s:
		"""
		for _ in range(n_epochs):
			random.shuffle(self.train_s_list)
			start_idx, end_idx = 0, self.mbatch_size
			for _ in range(self.n_iter):
				s_mbatch_list = self.train_s_list[start_idx:end_idx]
				d_mbatch_list = random.sample(self.train_d_list, end_idx-start_idx)
				s_mbatch, d_mbatch, s_mbatch_len, d_mbatch_len, snr_mbatch = self.wav_batch(s_mbatch_list, d_mbatch_list)
				x_STMS, xi_bar, _ = self.training_example(s_mbatch, d_mbatch, s_mbatch_len, d_mbatch_len, snr_mbatch)
				start_idx += self.mbatch_size; end_idx += self.mbatch_size
				if end_idx > self.n_examples: end_idx = self.n_examples
				yield x_STMS, xi_bar

	def example_batch(self, s_batch, d_batch, s_batch_len, d_batch_len, snr_batch): 
		"""


		Argument/s:

		Output/s:
		"""
		batch_size = len(s_batch)
		max_n_frames = self.n_frames(max(s_batch_len))
		inp_batch = np.zeros([batch_size, max_n_frames, self.n_feat], np.float32)
		tgt_batch = np.zeros([batch_size, max_n_frames, self.n_feat], np.float32)
		for i in tqdm(range(batch_size)):
			x_STMS, xi_bar, _ = self.training_example(s_batch[i:i+1], d_batch[i:i+1], 
				s_batch_len[i:i+1], d_batch_len[i:i+1], snr_batch[i:i+1])
			n_frames = self.n_frames(s_batch_len[i])
			inp_batch[i,:n_frames,:] = x_STMS.numpy()
			tgt_batch[i,:n_frames,:] = xi_bar.numpy()
		return inp_batch, tgt_batch

	def observation_batch(self, x_batch, x_batch_len): 
		"""
		Computes observations (noisy-speech STMS) from noisy speech recordings.

		Argument/s:

		Output/s:
		"""
		batch_size = len(x_batch)
		max_n_frames = self.n_frames(max(x_batch_len))
		x_STMS_batch = np.zeros([batch_size, max_n_frames, self.n_feat], np.float32)
		x_STPS_batch = np.zeros([batch_size, max_n_frames, self.n_feat], np.float32)
		for i in tqdm(range(batch_size)):
			x_STMS, x_STPS = self.observation(x_batch[i,:x_batch_len[i]])
			n_frames = self.n_frames(x_batch_len[i])
			x_STMS_batch[i,:n_frames,:] = x_STMS.numpy()
			x_STPS_batch[i,:n_frames,:] = x_STPS.numpy()
		n_frames = [self.n_frames(i) for i in x_batch_len]
		return x_STMS_batch, x_STPS_batch, n_frames

	def wav_batch(self, s_batch_list, d_batch_list):
		"""
		Loads .wav files into batches.

		Argument/s:

		Output/s:
		"""
		batch_size = len(s_batch_list)
		max_len = max([dic['seq_len'] for dic in s_batch_list]) 
		s_batch = np.zeros([batch_size, max_len], np.int16)
		d_batch = np.zeros([batch_size, max_len], np.int16)
		s_batch_len = np.zeros(batch_size, np.int32) 
		for i in range(batch_size):
			(wav, _) = read_wav(s_batch_list[i]['file_path'])		
			s_batch[i,:s_batch_list[i]['seq_len']] = wav
			s_batch_len[i] = s_batch_list[i]['seq_len'] 
			flag = True
			while flag:
				if d_batch_list[i]['seq_len'] < s_batch_len[i]: d_batch_list[i] = random.choice(self.train_d_list)
				else: flag = False
			(wav, _) = read_wav(d_batch_list[i]['file_path']) 
			rand_idx = np.random.randint(0, 1+d_batch_list[i]['seq_len']-s_batch_len[i])
			d_batch[i,:s_batch_len[i]] = wav[rand_idx:rand_idx+s_batch_len[i]]
		d_batch_len = s_batch_len
		snr_batch = np.random.randint(self.min_snr, self.max_snr+1, batch_size) 
		return s_batch, d_batch, s_batch_len, d_batch_len, snr_batch

class SaveWeights(Callback):
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

