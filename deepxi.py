## FILE:           deepxi.py
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
## BRIEF:          'DeepXi' training and testing.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import tensorflow as tf
from tensorflow.python.data import Dataset, Iterator
import numpy as np
from datetime import datetime
from scipy.io.wavfile import read
import scipy.io.wavfile
import scipy.io as spio
import scipy.special as spsp
import os, random, math, time, sys, pickle
sys.path.insert(0, 'lib')
import feat, batch, residual, utils
np.set_printoptions(threshold=1e6)

## VARIABLE DESCRIPTIONS
# s - clean speech.
# d - noise.
# x - noisy speech.

## DEEP XI ARGUMENTS
def deepxi_args(args):

	## DEPENDANT OPTIONS
	args.model_path = args.model_path + '/' + args.ver # model save path.
	args.train_s_path = args.set_path + '/train_clean_speech' # path to the clean speech training set.
	args.train_d_path = args.set_path + '/train_noise' # path to the noise training set.
	args.val_s_path = args.set_path + '/val_clean_speech' # path to the clean speech validation set.
	args.val_d_path = args.set_path + '/val_noise' # path to the noise validation set.
	args.out_path = args.out_path + '/' + args.ver + '/' + 'e' + str(args.epoch) # output path.
	args.Nw = int(args.fs*args.Tw*0.001) # window length (samples).
	args.Ns = int(args.fs*args.Ts*0.001) # window shift (samples).
	args.NFFT = int(pow(2, np.ceil(np.log2(args.Nw)))) # number of DFT components.

	## A PRIORI SNR IN DB STATS
	mu_mat = spio.loadmat(args.stats_path + '/mu.mat') # mean of a priori SNR in dB from MATLAB.
	args.mu = tf.constant(mu_mat['mu'], dtype=tf.float32) 
	sigma_mat = spio.loadmat(args.stats_path + '/sigma.mat') # standard deviation of a priori SNR in dB from MATLAB.
	args.sigma = tf.constant(sigma_mat['sigma'], dtype=tf.float32) 

	## DATASETS
	if args.train: ## TRAINING AND VALIDATION CLEAN SPEECH AND NOISE SET
		args.train_s_list = batch.Train_list(args.train_s_path, '*.wav', 'clean') # clean speech training list.
		args.train_d_list = batch.Train_list(args.train_d_path, '*.wav', 'noise') # noise training list.
		if not os.path.exists(args.model_path): os.makedirs(args.model_path) # make model path directory.
		args.val_s, args.val_s_len, args.val_snr, _ = batch.Batch(args.val_s_path, '*.wav', 
			list(range(args.min_snr, args.max_snr + 1))) # clean validation waveforms and lengths.
		args.val_d, args.val_d_len, _, _ = batch.Batch(args.val_d_path, '*.wav', 
			list(range(args.min_snr, args.max_snr + 1))) # noise validation waveforms and lengths.

	## INFERENCE
	if args.infer: args.test_x, args.test_x_len, args.test_snr, args.test_fnames = batch.Batch(args.test_x_path, '*.wav', []) # noisy speech test waveforms and lengths.
	return args

## DEEP XI ARTIFICIAL NEURAL NETWORK 
class deepxi_net:
	def __init__(self, args):

		## PLACEHOLDERS
		self.s_ph = tf.placeholder(tf.int16, shape=[None, None], name='s_ph') # clean speech placeholder.
		self.d_ph = tf.placeholder(tf.int16, shape=[None, None], name='d_ph') # noise placeholder.
		self.x_ph = tf.placeholder(tf.int16, shape=[None, None], name='x_ph') # noisy speech placeholder.
		self.s_len_ph = tf.placeholder(tf.int32, shape=[None], name='s_len_ph') # clean speech sequence length placeholder.
		self.d_len_ph = tf.placeholder(tf.int32, shape=[None], name='d_len_ph') # noise sequence length placeholder.
		self.x_len_ph = tf.placeholder(tf.int32, shape=[None], name='x_len_ph') # noisy speech sequence length placeholder.
		self.snr_ph = tf.placeholder(tf.float32, shape=[None], name='snr_ph') # SNR placeholder.
		self.x_MS_ph = tf.placeholder(tf.float32, shape=[None, None, args.input_dim], name='x_MS_ph') # noisy speech MS placeholder.
		self.x_MS_len_ph = tf.placeholder(tf.int32, shape=[None], name='x_MS_len_ph') # noisy speech MS sequence length placeholder.
		self.target_ph = tf.placeholder(tf.float32, shape=[None, args.input_dim], name='target_phh') # training target placeholder.
		self.keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph') # keep probability placeholder.
		self.training_ph = tf.placeholder(tf.bool, name='training_ph') # training placeholder.

		## FEATURE GRAPH
		print('Preparing graph...')
		self.P = tf.reduce_max(self.s_len_ph) # padded waveform length.
		self.feature = feat.xi_mapped(self.s_ph, self.d_ph, self.s_len_ph, self.d_len_ph, self.snr_ph, args.Nw, args.Ns, args.NFFT, 
			args.fs, self.P, args.nconst, args.mu, args.sigma) # feature graph.

		## RESNET
		self.output = residual.Residual(self.x_MS_ph, self.x_MS_len_ph, self.keep_prob_ph, 
			self.training_ph, args.num_outputs, args)

		## LOSS & OPTIMIZER
		self.loss = residual.loss(self.target_ph, self.output, 'sigmoid_cross_entropy')
		self.total_loss = tf.reduce_mean(self.loss, axis=0)
		self.trainer, _ = residual.optimizer(self.total_loss, optimizer='adam', grad_clip=True)

		## SAVE VARIABLES
		self.saver = tf.train.Saver(max_to_keep=256)

		## NUMBER OF PARAMETERS
		if args.verbose: print("No. of trainable parameters: %g." % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

		## INFERENCE GRAPH
		if args.infer:

			## PLACEHOLDERS
			self.output_ph = tf.placeholder(tf.float32, shape=[None, args.input_dim], name='output_ph') # network output placeholder.
			self.x_MS_2D_ph = tf.placeholder(tf.float32, shape=[None, args.input_dim], name='x_MS_2D_ph') # noisy speech MS placeholder (in 2D form).
			self.x_PS_ph = tf.placeholder(tf.float32, shape=[None, args.input_dim], name='x_PS_ph') # noisy speech PS placeholder.
			self.xi_hat_ph = tf.placeholder(tf.float32, shape=[None, args.input_dim], name='xi_hat_ph') # a priori SNR estimate placeholder.
			self.G_ph = tf.placeholder(tf.float32, shape=[None, args.input_dim], name='G_ph') # gain function placeholder.
			
			## ANALYSIS
			self.x = tf.truediv(tf.cast(tf.slice(tf.squeeze(self.x_ph), [0], [tf.squeeze(self.x_len_ph)]), tf.float32), args.nconst) # remove padding and normalise.
			self.x_DFT = feat.stft(self.x, args.Nw, args.Ns, args.NFFT) # noisy speech single-sided short-time Fourier transform.
			self.x_MS_3D = tf.expand_dims(tf.abs(self.x_DFT), 0) # noisy speech single-sided magnitude spectrum (in 3D form).
			self.x_MS = tf.abs(self.x_DFT) # noisy speech single-sided magnitude spectrum.
			self.x_PS = tf.angle(self.x_DFT) # noisy speech single-sided phase spectrum.
			self.x_seq_len = feat.nframes(self.x_len_ph, args.Ns) # length of each sequence.

			## MODIFICATION (SPEECH ENHANCEMENT)
			if args.gain == 'ibm': self.G = tf.cast(tf.greater(self.xi_hat_ph, 1), tf.float32) # IBM gain function.
			if args.gain == 'wf': self.G = tf.truediv(self.xi_hat_ph, tf.add(self.xi_hat_ph, 1.0)) # WF gain function.
			if args.gain == 'srwf': self.G = tf.sqrt(tf.truediv(self.xi_hat_ph, tf.add(self.xi_hat_ph, 1.0))) # SRWF gain function.
			if args.gain == 'irm': self.G = tf.sqrt(tf.truediv(self.xi_hat_ph, tf.add(self.xi_hat_ph, 1.0))) # IRM gain function.
			if args.gain == 'cwf':
				self.G = tf.sqrt(self.xi_hat_ph)
				self.G = tf.truediv(self.G, tf.add(self.G, 1.0)) # cWF gain function.
			self.s_hat_MS = tf.multiply(self.x_MS_2D_ph, self.G_ph) # enhanced speech single-sided magnitude spectrum.

			## SYNTHESIS GRAPH
			self.y_DFT = tf.cast(self.s_hat_MS, tf.complex64) * tf.exp(1j * tf.cast(self.x_PS_ph, tf.complex64)) # enhanced speech single-sided short-time Fourier transform.
			self.y = tf.contrib.signal.inverse_stft(self.y_DFT, args.Nw, args.Ns, args.NFFT, 
				tf.contrib.signal.inverse_stft_window_fn(args.Ns, 
				forward_window_fn=tf.contrib.signal.hamming_window)) # synthesis.

## TRAINING
def train(sess, net, args):
	print("Training...")

	## CONTINUE FROM LAST EPOCH
	if args.cont:
		epoch_par = {'epoch_size': len(args.train_s_list), 'epoch_comp': args.epoch, 
			'start_idx': 0, 'end_idx': args.mbatch_size, 'val_error': float("inf")} # create epoch parameters.
		with open('data/epoch_par_' + args.ver + '.p', 'wb') as f: pickle.dump(epoch_par, f) # save epoch parameters.
		net.saver.restore(sess, args.model_path + '/epoch-' + str(args.epoch)) # load model from last epoch.

	## TRAIN RAW NETWORK
	else:
		if os.path.isfile('data/epoch_par_' + args.ver + '.p'): os.remove('data/epoch_par_' + args.ver + '.p') # remove epoch parameters.
		print('Creating epoch parameters, as no pickle file exists...')
		epoch_par = {'epoch_size': len(args.train_s_list), 'epoch_comp': 0, 
			'start_idx': 0, 'end_idx': args.mbatch_size, 'val_error': float("inf")} # create epoch parameters.
		if args.mbatch_size > epoch_par['epoch_size']: raise ValueError('Error: mini-batch size is greater than the epoch size.')
		with open('data/epoch_par_' + args.ver + '.p', 'wb') as f:
			pickle.dump(epoch_par, f) # save epoch parameters.
		sess.run(tf.global_variables_initializer()) # initialise model variables.
		net.saver.save(sess, args.model_path + '/epoch', global_step=epoch_par['epoch_comp']) # save model.

	## TRAINING LOG
	if not os.path.exists('log'): os.makedirs('log') # create log directory.
	with open("log/val_error_" + args.ver + ".txt", "a") as results:
		results.write("val err, train err, epoch count, D/T:\n")

	train_err = 0; mbatch_count = 0
	while args.train:
		## MINI-BATCH GENERATION
		mbatch_size_iter = epoch_par['end_idx'] - epoch_par['start_idx'] # number of examples in mini-batch for the training iteration.
		train_s_mbatch, train_s_mbatch_seq_len = batch.Clean_mbatch(args.train_s_list, 
			mbatch_size_iter, epoch_par['start_idx'], epoch_par['end_idx']) # generate mini-batch of clean training waveforms.
		train_d_mbatch, train_d_mbatch_seq_len = batch.Noise_mbatch(args.train_d_list, 
			mbatch_size_iter, train_s_mbatch_seq_len) # generate mini-batch of noise training waveforms.
		train_snr_mbatch = np.random.randint(args.min_snr, args.max_snr + 1, epoch_par['end_idx'] - epoch_par['start_idx']) # generate mini-batch of SNR levels.

		## TRAINING ITERATION
		train_mbatch = sess.run(net.feature, feed_dict={net.s_ph: train_s_mbatch, net.d_ph: train_d_mbatch, 
			net.s_len_ph: train_s_mbatch_seq_len, net.d_len_ph: train_d_mbatch_seq_len, net.snr_ph: train_snr_mbatch}) # mini-batch.
		[_, train_mbatch_err] = sess.run([net.trainer, net.total_loss], feed_dict={net.x_MS_ph: train_mbatch[0], net.target_ph: train_mbatch[1], 
			net.x_MS_len_ph: train_mbatch[2], net.training_ph: True}) # training iteration.
		train_err += train_mbatch_err; mbatch_count += 1
		print("E%d: %3.1f%% (train err %3.2f), E%d val err: %3.2f, %s, GPU:%s.       " % 
			(epoch_par['epoch_comp'] + 1, 100*(epoch_par['end_idx']/epoch_par['epoch_size']), train_err/mbatch_count,
			epoch_par['epoch_comp'], epoch_par['val_error'], args.ver, args.gpu), end="\r")

		## UPDATE EPOCH PARAMETERS
		epoch_par['start_idx'] += args.mbatch_size; epoch_par['end_idx'] += args.mbatch_size # start and end index of mini-batch.

		## VALIDATION SET ERROR
		if epoch_par['start_idx'] >= epoch_par['epoch_size']:
			epoch_par['start_idx'] = 0; epoch_par['end_idx'] = args.mbatch_size # reset start and end index of mini-batch.
			random.shuffle(args.train_s_list) # shuffle list.
			start_idx = 0; end_idx = args.mbatch_size; val_flag = True; frames = 0; epoch_par['val_error'] = 0; # validation variables.
			while val_flag:
				val_mbatch = sess.run(net.feature, feed_dict={net.s_ph: args.val_s[start_idx:end_idx], net.d_ph: args.val_d[start_idx:end_idx], 
					net.s_len_ph: args.val_s_len[start_idx:end_idx], net.d_len_ph: args.val_d_len[start_idx:end_idx], net.snr_ph: args.val_snr[start_idx:end_idx]}) # mini-batch.
				val_error_mbatch = sess.run(net.loss, feed_dict={net.x_MS_ph: val_mbatch[0], net.target_ph: val_mbatch[1], 
					net.x_MS_len_ph: val_mbatch[2], net.training_ph: False}) # validation error for each frame in mini-batch.
				frames += val_error_mbatch.shape[0] # total number of frames.
				epoch_par['val_error'] += np.sum(val_error_mbatch)
				print("Validation error for Epoch %d: %3.2f%% complete.       " % 
					(epoch_par['epoch_comp'] + 1, 100*(end_idx/args.val_s_len.shape[0])), end="\r")
				start_idx += args.mbatch_size; end_idx += args.mbatch_size
				if start_idx >= args.val_s_len.shape[0]: val_flag = False
				elif end_idx > args.val_s_len.shape[0]: end_idx = args.val_s_len.shape[0]
			epoch_par['val_error'] /= frames # validation error.
			epoch_par['epoch_comp'] += 1 # an epoch has been completed.
			net.saver.save(sess, args.model_path + '/epoch', global_step=epoch_par['epoch_comp']) # save model.
			with open("log/val_error_" + args.ver + ".txt", "a") as results:
				results.write("%g, %g, %d, %s.\n" % (epoch_par['val_error'], train_err/mbatch_count,
				epoch_par['epoch_comp'], datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
			train_err = 0; mbatch_count = 0
			epoch_par['val_flag'] = False # reset validation flag.
			with open('data/epoch_par_' + args.ver + '.p', 'wb') as f:
				pickle.dump(epoch_par, f)
			if epoch_par['epoch_comp'] >= args.max_epochs:
				args.train = False
				print('\nTraining complete. Validation error for epoch %d: %g.                 ' % 
					(epoch_par['epoch_comp'], epoch_par['val_error']))
		elif epoch_par['end_idx'] > epoch_par['epoch_size']: epoch_par['end_idx'] = epoch_par['epoch_size'] # if less than the mini-batch size of examples is left.
 	
def infer(sess, net, args):
	print("Inference...")

	## LOAD MODEL
	net.saver.restore(sess, args.model_path + '/epoch-' + str(args.epoch)) # load model from epoch.

	## CONVERT STATISTIC CONSTANTS TO NUMPY ARRAY
	mu_np = sess.run(args.mu); # place mean constant into a numpy array.
	sigma_np = sess.run(args.sigma); # place standard deviation constant into a numpy array.

	for j in range(len(args.test_x_len)):
		x_MS_out = sess.run(net.x_MS, feed_dict={net.x_ph: [args.test_x[j]], 
			net.x_len_ph: [args.test_x_len[j]]}) 
		x_MS_3D_out = sess.run(net.x_MS_3D, feed_dict={net.x_ph: [args.test_x[j]], 
			net.x_len_ph: [args.test_x_len[j]]}) 
		x_PS_out = sess.run(net.x_PS, feed_dict={net.x_ph: [args.test_x[j]], 
			net.x_len_ph: [args.test_x_len[j]]}) 
		x_seq_len_out = sess.run(net.x_seq_len, feed_dict={net.x_len_ph: [args.test_x_len[j]]}) 

		output_out = sess.run(net.output, feed_dict={net.x_MS_ph: x_MS_3D_out, net.x_MS_len_ph: x_seq_len_out, net.training_ph: False}) # output of network.
		output_out = utils.np_sigmoid(output_out)
		xi_dB_hat_out = np.add(np.multiply(np.multiply(sigma_np, np.sqrt(2.0)), spsp.erfinv(np.subtract(np.multiply(2.0, output_out), 1))), mu_np); # a priori SNR estimate.			
		xi_hat_out = np.power(10.0, np.divide(xi_dB_hat_out, 10.0))
		
		if args.gain == 'mmse-stsa': gain_out = feat.mmse_stsa(xi_hat_out, feat.ml_gamma_hat(xi_hat_out)) # MMSE-STSA estimator gain.
		elif args.gain == 'mmse-lsa': gain_out = feat.mmse_lsa(xi_hat_out, feat.ml_gamma_hat(xi_hat_out)) # MMSE-LSA estimator gain.
		else: gain_out = sess.run(net.G, feed_dict={net.xi_hat_ph: xi_hat_out}) # gain.	

		if args.out_type == 'raw': # raw outputs from network (.mat).
			if not os.path.exists(args.out_path + '/raw'): os.makedirs(args.out_path + '/raw') # make output directory.
			spio.savemat(args.out_path + '/raw/' + args.test_fnames[j] + '.mat', {'raw':output_out})

		if args.out_type == 'xi_hat': # a priori SNR estimate output (.mat).
			if not os.path.exists(args.out_path + '/xi_hat'): os.makedirs(args.out_path + '/xi_hat') # make output directory.
			spio.savemat(args.out_path + '/xi_hat/' + args.test_fnames[j] + '.mat', {'xi_hat':xi_hat_out})
		
		if args.out_type == 'gain': # gain function output (.mat).
			if not os.path.exists(args.out_path + '/gain/' + gain): os.makedirs(args.out_path + '/gain/' + args.gain) # make output directory.
			spio.savemat(args.out_path + '/gain/' + args.gain + '/' + args.test_fnames[j] + '.mat', {gain:gain_out})

		if args.out_type == 'y': # enahnced speech output (.wav).
			if not os.path.exists(args.out_path + '/y/' + args.gain): os.makedirs(args.out_path + '/y/' + args.gain) # make output directory.
			y_out = sess.run(net.y, feed_dict={net.G_ph: gain_out, net.x_PS_ph: x_PS_out, net.x_MS_2D_ph: x_MS_out, 
				net.output_ph: output_out}) # enhanced speech output.		
			scipy.io.wavfile.write(args.out_path + '/y/' + args.gain + '/' + args.test_fnames[j] + '.wav', args.fs, y_out)

		print("Inference (%s): %3.2f%%.       " % (args.out_type, 100*((j+1)/len(args.test_x_len))), end="\r")
	print('\nInference complete.')

if __name__ == '__main__':
	## GET COMMAND LINE ARGUMENTS
	args = utils.args()

	## ARGUMENTS
	args.ver = '3a'
	args.blocks = ['C3'] + ['B5']*40 + ['O1']
	args.epoch = 175 # for inference.

	## TRAINING AND TESTING SET ARGUMENTS
	args = deepxi_args(args)

	## MAKE DEEP XI NNET
	net = deepxi_net(args)

	## GPU CONFIGURATION
	config = utils.gpu_config(args.gpu)

	with tf.Session(config=config) as sess:
		if args.train: train(sess, net, args)
		if args.infer: infer(sess, net, args)
