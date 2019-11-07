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
from tqdm import tqdm
import scipy.io.wavfile
import scipy.io as spio
import scipy.special as spsp
import os, random, math, time, sys, pickle
sys.path.insert(0, 'lib')
from dev.acoustic.analysis_synthesis.polar import synthesis
from dev.acoustic.feat import polar
from dev.args import add_args, get_args
from dev.ResNet import ResNet
import dev.se_batch as batch
import dev.utils as utils
import dev.optimisation as optimisation
import dev.gain as gain

np.set_printoptions(threshold=1e6)

## VARIABLE DESCRIPTIONS
# s - clean speech.
# d - noise.
# x - noisy speech.

## ARTIFICIAL NEURAL NETWORK 
class deepxi_net:
	def __init__(self, args):
		print('Preparing graph...')

		## RESNET		
		self.input_ph = tf.placeholder(tf.float32, shape=[None, None, args.d_in], name='input_ph') # noisy speech MS placeholder.
		self.nframes_ph = tf.placeholder(tf.int32, shape=[None], name='nframes_ph') # noisy speech MS sequence length placeholder.
		self.output = ResNet(self.input_ph, self.nframes_ph, args.norm_type, n_blocks=args.n_blocks, boolean_mask=True, d_out=args.d_out, 
			d_model=args.d_model, d_f=args.d_f, k_size=args.k_size, max_d_rate=args.max_d_rate)

		## TRAINING FEATURE EXTRACTION GRAPH
		self.s_ph = tf.placeholder(tf.int16, shape=[None, None], name='s_ph') # clean speech placeholder.
		self.d_ph = tf.placeholder(tf.int16, shape=[None, None], name='d_ph') # noise placeholder.
		self.s_len_ph = tf.placeholder(tf.int32, shape=[None], name='s_len_ph') # clean speech sequence length placeholder.
		self.d_len_ph = tf.placeholder(tf.int32, shape=[None], name='d_len_ph') # noise sequence length placeholder.
		self.snr_ph = tf.placeholder(tf.float32, shape=[None], name='snr_ph') # SNR placeholder.
		self.train_feat = polar.input_target_xi(self.s_ph, self.d_ph, self.s_len_ph, 
			self.d_len_ph, self.snr_ph, args.N_w, args.N_s, args.NFFT, args.f_s, args.stats['mu_hat'], args.stats['sigma_hat'])

		## INFERENCE FEATURE EXTRACTION GRAPH
		self.infer_feat = polar.input(self.s_ph, self.s_len_ph, args.N_w, args.N_s, args.NFFT, args.f_s)

		## PLACEHOLDERS
		self.x_ph = tf.placeholder(tf.int16, shape=[None, None], name='x_ph') # noisy speech placeholder.
		self.x_len_ph = tf.placeholder(tf.int32, shape=[None], name='x_len_ph') # noisy speech sequence length placeholder.
		self.target_ph = tf.placeholder(tf.float32, shape=[None, args.d_out], name='target_ph') # training target placeholder.
		self.keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph') # keep probability placeholder.
		self.training_ph = tf.placeholder(tf.bool, name='training_ph') # training placeholder.

		## SYNTHESIS GRAPH
		if args.infer:	
			self.infer_output = tf.nn.sigmoid(self.output)
			self.y_MAG_ph = tf.placeholder(tf.float32, shape=[None, None, args.d_in], name='y_MAG_ph') 
			self.x_PHA_ph = tf.placeholder(tf.float32, [None, None, args.d_in], name='x_PHA_ph')
			self.y = synthesis(self.y_MAG_ph, self.x_PHA_ph, args.N_w, args.N_s, args.NFFT)

		## LOSS & OPTIMIZER
		self.loss = optimisation.loss(self.target_ph, self.output, 'mean_sigmoid_cross_entropy', axis=[1])
		self.total_loss = tf.reduce_mean(self.loss, axis=0)
		self.trainer, _ = optimisation.optimiser(self.total_loss, optimizer='adam', grad_clip=True)

		## SAVE VARIABLES
		self.saver = tf.train.Saver(max_to_keep=256)

		## NUMBER OF PARAMETERS
		args.params = (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

## GET STATISTICS OF SAMPLE
def get_stats(args, config):
	if os.path.exists(args.data_path + '/' + args.ver + '_' + args.set_path.rsplit('/', 1)[-1] + '/stats.p'):
		print('Loading sample statistics from pickle file...')
		with open(args.data_path + '/' + args.ver + '_' + args.set_path.rsplit('/', 1)[-1] + '/stats.p', 'rb') as f:
			args.stats = pickle.load(f)
		return args
	else:
		print('Finding sample statistics...')
		random.shuffle(args.train_s_list) # shuffle list.
		s_sample, s_sample_seq_len = batch.Clean_mbatch(args.train_s_list, 
			args.sample_size, 0, args.sample_size) # generate mini-batch of clean training waveforms.
		d_sample, d_sample_seq_len = batch.Noise_mbatch(args.train_d_list, 
			args.sample_size, s_sample_seq_len) # generate mini-batch of noise training waveforms.
		snr_sample = np.random.randint(args.min_snr, args.max_snr + 1, args.sample_size) # generate mini-batch of SNR levels.
		s_ph = tf.placeholder(tf.int16, shape=[None, None], name='s_ph') # clean speech placeholder.
		d_ph = tf.placeholder(tf.int16, shape=[None, None], name='d_ph') # noise placeholder.
		s_len_ph = tf.placeholder(tf.int32, shape=[None], name='s_len_ph') # clean speech sequence length placeholder.
		d_len_ph = tf.placeholder(tf.int32, shape=[None], name='d_len_ph') # noise sequence length placeholder.
		snr_ph = tf.placeholder(tf.float32, shape=[None], name='snr_ph') # SNR placeholder.
		analysis = polar.target_xi(s_ph, d_ph, s_len_ph, d_len_ph, snr_ph, args.N_w, args.N_s, args.NFFT, args.f_s)
		sample_graph = analysis[0]
		samples = []
		with tf.Session(config=config) as sess:
			for i in tqdm(range(s_sample.shape[0])):
				sample = sess.run(sample_graph, feed_dict={s_ph: [s_sample[i]], d_ph: [d_sample[i]], s_len_ph: [s_sample_seq_len[i]], 
					d_len_ph: [d_sample_seq_len[i]], snr_ph: [snr_sample[i]]}) # sample of training set.
				samples.append(sample)
		samples = np.vstack(samples)
		args.stats = {'mu_hat': np.mean(samples, axis=0), 'sigma_hat': np.std(samples, axis=0)}
		if not os.path.exists(args.data_path + '/' + args.ver + '_' + args.set_path.rsplit('/', 1)[-1]): os.makedirs(args.data_path + 
			'/' + args.ver + '_' + args.set_path.rsplit('/', 1)[-1]) # make directory.
		with open(args.data_path + '/' + args.ver + '_' + args.set_path.rsplit('/', 1)[-1] + '/stats.p', 'wb') as f: 		
			pickle.dump(args.stats, f)
		spio.savemat(args.data_path + '/' + args.ver + '_' + args.set_path.rsplit('/', 1)[-1] + 
			'/stats.mat', mdict={'mu_hat': args.stats['mu_hat'], 'sigma_hat': args.stats['sigma_hat']})
		print('Sample statistics saved to pickle file.')
	return args

## TRAINING
def train(sess, net, args):
	print("Training...")

	## CONTINUE FROM LAST EPOCH
	if args.cont:
		epoch_size = len(args.train_s_list); epoch_comp = args.epoch; start_idx = 0; 
		end_idx = args.mbatch_size; val_error = float("inf") # create epoch parameters.
		net.saver.restore(sess, args.model_path + '/epoch-' + str(args.epoch)) # load model from last epoch.

	## TRAIN RAW NETWORK
	else:
		epoch_size = len(args.train_s_list); epoch_comp = 0; start_idx = 0;
		end_idx = args.mbatch_size; val_error = float("inf") # create epoch parameters.
		if args.mbatch_size > epoch_size: raise ValueError('Error: mini-batch size is greater than the epoch size.')
		sess.run(tf.global_variables_initializer()) # initialise model variables.
		net.saver.save(sess, args.model_path + '/epoch', global_step=epoch_comp) # save model.

	## TRAINING LOG
	if not os.path.exists('log'): os.makedirs('log') # create log directory.
	with open("log/" + args.ver + ".csv", "a") as results: results.write("'"'Validation error'"', '"'Training error'"', '"'Epoch'"', '"'D/T'"'\n")

	train_err = 0; mbatch_count = 0
	while args.train:

		print('Training E%d (ver=%s, gpu=%s, params=%g)...' % (epoch_comp + 1, args.ver, args.gpu, args.params))
		for _ in tqdm(range(args.train_steps)):

			## MINI-BATCH GENERATION
			mbatch_size_iter = end_idx - start_idx # number of examples in mini-batch for the training iteration.
			s_mbatch, s_mbatch_seq_len = batch.Clean_mbatch(args.train_s_list, 
				mbatch_size_iter, start_idx, end_idx) # generate mini-batch of clean training waveforms.
			d_mbatch, d_mbatch_seq_len = batch.Noise_mbatch(args.train_d_list, 
				mbatch_size_iter, s_mbatch_seq_len) # generate mini-batch of noise training waveforms.
			snr_mbatch = np.random.randint(args.min_snr, args.max_snr + 1, end_idx - start_idx) # generate mini-batch of SNR levels.

			## TRAINING ITERATION
			mbatch = sess.run(net.train_feat, feed_dict={net.s_ph: s_mbatch, net.d_ph: d_mbatch, 
				net.s_len_ph: s_mbatch_seq_len, net.d_len_ph: d_mbatch_seq_len, net.snr_ph: snr_mbatch}) # mini-batch.
			[_, mbatch_err] = sess.run([net.trainer, net.total_loss], feed_dict={net.input_ph: mbatch[0], net.target_ph: mbatch[1], 
			net.nframes_ph: mbatch[2], net.training_ph: True}) # training iteration.						
			if not math.isnan(mbatch_err):
				train_err += mbatch_err; mbatch_count += 1

			## UPDATE EPOCH PARAMETERS
			start_idx += args.mbatch_size; end_idx += args.mbatch_size # start and end index of mini-batch.
			if end_idx > epoch_size: end_idx = epoch_size # if less than the mini-batch size of examples is left.

		## VALIDATION SET ERROR
		start_idx = 0; end_idx = args.mbatch_size # reset start and end index of mini-batch.
		random.shuffle(args.train_s_list) # shuffle list.
		start_idx = 0; end_idx = args.mbatch_size; frames = 0; val_error = 0; # validation variables.
		print('Validation error for E%d...' % (epoch_comp + 1))
		for _ in tqdm(range(args.val_steps)):
			mbatch = sess.run(net.train_feat, feed_dict={net.s_ph: args.val_s[start_idx:end_idx], 
				net.d_ph: args.val_d[start_idx:end_idx], net.s_len_ph: args.val_s_len[start_idx:end_idx], 
				net.d_len_ph: args.val_d_len[start_idx:end_idx], net.snr_ph: args.val_snr[start_idx:end_idx]}) # mini-batch.
			val_error_mbatch = sess.run(net.loss, feed_dict={net.input_ph: mbatch[0], 
				net.target_ph: mbatch[1], net.nframes_ph: mbatch[2], net.training_ph: False}) # validation error for each frame in mini-batch.
			val_error += np.sum(val_error_mbatch)
			frames += mbatch[1].shape[0] # total number of frames.
			print("Validation error for Epoch %d: %3.2f%% complete.       " % 
				(epoch_comp + 1, 100*(end_idx/args.val_s_len.shape[0])), end="\r")
			start_idx += args.mbatch_size; end_idx += args.mbatch_size
			if end_idx > args.val_s_len.shape[0]: end_idx = args.val_s_len.shape[0]
		val_error /= frames # validation error.
		epoch_comp += 1 # an epoch has been completed.
		net.saver.save(sess, args.model_path + '/epoch', global_step=epoch_comp) # save model.
		print("E%d: train err=%3.2f, val err=%3.2f.           " % 
			(epoch_comp, train_err/mbatch_count, val_error))
		with open("log/" + args.ver + ".csv", "a") as results:
			results.write("%g, %g, %d, %s\n" % (val_error, train_err/mbatch_count,
			epoch_comp, datetime.now().strftime('%Y-%m-%d/%H:%M:%S')))
		train_err = 0; mbatch_count = 0; start_idx = 0; end_idx = args.mbatch_size

		if epoch_comp >= args.max_epochs:
			args.train = False
			print('\nTraining complete. Validation error for epoch %d: %g.                 ' % 
				(epoch_comp, val_error))

## INFERENCE
def infer(sess, net, args):
	print("Inference...")
	net.saver.restore(sess, args.model_path + '/epoch-' + str(args.epoch)) # load model from epoch.
	
	if args.out_type == 'xi_hat':
		args.out_path = args.out_path + '/xi_hat'
		if not os.path.exists(args.out_path): os.makedirs(args.out_path) # make output directory.
	elif args.out_type == 'y':
		args.out_path = args.out_path + '/' + args.gain + '/y'
		if not os.path.exists(args.out_path): os.makedirs(args.out_path) # make output directory.
	else: ValueError('Incorrect output type.')

	for j in range(len(args.test_x_len)):
		input_feat = sess.run(net.infer_feat, feed_dict={net.s_ph: [args.test_x[j][0:args.test_x_len[j]]], 
			net.s_len_ph: [args.test_x_len[j]]}) # sample of training set.
	
		xi_mapped_hat = sess.run(net.infer_output, feed_dict={net.input_ph: input_feat[0], 
			net.nframes_ph: input_feat[1], net.P_drop_ph: 0.0, net.training_ph: False}) # output of network.
		xi_dB_hat = np.add(np.multiply(np.multiply(args.stats['sigma_hat'], np.sqrt(2.0)), 
			spsp.erfinv(np.subtract(np.multiply(2.0, xi_mapped_hat), 1))), args.stats['mu_hat']); # a priori SNR estimate.			
		xi_hat = np.power(10.0, np.divide(xi_dB_hat, 10.0))
		
		if args.out_type == 'xi_hat':
			spio.savemat(args.out_path + '/' + args.test_fnames[j] + '.mat', {'xi_hat':xi_hat})

		elif args.out_type == 'y':
			y_MAG = np.multiply(input_feat[0], gain.gfunc(xi_hat, xi_hat+1, gtype=args.gain))
			y = np.squeeze(sess.run(net.y, feed_dict={net.y_MAG_ph: y_MAG, 
				net.x_PHA_ph: input_feat[2], net.nframes_ph: input_feat[1], net.training_ph: False})) # output of network.
			if np.isnan(y).any(): ValueError('NaN values found in enhanced speech.')
			if np.isinf(y).any(): ValueError('Inf values found in enhanced speech.')
			utils.save_wav(args.out_path + '/' + args.test_fnames[j] + '.wav', args.f_s, y)

		print("Inference (%s): %3.2f%%.       " % (args.out_type, 100*((j+1)/len(args.test_x_len))), end="\r")
	print('\nInference complete.')

if __name__ == '__main__':
	## GET COMMAND LINE ARGUMENTS
	args = get_args()

	## TRAINING AND TESTING SET ARGUMENTS
	args = add_args(args)

	## GPU CONFIGURATION
	config = utils.gpu_config(args.gpu)

	## GET STATISTICS
	args = get_stats(args, config)

	## MAKE DEEP XI NNET
	net = deepxi_net(args)

	with tf.Session(config=config) as sess:
		if args.train: train(sess, net, args)
		if args.infer: infer(sess, net, args)
