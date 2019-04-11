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
import os, argparse, random, math, time, sys, pickle
sys.path.insert(0, '../../../../lib')
import feat, batch, residual
np.set_printoptions(threshold=1e6)

print('DeepXi: a priori SNR estimator')

## ARGUMENTS
parser = argparse.ArgumentParser()

## OPTIONS (GENERAL)
parser.add_argument('--gpu', default='0', type=str, help='GPU selection')
parser.add_argument('--ver', default='c1.13a', type=str, help='Model version')
parser.add_argument('--par_iter', default=256, type=int, help='dynamic_rnn/bidirectional_dynamic_rnn parallel iterations')

## OPTIONS (TRAIN)
parser.add_argument('--train', default=False, type=bool, help='Training flag')
parser.add_argument('--cont', default=False, type=bool, help='Continue testing from last epoch')
parser.add_argument('--train_clean_speech_ver', default='v1', type=str, help='Clean speech training set version')
parser.add_argument('--train_noise_ver', default='v1', type=str, help='Noise training set version')
parser.add_argument('--mbatch_size', default=10, type=int, help='Mini-batch size')
parser.add_argument('--sample_size', default=250, type=int, help='Sample size')
parser.add_argument('--max_epochs', default=100, type=int, help='Maximum number of epochs')

## OPTIONS (TEST)
parser.add_argument('--test', default=False, type=bool, help='Testing flag')
parser.add_argument('--val', default=False, type=bool, help='Find validation error for "test_epoch"')
parser.add_argument('--test_epoch', default=15, type=int, help='Epoch for testing')
parser.add_argument('--test_noisy_speech_ver', default='v2', type=str, help='Noisy speech test set version')

# TEST OUTPUT TYPE
# 'raw' - output from network (.mat), 'xi_hat' - a priori SNR estimate (.mat),
# 'gain' - gain function (.mat), 'y' - enhanced speech (.wav).
parser.add_argument('--out_type', default='y', type=str, help='Output type for testing')

## GAIN FUNCTION
# 'ibm' - Ideal Binary Mask (IBM), 'wf' - Wiener Filter (WF), 'srwf' - Square-Root Wiener Filter (SRWF),
# 'cwf' - Constrained Wiener Filter (cWF), 'mmse-stsa' - Minimum-Mean Square Error - Short-Time Spectral Amplitude (MMSE-STSA) estimator,
# 'mmse-lsa' - Minimum-Mean Square Error - Log-Spectral Amplitude (MMSE-LSA) estimator.
parser.add_argument('--gain', default='mmse-lsa', type=str, help='Gain function for testing')

## PATHS
parser.add_argument('--model_path', default='./../../../../model', type=str, help='Model save path')
parser.add_argument('--set_path', default='./../../../../set', type=str, help='Path to datasets')
parser.add_argument('--stats_path', default='./../../../../stats', 
	type=str, help='Path to training set statistics')
parser.add_argument('--test_noisy_speech_path', default='./../../../../set/test_noisy_speech', 
	type=str, help='Path to the noisy speech test set')
parser.add_argument('--out_path', default='./../../../../out', 
	type=str, help='Output path')

## NETWORK PARAMETERS
parser.add_argument('--blocks', default=['C3'] + ['B3']*5 + ['O1'], type=list, help='Residual blocks')
parser.add_argument('--cell_size', default=512, type=int, help='Cell size')
parser.add_argument('--cell_proj', default=None, type=int, help='Cell projection size (None for no proj.)')
parser.add_argument('--cell_type', default='LSTMCell', type=str, help='RNN cell type')
parser.add_argument('--peep', default=False, type=bool, help='Use peephole connections')
parser.add_argument('--bidi', default=False, type=bool, help='Bidirectional recurrent neural network flag')
parser.add_argument('--bidi_con', default=None, type=str, help='Forward and backward cell activation connection')
parser.add_argument('--res_con', default='add', type=str, help='Residual connection (add or concat)')
parser.add_argument('--res_proj', default=None, type=int, help='Output size of the residual projection weight (None for no projection)')
parser.add_argument('--block_unit', default=None, type=str, help='Residual unit')
parser.add_argument('--coup_unit', default='CU1', type=str, help='Coupling unit')
parser.add_argument('--coup_conv_filt', default=512, type=int, help='Number of filters for coupling unit')
parser.add_argument('--conv_size', default=None, type=int, help='Convolution kernel size')
parser.add_argument('--conv_filt', default=None, type=int, help='Number of convolution filters')
parser.add_argument('--conv_caus', default=None, type=bool, help='Causal convolution flag')
parser.add_argument('--max_dilation_rate', default=None, type=int, help='Maximum dilation rate')
parser.add_argument('--dropout', default=False, type=bool, help='Use droput during training flag')
parser.add_argument('--keep_prob', default=False, type=float, help='Keep probability during training (0.75 typically)')
parser.add_argument('--context', default=None, type=int, help='Input context (no. of frames)')
parser.add_argument('--depth', default=None, type=int, help='Temporal residual-dense lattice (tRDL) depth')
parser.add_argument('--dilation_strat', default=None, type=str, help='tRDL dilation strategy (height)')
parser.add_argument('--verbose', default=False, type=bool, help='Verbose')

## FEATURES
parser.add_argument('--min_snr', default=-10, type=int, help='Minimum trained SNR level')
parser.add_argument('--max_snr', default=20, type=int, help='Maximum trained SNR level')
parser.add_argument('--input_dim', default=257, type=int, help='Number of inputs')
parser.add_argument('--num_outputs', default=257, type=int, help='Number of outputs')
parser.add_argument('--fs', default=16000, type=int, help='Sampling frequency (Hz)')
parser.add_argument('--Tw', default=32, type=int, help='Window length (ms)')
parser.add_argument('--Ts', default=16, type=int, help='Window shift (ms)')
parser.add_argument('--nconst', default=32768, type=int, help='Normalisation constant (see feat.addnoisepad())')
args = parser.parse_args()
print("Version: %s on GPU:%s." % (args.ver, args.gpu)) # print version.

## DEPENDANT OPTIONS
args.model_path = args.model_path + '/' + args.ver # model save path.
args.train_clean_speech_path = args.set_path + '/train_clean_speech' # path to the clean speech training set.
args.train_noise_path = args.set_path + '/train_noise' # path to the clean speech training set.
args.val_clean_speech_path = args.set_path + '/val_clean_speech' # path to the clean speech validation set.
args.val_noise_path = args.set_path + '/val_noise' # path to the noise validation set.
args.out_path = args.out_path + '/' + args.ver + '/' + 'e' + str(args.test_epoch) # output path.
args.Nw = int(args.fs*args.Tw*0.001) # window length (samples).
args.Ns = int(args.fs*args.Ts*0.001) # window shift (samples).
args.NFFT = int(pow(2, np.ceil(np.log2(args.Nw)))) # number of DFT components.

## A PRIORI SNR IN DB STATS
mu_mat = spio.loadmat(args.stats_path + '/mu.mat') # mean of a priori SNR in dB from MATLAB.
mu = tf.constant(mu_mat['mu'], dtype=tf.float32) 
sigma_mat = spio.loadmat(args.stats_path + '/sigma.mat') # standard deviation of a priori SNR in dB from MATLAB.
sigma = tf.constant(sigma_mat['sigma'], dtype=tf.float32) 

## DATASETS
if args.train: ## TRAINING CLEAN SPEECH AND NOISE SET
	train_clean_speech_list = batch._train_list(args.train_clean_speech_path, '*.wav', 'clean') # clean speech training list.
	train_noise_list = batch._train_list(args.train_noise_path, '*.wav', 'noise') # noise training list.
	if not os.path.exists(args.model_path): os.makedirs(args.model_path) # make model path directory.

if args.train or args.val: ## VALIDATION CLEAN SPEECH AND NOISE SET
	val_clean_speech, val_clean_speech_len, val_snr, _ = batch._batch(args.val_clean_speech_path, '*.wav', 
		list(range(args.min_snr, args.max_snr + 1))) # clean validation waveforms and lengths.
	val_noise, val_noise_len, _, _ = batch._batch(args.val_noise_path, '*.wav', 
		list(range(args.min_snr, args.max_snr + 1))) # noise validation waveforms and lengths.

## TEST NOISY SPEECH SET
if args.test: test_noisy_speech, test_noisy_speech_len, test_snr, test_fnames = batch._batch(args.test_noisy_speech_path, '*.wav', []) # noisy speech test waveforms and lengths.

## GPU CONFIGURATION
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
config = tf.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.allow_growth=True
config.log_device_placement=False

## PLACEHOLDERS
s_ph = tf.placeholder(tf.int16, shape=[None, None], name='s_ph') # clean speech placeholder.
d_ph = tf.placeholder(tf.int16, shape=[None, None], name='d_ph') # noise placeholder.
x_ph = tf.placeholder(tf.int16, shape=[None, None], name='x_ph') # noisy speech placeholder.
s_len_ph = tf.placeholder(tf.int32, shape=[None], name='s_len_ph') # clean speech sequence length placeholder.
d_len_ph = tf.placeholder(tf.int32, shape=[None], name='d_len_ph') # noise sequence length placeholder.
x_len_ph = tf.placeholder(tf.int32, shape=[None], name='x_len_ph') # noisy speech sequence length placeholder.
snr_ph = tf.placeholder(tf.float32, shape=[None], name='snr_ph') # SNR placeholder.
x_MS_ph = tf.placeholder(tf.float32, shape=[None, None, args.input_dim], name='x_MS_ph') # noisy speech MS placeholder.
x_MS_len_ph = tf.placeholder(tf.int32, shape=[None], name='x_MS_len_ph') # noisy speech MS sequence length placeholder.
target_ph = tf.placeholder(tf.float32, shape=[None, args.input_dim], name='target_phh') # training target placeholder.
keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph') # keep probability placeholder.
training_ph = tf.placeholder(tf.bool, name='training_ph') # training placeholder.

## LOG_10
def log10(x):
  numerator = tf.log(x)
  denominator = tf.constant(np.log(10), dtype=numerator.dtype)
  return tf.div(numerator, denominator)

## FEATURE EXTRACTION FUNCTION
def feat_extr(s, d, s_len, d_len, Q, Nw, Ns, NFFT, fs, P, nconst, mu, sigma):
	'''
	Extracts input features and targets from given clean speech and noise.

	Inputs:
		s - clean waveform (dtype=tf.int32).
		d - noisy waveform (dtype=tf.int32).
		s_len - clean waveform length without padding (samples).
		d_len - noise waveform length without padding (samples).
		Q - SNR level.
		Nw - window length (samples).
		Ns - window shift (samples).
		NFFT - DFT components.
		fs - sampling frequency (Hz).
		P - padded waveform length (samples).
		nconst - normalization constant.
		mu - mean of a priori SNR in dB.
		sigma - standard deviation of a priori SNR in dB.

	Outputs:
		x_MS - padded noisy single-sided magnitude spectrum.
		xi_mapped - mapped a priori SNR.	
		seq_len - length of each sequence without padding.
	'''
	(s, x, d) = tf.map_fn(lambda z: feat.addnoisepad(z[0], z[1], z[2], z[3], z[4],
		P, nconst), (s, d, s_len, d_len, Q), dtype=(tf.float32, tf.float32,
		tf.float32)) # padded waveforms.
	seq_len = feat.nframes(s_len, Ns) # length of each sequence.
	s_MS = feat.stms(s, Nw, Ns, NFFT) # clean speech magnitude spectrum.
	d_MS = feat.stms(d, Nw, Ns, NFFT) # noise magnitude spectrum.
	x_MS = feat.stms(x, Nw, Ns, NFFT) # noisy speech magnitude spectrum.
	xi = tf.div(tf.square(tf.maximum(s_MS, 1e-12)), tf.square(tf.maximum(d_MS, 1e-12))) # a priori SNR.
	xi_dB = tf.multiply(10.0, log10(xi)) # a priori SNR dB.
	xi_mapped = tf.multiply(0.5, tf.add(1.0, tf.erf(tf.div(tf.subtract(xi_dB, mu), 
		tf.multiply(sigma, tf.sqrt(2.0)))))) # cdf of a priori SNR in dB.
	xi_mapped = tf.boolean_mask(xi_mapped, tf.sequence_mask(seq_len)) # convert to 2D.
	return (x_MS, xi_mapped, seq_len)

## FEATURE GRAPH
print('Preparing graph...')
P = tf.reduce_max(s_len_ph) # padded waveform length.
feature = feat_extr(s_ph, d_ph, s_len_ph, d_len_ph, snr_ph, args.Nw, args.Ns, args.NFFT, 
	args.fs, P, args.nconst, mu, sigma) # feature graph.

## RESNET
output = residual.Residual(x_MS_ph, x_MS_len_ph, keep_prob_ph, 
	training_ph, args.num_outputs, args)

## LOSS & OPTIMIZER
loss = residual.loss(target_ph, output, 'sigmoid_cross_entropy')
total_loss = tf.reduce_mean(loss, axis=0)
trainer, _ = residual.optimizer(total_loss, optimizer='adam', grad_clip=True)

## SAVE VARIABLES
saver = tf.train.Saver(max_to_keep=256)

## NUMBER OF PARAMETERS
if args.verbose: print("No. of trainable parameters: %g." % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

## TRAINING
if args.train:
	print("Training...")
	with tf.Session(config=config) as sess:

		## CONTINUE FROM LAST EPOCH
		if args.cont:
			with open('data/epoch_par_' + args.ver + '.p', 'rb') as f: epoch_par = pickle.load(f) # load epoch parameters from last epoch.
			epoch_par['start_idx'] = 0; epoch_par['end_idx'] = args.mbatch_size # reset start and end index of mini-batch. 
			random.shuffle(train_clean_speech_list) # shuffle list.
			with open('data/epoch_par_' + args.ver + '.p', 'wb') as f: pickle.dump(epoch_par, f) # save epoch parameters.
			saver.restore(sess, args.model_path + '/epoch-' + str(epoch_par['epoch_comp'])) # load model from last epoch.

		## TRAIN RAW NETWORK
		else:
			if os.path.isfile('data/epoch_par_' + args.ver + '.p'): os.remove('data/epoch_par_' + args.ver + '.p') # remove epoch parameters.
			print('Creating epoch parameters, as no pickle file exists...')
			epoch_par = {'epoch_size': len(train_clean_speech_list), 'epoch_comp': 0, 
				'start_idx': 0, 'end_idx': args.mbatch_size, 'val_error': float("inf")} # create epoch parameters.
			if args.mbatch_size > epoch_par['epoch_size']: raise ValueError('Error: mini-batch size is greater than the epoch size.')
			with open('data/epoch_par_' + args.ver + '.p', 'wb') as f:
				pickle.dump(epoch_par, f) # save epoch parameters.
			sess.run(tf.global_variables_initializer()) # initialise model variables.
			saver.save(sess, args.model_path + '/epoch', global_step=epoch_par['epoch_comp']) # save model.

		## TRAINING LOG
		if not os.path.exists('log'): os.makedirs('log') # create log directory.
		with open("log/val_error_" + args.ver + ".txt", "a") as results:
			results.write("val err, train err, epoch count, D/T:\n")

		train_err = 0; mbatch_count = 0
		while args.train:
			## MINI-BATCH GENERATION
			mbatch_size_iter = epoch_par['end_idx'] - epoch_par['start_idx'] # number of examples in mini-batch for the training iteration.
			train_clean_speech_mbatch, train_clean_speech_mbatch_seq_len = batch._clean_mbatch(train_clean_speech_list, 
				mbatch_size_iter, epoch_par['start_idx'], epoch_par['end_idx']) # generate mini-batch of clean training waveforms.
			train_noise_mbatch, train_noise_mbatch_seq_len = batch._noise_mbatch(train_noise_list, 
				mbatch_size_iter, train_clean_speech_mbatch_seq_len) # generate mini-batch of noise training waveforms.
			train_snr_mbatch = np.random.randint(args.min_snr, args.max_snr + 1, epoch_par['end_idx'] - epoch_par['start_idx']) # generate mini-batch of SNR levels.

			## TRAINING ITERATION
			train_mbatch = sess.run(feature, feed_dict={s_ph: train_clean_speech_mbatch, d_ph: train_noise_mbatch, 
				s_len_ph: train_clean_speech_mbatch_seq_len, d_len_ph: train_noise_mbatch_seq_len, snr_ph: train_snr_mbatch}) # mini-batch.
			[_, train_mbatch_err] = sess.run([trainer, total_loss], feed_dict={x_MS_ph: train_mbatch[0], target_ph: train_mbatch[1], 
				x_MS_len_ph: train_mbatch[2], training_ph: True}) # training iteration.
			train_err += train_mbatch_err; mbatch_count += 1
			print("E%d: %3.1f%% (terr %3.2f), E%d verr: %3.2f, %s, GPU:%s.       " % 
				(epoch_par['epoch_comp'] + 1, 100*(epoch_par['end_idx']/epoch_par['epoch_size']), train_err/mbatch_count,
				epoch_par['epoch_comp'], epoch_par['val_error'], args.ver, args.gpu), end="\r")

			## UPDATE EPOCH PARAMETERS
			epoch_par['start_idx'] += args.mbatch_size; epoch_par['end_idx'] += args.mbatch_size # start and end index of mini-batch.

			## VALIDATION SET ERROR
			if epoch_par['start_idx'] >= epoch_par['epoch_size']:
				epoch_par['start_idx'] = 0; epoch_par['end_idx'] = args.mbatch_size # reset start and end index of mini-batch.
				random.shuffle(train_clean_speech_list) # shuffle list.
				start_idx = 0; end_idx = args.mbatch_size; val_flag = True; frames = 0; epoch_par['val_error'] = 0; # validation variables.
				while val_flag:
					val_mbatch = sess.run(feature, feed_dict={s_ph: val_clean_speech[start_idx:end_idx], d_ph: val_noise[start_idx:end_idx], 
						s_len_ph: val_clean_speech_len[start_idx:end_idx], d_len_ph: val_noise_len[start_idx:end_idx], snr_ph: val_snr[start_idx:end_idx]}) # mini-batch.
					val_error_mbatch = sess.run(loss, feed_dict={x_MS_ph: val_mbatch[0], target_ph: val_mbatch[1], 
						x_MS_len_ph: val_mbatch[2], training_ph: False}) # validation error for each frame in mini-batch.
					frames += val_error_mbatch.shape[0] # total number of frames.
					epoch_par['val_error'] += np.sum(val_error_mbatch)
					print("Validation error for Epoch %d: %3.2f%% complete.       " % 
						(epoch_par['epoch_comp'] + 1, 100*(end_idx/val_clean_speech_len.shape[0])), end="\r")
					start_idx += args.mbatch_size; end_idx += args.mbatch_size
					if start_idx >= val_clean_speech_len.shape[0]: val_flag = False
					elif end_idx > val_clean_speech_len.shape[0]: end_idx = val_clean_speech_len.shape[0]
				epoch_par['val_error'] /= frames # validation error.
				epoch_par['epoch_comp'] += 1 # an epoch has been completed.
				saver.save(sess, args.model_path + '/epoch', global_step=epoch_par['epoch_comp']) # save model.
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
 	
def ml_gamma_hat(xi_hat):
	return np.add(xi_hat, 1) # ML estimate of a posteriori SNR.

def mmse_stsa(xi, gamma):
	nu = np.multiply(xi, np.divide(gamma, np.add(1, xi)))
	G = np.multiply(np.multiply(np.multiply(np.divide(np.sqrt(np.pi), 2), 
		np.divide(np.sqrt(nu), gamma)), np.exp(np.divide(-nu,2))), 
		np.add(np.multiply(np.add(1, nu), spsp.i0(np.divide(nu,2))), 
		np.multiply(nu, spsp.i1(np.divide(nu, 2))))) # MMSE-STSA gain function.
	idx = np.isnan(G) | np.isinf(G) # replace by Wiener gain.
	G[idx] = np.divide(xi[idx], np.add(1, xi[idx])) # Wiener gain.
	return G

def mmse_lsa(xi, gamma):
	nu = np.multiply(np.divide(xi, np.add(1, xi)), gamma)
	return np.multiply(np.divide(xi, np.add(1, xi)), np.exp(np.multiply(0.5, spsp.exp1(nu)))) # MMSE-LSA gain function.

def np_sigmoid(x): return np.divide(1, np.add(1, np.exp(np.negative(x))))

if args.test:
	print("Test...")

	## TEST PLACEHOLDERS
	output_ph = tf.placeholder(tf.float32, shape=[None, args.input_dim], name='output_ph') # network output placeholder.
	x_MS_2D_ph = tf.placeholder(tf.float32, shape=[None, args.input_dim], name='x_MS_2D_ph') # noisy speech MS placeholder (in 2D form).
	x_PS_ph = tf.placeholder(tf.float32, shape=[None, args.input_dim], name='x_PS_ph') # noisy speech PS placeholder.
	xi_hat_ph = tf.placeholder(tf.float32, shape=[None, args.input_dim], name='xi_hat_ph') # a priori SNR estimate placeholder.
	G_ph = tf.placeholder(tf.float32, shape=[None, args.input_dim], name='G_ph') # gain function placeholder.

	## ANALYSIS
	x = tf.div(tf.cast(tf.slice(tf.squeeze(x_ph), [0], [tf.squeeze(x_len_ph)]), tf.float32), args.nconst) # remove padding and normalise.
	x_DFT = feat.stft(x, args.Nw, args.Ns, args.NFFT) # noisy speech single-sided short-time Fourier transform.
	x_MS_3D = tf.expand_dims(tf.abs(x_DFT), 0) # noisy speech single-sided magnitude spectrum (in 3D form).
	x_MS = tf.abs(x_DFT) # noisy speech single-sided magnitude spectrum.
	x_PS = tf.angle(x_DFT) # noisy speech single-sided phase spectrum.
	x_seq_len = feat.nframes(x_len_ph, args.Ns) # length of each sequence.

	## ENHANCEMENT
	if args.gain == 'ibm': G = tf.cast(tf.greater(xi_hat_ph, 1), tf.float32) # IBM gain function.
	if args.gain == 'wf': G = tf.div(xi_hat_ph, tf.add(xi_hat_ph, 1.0)) # WF gain function.
	if args.gain == 'srwf': G = tf.sqrt(tf.div(xi_hat_ph, tf.add(xi_hat_ph, 1.0))) # SRWF gain function.
	if args.gain == 'irm': G = tf.sqrt(tf.div(xi_hat_ph, tf.add(xi_hat_ph, 1.0))) # IRM gain function.
	if args.gain == 'cwf':
		G = tf.sqrt(xi_hat_ph)
		G = tf.div(G, tf.add(G, 1.0)) # SPP gain function.
	s_hat_MS = tf.multiply(x_MS_2D_ph, G_ph) # enhanced speech single-sided magnitude spectrum.
	
	## SYNTHESIS GRAPH
	y_DFT = tf.cast(s_hat_MS, tf.complex64) * tf.exp(1j * tf.cast(x_PS_ph, tf.complex64)) # enhanced speech single-sided short-time Fourier transform.
	
	y = tf.contrib.signal.inverse_stft(y_DFT, args.Nw, args.Ns, args.NFFT, 
		tf.contrib.signal.inverse_stft_window_fn(args.Ns, 
		forward_window_fn=tf.contrib.signal.hamming_window)) # synthesis.

	## INFERENCE
	with tf.Session(config=config) as sess:

		## LOAD MODEL
		saver.restore(sess, args.model_path + '/epoch-' + str(args.test_epoch)) # load model from epoch.

		## CONVERT STATISTIC CONSTANTS TO NUMPY ARRAY
		mu_np = sess.run(mu); # place mean constant into a numpy array.
		sigma_np = sess.run(sigma); # place standard deviation constant into a numpy array.

		for j in range(len(test_noisy_speech_len)):
			x_MS_out = sess.run(x_MS, feed_dict={x_ph: [test_noisy_speech[j]], 
				x_len_ph: [test_noisy_speech_len[j]]}) 
			x_MS_3D_out = sess.run(x_MS_3D, feed_dict={x_ph: [test_noisy_speech[j]], 
				x_len_ph: [test_noisy_speech_len[j]]}) 
			x_PS_out = sess.run(x_PS, feed_dict={x_ph: [test_noisy_speech[j]], 
				x_len_ph: [test_noisy_speech_len[j]]}) 
			x_seq_len_out = sess.run(x_seq_len, feed_dict={x_len_ph: [test_noisy_speech_len[j]]}) 

			output_out = sess.run(output, feed_dict={x_MS_ph: x_MS_3D_out, x_MS_len_ph: x_seq_len_out, training_ph: False}) # output of network.
			output_out = np_sigmoid(output_out)
			xi_dB_hat_out = np.add(np.multiply(np.multiply(sigma_np, np.sqrt(2.0)), spsp.erfinv(np.subtract(np.multiply(2.0, output_out), 1))), mu_np); # a priori SNR estimate.			
			xi_hat_out = np.power(10.0, np.divide(xi_dB_hat_out, 10.0))
			
			if args.gain == 'mmse-stsa': gain_out = mmse_stsa(xi_hat_out, ml_gamma_hat(xi_hat_out)) # MMSE-STSA estimator gain.
			elif args.gain == 'mmse-lsa': gain_out = mmse_lsa(xi_hat_out, ml_gamma_hat(xi_hat_out)) # MMSE-LSA estimator gain.
			else: gain_out = sess.run(G, feed_dict={xi_hat_ph: xi_hat_out}) # gain.	

			if args.out_type == 'raw': # raw outputs from network (.mat).
				if not os.path.exists(args.out_path + '/raw'): os.makedirs(args.out_path + '/raw') # make output directory.
				spio.savemat(args.out_path + '/raw/' + test_fnames[j], {'raw':output_out})

			if args.out_type == 'xi_hat': # a priori SNR estimate output (.mat).
				if not os.path.exists(args.out_path + '/xi_hat'): os.makedirs(args.out_path + '/xi_hat') # make output directory.
				spio.savemat(args.out_path + '/xi_hat/' + test_fnames[j], {'xi_hat':xi_hat_out})
			
			if args.out_type == 'gain': # gain function output (.mat).
				if not os.path.exists(args.out_path + '/gain/' + gain): os.makedirs(args.out_path + '/gain/' + args.gain) # make output directory.
				spio.savemat(args.out_path + '/gain/' + args.gain + '/' + test_fnames[j], {gain:gain_out})

			if args.out_type == 'y': # enahnced speech output (.wav).
				if not os.path.exists(args.out_path + '/y/' + args.gain): os.makedirs(args.out_path + '/y/' + args.gain) # make output directory.
				y_out = sess.run(y, feed_dict={G_ph: gain_out, x_PS_ph: x_PS_out, x_MS_2D_ph: x_MS_out, 
					output_ph: output_out}) # enhanced speech output.		
				scipy.io.wavfile.write(args.out_path + '/y/' + args.gain + '/' + test_fnames[j] + '.wav', args.fs, y_out)

			print("Inference (%s): %3.2f%%.       " % (args.out_type, 100*((j+1)/len(test_noisy_speech_len))), end="\r")
	print('\nInference complete.')

## ERROR FOR VALIDATION SET
if args.val:
	print("Computing error for validation set...")
	with tf.Session(config=config) as sess:
		saver.restore(sess, args.model_path + '/epoch-' + str(args.test_epoch)) # load model from epoch.
		start_idx = 0; end_idx = args.mbatch_size; val_flag = True; frames = 0; val_error = 0; # validation variables.
		while val_flag:
			val_mbatch = sess.run(feature, feed_dict={s_ph: val_clean_speech[start_idx:end_idx], d_ph: val_noise[start_idx:end_idx], 
					s_len_ph: val_clean_speech_len[start_idx:end_idx], d_len_ph: val_noise_len[start_idx:end_idx], snr_ph: val_snr[start_idx:end_idx]}) # mini-batch.
			val_error_frame = sess.run(loss, feed_dict={x_MS_ph: val_mbatch[0], target_ph: val_mbatch[1], 
				x_MS_len_ph: val_mbatch[2]}) # validation error for each frame.
			frames += val_error_frame.shape[0] # total number of frames.
			val_error += np.sum(val_error_frame)
			print("Validation error for Epoch %d: %3.2f%% complete.       " % 
				(args.test_epoch, 100*(end_idx/val_clean_speech_len.shape[0])), end="\r")
			start_idx += args.mbatch_size; end_idx += args.mbatch_size
			if end_idx > val_clean_speech_len.shape[0]: end_idx = val_clean_speech_len.shape[0]
			if start_idx >= val_clean_speech_len.shape[0]: val_flag = False
		val_error /= frames # validation error.
		print('Validation error for Epoch %d: %g.       ' % (args.test_epoch, val_error))