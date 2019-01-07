# FILE:           deepxi.py
# DATE:           2018
# AUTHOR:         Aaron Nicolson
# AFFILIATION:    Signal Processing Laboratory, Griffith University
# BRIEF:          'DeepXi' training and testing.

import tensorflow as tf
from tensorflow.python.data import Dataset, Iterator
import numpy as np
from datetime import datetime
from scipy.io.wavfile import read
import scipy.io.wavfile
import scipy.io as spio
import scipy.special as spsp
import os, argparse, random, math, time, sys, pickle
from lib import feat, batch, residual
np.set_printoptions(threshold=np.nan)
from os.path import expanduser

print('DeepXi a priori SNR estimator')

## OPTIONS (GENERAL)
version = 'n1.5a' # model version.
gpu = "0" # select GPU.
par_iter = 512 # dynamic_rnn/bidirectional_dynamic_rnn parallel iterations.
print("Version: %s on GPU:%s." % (version, gpu)) # print version.
HOME = expanduser("~")

## OPTIONS (TRAIN)
train = False # perform training flag.
cont = False # continue testing from last epoch.
train_clean_speech_ver = 'v1' # train clean speech set version.
train_noise_ver = 'v1' # train noise set version.
model_path = HOME + '/model/' + version # model save path.
mbatch_size = 10 # mini-batch size.
max_epochs = 10 # maximum number of epochs.
train_clean_speech_path = '' # path to the clean speech training set.
train_noise_path = '' # path to the clean speech training set.
val_clean_speech_path = '' # path to the clean speech validation set.
val_noise_path = '' # path to the noise validation set.

## OPTIONS (TEST)
test = True # perform test flag.
val = False # find validation error for "test_epoch".
test_epoch = 10 # epoch for testing.
test_noisy_speech_path = HOME + '/github/DeepXi/noisy_speech'
out_path = HOME + '/github/DeepXi/enhanced_speech'

## TEST OUTPUT TYPE
# 'raw' - output from network (.mat).
# 'xi_hat' - a priori SNR estimate (.mat).
# 'gain' - gain function (.mat).
# 'y' - enhanced speech (.wav).
out_type = 'y' 

## GAIN FUNCTION
# 'ibm' - Ideal Binary Mask (IBM).
# 'wf' - Wiener Filter (WF).
# 'srwf' - Square-Root Wiener Filter (SRWF).
# 'mmse-stsa' - Minimum-Mean Square Error - Short-Time Spectral Amplitude (MMSE-STSA) estimator.
# 'mmse-lsa' - Minimum-Mean Square Error - Log-Spectral Amplitude (MMSE-LSA) estimator.
gain = 'mmse-lsa'

## NETWORK PARAMETERS
blocks = ['C3'] + ['B3']*5 + ['O1'] # residual blocks.
cell_size = 512 # cell size.
cell_proj = None # output size of the cell projection weight (None for no projection).
cell_type = 'LSTMCell' # RNN cell type.
bidi = True # use a Bidirectional Recurrent Neural Network.
bidi_con = 'add' # forward and backward cell activation connection.
layer_norm = None # layer normalisation in LSTM block.
res_con = 'add' # residual connection either by addition ('add') or concatenation ('concat').
res_proj = None # output size of the residual projection weight (None for no projection).
peep = None # use peephole connections.
block_unit = None # residual unit.
coup_unit = 'CU1' # coupling unit.
conv_size = None # convolution kernel size.
conv_filt = None # number of convolution filters.
conv_caus = None # causal convolution.
dropout = None # use dropout training.
keep_prob = None # keep probability during training (0.75 typically).
context = None # input context.

## FEATURES
min_snr = -10 # minimum trained SNR level.
max_snr = 20 # maximum trained SNR level.
train_snr_list = list(range(min_snr, max_snr + 1)) # list of SNR levels from min_snr to max_snr with an increment of 1.
input_dim = 257 # number of inputs.
num_outputs = input_dim # number of output dimensions.
fs = 16000 # sampling frequency (Hz).
Tw = 32 # window length (ms).
Ts = 16 # window shift (ms).
Nw = int(fs*Tw*0.001) # window length (samples).
Ns = int(fs*Ts*0.001) # window shift (samples).
NFFT = int(pow(2, np.ceil(np.log2(Nw)))) # number of DFT components.
nconst = 32768 # normalisation constant (see feat.addnoisepad()).

## A PRIORI SNR IN DB STATS
mu_mat = spio.loadmat(HOME + '/github/DeepXi/stats/mu.mat') # mean of a priori SNR in dB from MATLAB.
mu = tf.constant(mu_mat['mu'], dtype=tf.float32) 
sigma_mat = spio.loadmat(HOME + '/github/DeepXi/stats/sigma.mat') # standard deviation of a priori SNR in dB from MATLAB.
sigma = tf.constant(sigma_mat['sigma'], dtype=tf.float32) 

## DATASETS
if train:
	## TRAINING CLEAN SPEECH AND NOISE SET
	train_clean_speech_list = batch._train_list(train_clean_speech_path, '*.wav', 'clean') # clean speech training list.
	train_noise_list = batch._train_list(train_noise_path, '*.wav', 'noise') # noise training list.
	if not os.path.exists(model_path): os.makedirs(model_path) # make model path directory.

if train or val:
	## VALIDATION CLEAN SPEECH AND NOISE SET
	val_clean_speech, val_clean_speech_len, val_snr, _ = batch._batch(val_clean_speech_path, '*.wav', train_snr_list) # clean validation waveforms and lengths.
	val_noise, val_noise_len, _, _ = batch._batch(val_noise_path, '*.wav', train_snr_list) # noise validation waveforms and lengths.

## TEST NOISY SPEECH SET
if test: test_noisy_speech, test_noisy_speech_len, test_snr, test_fnames = batch._batch(test_noisy_speech_path, '*.wav', []) # noisy speech test waveforms and lengths.

## GPU CONFIGURATION
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=gpu
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
x_MS_ph = tf.placeholder(tf.float32, shape=[None, None, input_dim], name='x_MS_ph') # noisy speech MS placeholder.
x_MS_len_ph = tf.placeholder(tf.int32, shape=[None], name='x_MS_len_ph') # noisy speech MS sequence length placeholder.
target_ph = tf.placeholder(tf.float32, shape=[None, input_dim], name='target_phh') # training target placeholder.
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
		phi_xi_dB - CDF of a priori SNR dB.	
		seq_len - length of each sequence without padding.
	'''
	(s, x, d) = tf.map_fn(lambda z: feat.addnoisepad(z[0], z[1], z[2], z[3], z[4],
		P, nconst), (s, d, s_len, d_len, Q), dtype=(tf.float32, tf.float32,
		tf.float32)) # padded waveforms.
	seq_len = feat.nframes(s_len, Ns) # length of each sequence.
	s_MS = feat.stms(s, Nw, Ns, NFFT) # clean speech magnitude spectrum.
	d_MS = feat.stms(d, Nw, Ns, NFFT) # noise magnitude spectrum.
	x_MS = feat.stms(x, Nw, Ns, NFFT) # noisy speech magnitude spectrum.
	xi = tf.div(tf.square(s_MS), tf.add(tf.square(d_MS), 1e-12)) # a priori SNR.
	xi_dB = tf.multiply(10.0, tf.add(log10(xi), 1e-12)) # a priori SNR dB.
	phi_xi_dB = tf.multiply(0.5, tf.add(1.0, tf.erf(tf.div(tf.subtract(xi_dB, mu), 
		tf.multiply(sigma, tf.sqrt(2.0)))))) # cdf of a priori SNR in dB.
	phi_xi_dB = tf.boolean_mask(phi_xi_dB, tf.sequence_mask(seq_len)) # convert to 2D.
	return (x_MS, phi_xi_dB, seq_len)

## FEATURE GRAPH
print('Preparing graph...')
P = tf.reduce_max(s_len_ph) # padded waveform length.
feature = feat_extr(s_ph, d_ph, s_len_ph, d_len_ph, snr_ph, Nw, Ns, NFFT, 
	fs, P, nconst, mu, sigma) # feature graph.

## RESNET
parser = argparse.ArgumentParser()
parser.add_argument('--blocks', default=blocks, type=list)
parser.add_argument('--input_dim', default=input_dim, type=int)
parser.add_argument('--cell_size', default=cell_size, type=int)
parser.add_argument('--cell_proj', default=cell_proj, type=int)
parser.add_argument('--cell_type', default=cell_type, type=str)
parser.add_argument('--peep', default=peep, type=bool)
parser.add_argument('--bidi', default=bidi, type=bool)
parser.add_argument('--bidi_con', default=bidi_con, type=str)
parser.add_argument('--layer_norm', default=layer_norm, type=bool)
parser.add_argument('--block_unit', default=block_unit, type=str)
parser.add_argument('--coup_unit', default=coup_unit, type=str)
parser.add_argument('--conv_size', default=conv_size, type=int)
parser.add_argument('--conv_filt', default=conv_filt, type=int)
parser.add_argument('--conv_caus', default=conv_caus, type=int)
parser.add_argument('--res_con', default=res_con, type=str)
parser.add_argument('--res_proj', default=res_proj, type=int)
parser.add_argument('--dropout', default=dropout, type=bool)
parser.add_argument('--context', default=context, type=int)
parser.add_argument('--verbose', default=True, type=bool)
parser.add_argument('--par_iter', default=par_iter, type=int)
args = parser.parse_args()

output = residual.Residual(x_MS_ph, x_MS_len_ph, keep_prob_ph, 
	training_ph, num_outputs, args)

## LOSS & OPTIMIZER
loss = residual.loss(target_ph, output, 'sigmoid_cross_entropy')
total_loss = tf.reduce_mean(loss, axis=0)
trainer, _ = residual.optimizer(total_loss, optimizer='adam')

## SAVE VARIABLES
saver = tf.train.Saver(max_to_keep=256)

## NUMBER OF PARAMETERS
print("No. of trainable parameters: %g." % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

## TRAINING
if train:
	print("Training...")
	with tf.Session(config=config) as sess:

		## CONTINUE FROM LAST EPOCH
		if cont:
			with open('data/epoch_par_' + version + '.p', 'rb') as f:
				epoch_par = pickle.load(f) # load epoch parameters from last epoch.
			epoch_par['start_idx'] = 0; epoch_par['end_idx'] = mbatch_size # reset start and end index of mini-batch. 
			random.shuffle(train_clean_speech_list) # shuffle list.
			with open('data/epoch_par_' + version + '.p', 'wb') as f:
				pickle.dump(epoch_par, f) # save epoch parameters.
			saver.restore(sess, model_path + '/epoch-' + str(epoch_par['epoch_comp'])) # load model from last epoch.

		## TRAIN RAW NETWORK
		else:
			if os.path.isfile('data/epoch_par_' + version + '.p'): os.remove('data/epoch_par_' + version + '.p') # remove epoch parameters.
			print('Creating epoch parameters, as no pickle file exists...')
			epoch_par = {'epoch_size': len(train_clean_speech_list), 'epoch_comp': 0, 
				'start_idx': 0, 'end_idx': mbatch_size, 'val_error_prev': float("inf")} # create epoch parameters.
			if mbatch_size > epoch_par['epoch_size']: raise ValueError('Error: mini-batch size is greater than the epoch size.')
			with open('data/epoch_par_' + version + '.p', 'wb') as f:
				pickle.dump(epoch_par, f) # save epoch parameters.
			sess.run(tf.global_variables_initializer()) # initialise model variables.
			saver.save(sess, model_path + '/epoch', global_step=epoch_par['epoch_comp']) # save model.

		## TRAINING LOG
		if not os.path.exists('log'): os.makedirs('log') # create log directory.
		with open("log/val_error_" + version + ".txt", "a") as results:
			results.write("Validation error, epoch count, D/T:\n")

		while train:
			## MINI-BATCH GENERATION
			mbatch_size_iter = epoch_par['end_idx'] - epoch_par['start_idx'] # number of examples in mini-batch for the training iteration.
			train_clean_speech_mbatch, train_clean_speech_mbatch_seq_len = batch._clean_mbatch(train_clean_speech_list, 
				mbatch_size_iter, epoch_par['start_idx'], epoch_par['end_idx']) # generate mini-batch of clean training waveforms.
			train_noise_mbatch, train_noise_mbatch_seq_len = batch._noise_mbatch(train_noise_list, 
				mbatch_size_iter, train_clean_speech_mbatch_seq_len) # generate mini-batch of noise training waveforms.
			train_snr_mbatch = np.random.randint(min_snr, max_snr + 1, epoch_par['end_idx'] - epoch_par['start_idx']) # generate mini-batch of SNR levels.

			## TRAINING ITERATION
			train_mbatch = sess.run(feature, feed_dict={s_ph: train_clean_speech_mbatch, d_ph: train_noise_mbatch, 
				s_len_ph: train_clean_speech_mbatch_seq_len, d_len_ph: train_noise_mbatch_seq_len, snr_ph: train_snr_mbatch}) # mini-batch.
			sess.run(trainer, feed_dict={x_MS_ph: train_mbatch[0], 
				target_ph: train_mbatch[1], x_MS_len_ph: train_mbatch[2]}) # training iteration.

			print("E%d: %3.1f%% (E%d error: %g). %s on GPU:%s.       " % 
				(epoch_par['epoch_comp'] + 1, 100*(epoch_par['end_idx']/epoch_par['epoch_size']), 
				epoch_par['epoch_comp'], epoch_par['val_error_prev'], version, gpu), end="\r")

			## UPDATE EPOCH PARAMETERS
			epoch_par['start_idx'] += mbatch_size; epoch_par['end_idx'] += mbatch_size # start and end index of mini-batch.

			## VALIDATION SET CROSS-ENTROPY
			if epoch_par['start_idx'] >= epoch_par['epoch_size']:
				epoch_par['start_idx'] = 0; epoch_par['end_idx'] = mbatch_size # reset start and end index of mini-batch.
				random.shuffle(train_clean_speech_list) # shuffle list.
				start_idx = 0; end_idx = mbatch_size; val_flag = True; frames = 0; val_error = 0; # validation variables.
				while val_flag:
					val_mbatch = sess.run(feature, feed_dict={s_ph: val_clean_speech[start_idx:end_idx], d_ph: val_noise[start_idx:end_idx], 
						s_len_ph: val_clean_speech_len[start_idx:end_idx], d_len_ph: val_noise_len[start_idx:end_idx], snr_ph: val_snr[start_idx:end_idx]}) # mini-batch.
					val_error_mbatch = sess.run(loss, feed_dict={x_MS_ph: val_mbatch[0], 
						target_ph: val_mbatch[1], x_MS_len_ph: val_mbatch[2]}) # validation cross-entropy for each frame in mini-batch.
					frames += val_error_mbatch.shape[0] # total number of frames.
					val_error += np.sum(val_error_mbatch)
					print("Validation error for Epoch %d: %3.2f%% complete.       " % 
						(epoch_par['epoch_comp'] + 1, 100*(end_idx/val_clean_speech_len.shape[0])), end="\r")
					start_idx += mbatch_size; end_idx += mbatch_size
					if start_idx >= val_clean_speech_len.shape[0]: val_flag = False
					elif end_idx > val_clean_speech_len.shape[0]: end_idx = val_clean_speech_len.shape[0]
				val_error /= frames # validation cross-entropy.
				if val_error < epoch_par['val_error_prev']:
					epoch_par['val_error_prev'] = val_error # lowest validation error achieved.
					epoch_par['epoch_comp'] += 1 # an epoch has been completed.
					saver.save(sess, model_path + '/epoch', global_step=epoch_par['epoch_comp']) # save model.
					with open("log/val_error_" + version + ".txt", "a") as results:
						results.write("%g, %d, %s.\n" % (val_error, epoch_par['epoch_comp'], datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
					epoch_par['val_flag'] = False # reset validation flag.
					with open('data/epoch_par_' + version + '.p', 'wb') as f:
						pickle.dump(epoch_par, f)
					if epoch_par['epoch_comp'] >= max_epochs:
						train = False
						print('\nTraining complete. Validation error for epoch %d: %g.                 ' % 
							(epoch_par['epoch_comp'], val_error))
				else: # exploding gradient.
					saver.restore(sess, model_path + '/epoch-' + str(epoch_par['epoch_comp'])) # load model from last epoch.
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

if test:
	print("Test...")

	## TEST PLACEHOLDERS
	output_ph = tf.placeholder(tf.float32, shape=[None, input_dim], name='output_ph') # network output placeholder.
	x_MS_2D_ph = tf.placeholder(tf.float32, shape=[None, input_dim], name='x_MS_2D_ph') # noisy speech MS placeholder (in 2D form).
	x_PS_ph = tf.placeholder(tf.float32, shape=[None, input_dim], name='x_PS_ph') # noisy speech PS placeholder.
	xi_hat_ph = tf.placeholder(tf.float32, shape=[None, input_dim], name='xi_hat_ph') # a priori SNR estimate placeholder.
	G_ph = tf.placeholder(tf.float32, shape=[None, input_dim], name='G_ph') # gain function placeholder.

	## ANALYSIS
	x = tf.div(tf.cast(tf.slice(tf.squeeze(x_ph), [0], [tf.squeeze(x_len_ph)]), tf.float32), nconst) # remove padding and normalise.
	x_DFT = feat.stft(x, Nw, Ns, NFFT) # noisy speech single-sided short-time Fourier transform.
	x_MS_3D = tf.expand_dims(tf.abs(x_DFT), 0) # noisy speech single-sided magnitude spectrum (in 3D form).
	x_MS = tf.abs(x_DFT) # noisy speech single-sided magnitude spectrum.
	x_PS = tf.angle(x_DFT) # noisy speech single-sided phase spectrum.
	x_seq_len = feat.nframes(x_len_ph, Ns) # length of each sequence.

	## ENHANCEMENT
	if gain is 'ibm': G = tf.cast(tf.greater(xi_hat_ph, 1), tf.float32) # IBM gain function.
	if gain is 'wf': G = tf.div(xi_hat_ph, tf.add(xi_hat_ph, 1.0)) # WF gain function.
	if gain is 'srwf': G = tf.sqrt(tf.div(xi_hat_ph, tf.add(xi_hat_ph, 1.0))) # SRWF gain function.
	s_hat_MS = tf.multiply(x_MS_2D_ph, G_ph) # enhanced speech single-sided magnitude spectrum.
	
	## SYNTHESIS GRAPH
	y_DFT = tf.cast(s_hat_MS, tf.complex64) * tf.exp(1j * tf.cast(x_PS_ph, tf.complex64)) # enhanced speech single-sided short-time Fourier transform.
	
	y = tf.contrib.signal.inverse_stft(y_DFT, Nw, Ns, NFFT, 
		tf.contrib.signal.inverse_stft_window_fn(Ns, 
		forward_window_fn=tf.contrib.signal.hamming_window)) # synthesis.

	## INFERENCE
	with tf.Session(config=config) as sess:

		## LOAD MODEL
		saver.restore(sess, model_path + '/epoch-' + str(test_epoch)) # load model from epoch.

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

			output_out = sess.run(output, feed_dict={x_MS_ph: x_MS_3D_out, x_MS_len_ph: x_seq_len_out}) # output of network.
			output_out = np_sigmoid(output_out)
			xi_dB_hat_out = np.add(np.multiply(np.multiply(sigma_np, np.sqrt(2.0)), spsp.erfinv(np.subtract(np.multiply(2.0, output_out), 1))), mu_np); # a priori SNR estimate.			
			xi_hat_out = np.power(10.0, np.divide(xi_dB_hat_out, 10.0))
			
			if gain is 'mmse-stsa': gain_out = mmse_stsa(xi_hat_out, ml_gamma_hat(xi_hat_out)) # MMSE-STSA estimator gain.
			elif gain is 'mmse-lsa': gain_out = mmse_lsa(xi_hat_out, ml_gamma_hat(xi_hat_out)) # MMSE-LSA estimator gain.
			else: gain_out = sess.run(G, feed_dict={xi_hat_ph: xi_hat_out}) # gain.	

			if out_type is 'raw': # raw outputs from network (.mat).
				if not os.path.exists(out_path + '/raw'): os.makedirs(out_path + '/raw') # make output directory.
				spio.savemat(out_path + '/raw/' + test_fnames[j], {'raw':output_out})

			if out_type is 'xi_hat': # a priori SNR estimate output (.mat).
				if not os.path.exists(out_path + '/xi_hat'): os.makedirs(out_path + '/xi_hat') # make output directory.
				spio.savemat(out_path + '/xi_hat/' + test_fnames[j], {'xi_hat':xi_hat_out})
			
			if out_type is 'gain': # gain function output (.mat).
				if not os.path.exists(out_path + '/gain/' + gain): os.makedirs(out_path + '/gain/' + gain) # make output directory.
				spio.savemat(out_path + '/gain/' + gain + '/' + test_fnames[j], {gain:gain_out})

			if out_type is 'y': # enahnced speech output (.wav).
				if not os.path.exists(out_path + '/y/' + gain): os.makedirs(out_path + '/y/' + gain) # make output directory.
				y_out = sess.run(y, feed_dict={G_ph: gain_out, x_PS_ph: x_PS_out, x_MS_2D_ph: x_MS_out, 
					output_ph: output_out}) # enhanced speech output.		
				scipy.io.wavfile.write(out_path + '/y/' + gain + '/' + test_fnames[j] + '.wav', fs, y_out)

			print("Inference (%s): %3.2f%%.       " % (out_type, 100*((j+1)/len(test_noisy_speech_len))), end="\r")
	print('\nInference complete.')

## ERROR FOR VALIDATION SET
if val:
	print("Computing error for validation set...")
	with tf.Session(config=config) as sess:
		saver.restore(sess, model_path + '/epoch-' + str(test_epoch)) # load model from epoch.
		start_idx = 0; end_idx = mbatch_size; val_flag = True; frames = 0; val_error = 0; # validation variables.
		while val_flag:
			val_mbatch = sess.run(feature, feed_dict={s_ph: val_clean_speech[start_idx:end_idx], d_ph: val_noise[start_idx:end_idx], 
					s_len_ph: val_clean_speech_len[start_idx:end_idx], d_len_ph: val_noise_len[start_idx:end_idx], snr_ph: val_snr[start_idx:end_idx]}) # mini-batch.
			val_error_frame = sess.run(loss, feed_dict={x_MS_ph: val_mbatch[0],
				target_ph: val_mbatch[1], x_MS_len_ph: val_mbatch[2]}) # validation cross-entropy for each frame.
			frames += val_error_frame.shape[0] # total number of frames.
			val_error += np.sum(val_error_frame)
			print("Validation error for Epoch %d: %3.2f%% complete.       " % 
				(test_epoch, 100*(end_idx/val_clean_speech_len.shape[0])), end="\r")
			start_idx += mbatch_size; end_idx += mbatch_size
			if end_idx > val_clean_speech_len.shape[0]: end_idx = val_clean_speech_len.shape[0]
			if start_idx >= val_clean_speech_len.shape[0]: val_flag = False
		val_error /= frames # validation cross-entropy.
		print('\nValidation error for Epoch %d: %g.       ' % (test_epoch, val_error))