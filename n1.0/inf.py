# FILE:           inf.py
# DATE:           2018
# AUTHOR:         Aaron Nicolson
# AFFILIATION:    Signal Processing Laboratory, Griffith University
# BRIEF:          Inference for the ResLSTM a-priori SNR estimator, or 'DeepXi'.

import tensorflow as tf
from tensorflow.python.data import Dataset, Iterator
import numpy as np
from datetime import datetime
import scipy.io.wavfile
import scipy.io as spio
import feat, os, batch, res, argparse, random, math, time, sys, pickle
np.set_printoptions(threshold=np.nan)

## UPPER ENDPOINT SCALING FACTOR VALUES
# INT	ALPHA
#  5	0.5889
#  10   0.2944
#  15   0.1963
#  20   0.1472
#  25   0.1178
#  30   0.0981
#  35   0.0841
#  40   0.0736
#  45   0.0654
#  50   0.0589

def DeepXi(test_clean, test_noise, out_path, model_path, epoch, snr_list, gpu):

	print('DeepXi: ResLSTM a priori SNR estimator')

	## OPTIONS
	version = 'DeepXi_3c30' # model version.
	scaling = 0.0981 # scaling factor for SNR dB intertest size.
	print("Scaling factor: %g." % (scaling))
	print("%s on GPU:%s for epoch %d." % (version, gpu, epoch)) # print version.
	out_path = out_path + '/xi_hat'
	if not os.path.exists(out_path): os.makedirs(out_path) # make output directory.

	## NETWORK PARAMETERS
	cell_size = 512 # cell size of forward & backward cells.
	rnn_depth = 5 # number of RNN layers.
	bidirectional = False # use a Bidirectional Recurrent Neural Network.
	cell_proj = None # output size of the cell projection weight (None for no projection).
	residual = 'add' # residual connection either by addition ('add') or concatenation ('concat').
	res_proj = None # output size of the residual projection weight (None for no projection).
	peepholes = False # use peephole connections.
	input_layer = True # use an input layer.
	input_size = 512 # size of the input layer output.

	## FEATURES
	input_dim = 257 # number of inputs.
	num_outputs = input_dim # number of output dimensions.
	fs = 16000 # sampling frequency (Hz).
	Tw = 32 # window length (ms).
	Ts = 16 # window shift (ms).
	Nw = int(fs*Tw*0.001) # window length (samples).
	Ns = int(fs*Ts*0.001) # window shift (samples).
	NFFT = int(pow(2, np.ceil(np.log2(Nw)))) # number of DFT components.
	nconst = 32768 # normalisation constant (see feat.addnoisepad()).

	## GPU CONFIGURATION
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
	config = tf.ConfigProto()
	config.allow_soft_placement=True
	config.gpu_options.allow_growth=True
	config.log_device_placement=False

	## PLACEHOLDERS
	s_ph = tf.placeholder(tf.int16, shape=[None, None]) # clean speech placeholder.
	d_ph = tf.placeholder(tf.int16, shape=[None, None]) # noise placeholder.
	s_len_ph = tf.placeholder(tf.int32, shape=[None]) # clean speech sequence length placeholder.
	d_len_ph = tf.placeholder(tf.int32, shape=[None]) # noise sequence length placeholder.
	snr_ph = tf.placeholder(tf.float32, shape=[None]) # SNR placeholder.

	## TEST SET
	test_clean, test_clean_len, test_snr, test_fnames = batch._test_set(test_clean, '*.wav', snr_list) # clean test waveforms and lengths.
	test_noise, test_noise_len, _, _ = batch._test_set(test_noise, '*.wav', snr_list) # noise test waveforms and lengths.

	## LOG_10
	def log10(x):
	  numerator = tf.log(x)
	  denominator = tf.constant(np.log(10), dtype=numerator.dtype)
	  return tf.div(numerator, denominator)

	## FEATURE EXTRACTION FUNCTION
	def feat_extr(s, d, s_len, d_len, Q, Nw, Ns, NFFT, fs, P, nconst, scaling):
		'''
		Computes Magnitude Spectrum (MS) input features, and the a priori SNR target.
		The sequences are padded, with seq_len providing the length of each sequence
		without padding.

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
			scaling - scaling factor for SNR dB intertest size.

		Outputs:
			x_MS - padded noisy single-sided magnitude spectrum.
			xi_dB_mapped - mapped a priori SNR dB.	
			seq_len - length of each sequence without padding.
		'''
		(s, x, d) = tf.map_fn(lambda z: feat.addnoisepad(z[0], z[1], z[2], z[3], z[4],
			P, nconst), (s, d, s_len, d_len, Q), dtype=(tf.float32, tf.float32,
			tf.float32)) # padded noisy waveform, and padded clean waveform.
		seq_len = feat.nframes(s_len, Ns) # length of each sequence.
		s_MS = feat.stms(s, Nw, Ns, NFFT) # clean magnitude spectrum.
		d_MS = feat.stms(d, Nw, Ns, NFFT) # noise magnitude spectrum.
		x_MS = feat.stms(x, Nw, Ns, NFFT) # noisy magnitude spectrum.
		xi = tf.div(tf.square(s_MS), tf.add(tf.square(d_MS), 1e-12)) # a-priori-SNR.
		xi_dB = tf.multiply(10.0, tf.add(log10(xi), 1e-12)) # a-priori-SNR in dB.
		xi_db_mapped = tf.div(1.0, tf.add(1.0, tf.exp(tf.multiply(-scaling, xi_dB)))) # scaled a-priori-SNR.
		xi_db_mapped = tf.boolean_mask(xi_db_mapped, tf.sequence_mask(seq_len)) # convert to 2D.
		return (x_MS, xi_db_mapped, seq_len)

	## FEATURE GRAPH
	print('Preparing graph...')
	P = tf.reduce_max(s_len_ph) # padded waveform length.
	feature = feat_extr(s_ph, d_ph, s_len_ph, d_len_ph, snr_ph, Nw, Ns, NFFT, 
		fs, P, nconst, scaling) # feature graph.

	## RESNET
	parser = argparse.ArgumentParser()
	parser.add_argument('--cell_size', default=cell_size, type=int, help='BLSTM cell size.')
	parser.add_argument('--rnn_depth', default=rnn_depth, type=int, help='Number of RNN layers.')
	parser.add_argument('--bidirectional', default=bidirectional, type=bool, help='Use a Bidirectional Recurrent Neural Network.')
	parser.add_argument('--cell_proj', default=cell_proj, type=int, help='Output size of the cell projection matrix (None for no projection).')
	parser.add_argument('--residual', default=residual, type=str, help='Residual connection. Either addition or concatenation.')
	parser.add_argument('--peepholes', default=peepholes, type=bool, help='Use peephole connections.')
	parser.add_argument('--input_layer', default=input_layer, type=bool, help='Use an input layer.')
	parser.add_argument('--input_size', default=input_size, type=int, help='Input layer output size.')
	parser.add_argument('--verbose', default=True, type=bool, help='Print network.')
	parser.add_argument('--parallel_iterations', default=512, type=int, help='Number of parallel iterations.')
	args = parser.parse_args()
	y_hat = res.ResNet(feature[0], feature[2], num_outputs, args)

	## LOSS & OPTIMIZER
	loss = res.loss(feature[1], y_hat, 'sigmoid_xentropy')
	total_loss = tf.reduce_mean(loss)
	trainer, _ = res.optimizer(loss, optimizer='adam')

	## SYNTHESIS PLACEHOLDERS
	xi_db_mapped_hat_ph = tf.placeholder(tf.float32, shape=[None, None]) # mapped a priori SNR dB placeholder.
	
	## A PRIORI SNR ESTIMATE GRAPH
	xi_db_mapped_hat = tf.sigmoid(xi_db_mapped_hat_ph) # mapped a priori SNR dB estimate.
	xi_dB_hat = tf.subtract(tf.div(1.0, xi_db_mapped_hat), 1.0) # a priori SNR dB estimate.
	xi_dB_hat = tf.negative(tf.div(tf.log(tf.maximum(xi_dB_hat, 1e-12)), scaling)) # a priori SNR dB estimate.
	xi_hat = tf.pow(10.0, tf.div(xi_dB_hat, 10.0)) # a priori SNR estimate.
	
	## SAVE VARIABLES
	saver = tf.train.Saver()

	## INFERENCE
	with tf.Session(config=config) as sess:

		## LOAD MODEL
		saver.restore(sess, model_path + '/' + version + '/epoch-' + str(epoch)) # load model from epoch.

		for snr in snr_list:
			for j in range(len(test_clean_len)):

				xi_db_mapped_hat_out = sess.run(y_hat, feed_dict={s_ph: [test_clean[j]], d_ph:
					[test_noise[j]], s_len_ph: [test_clean_len[j]], d_len_ph:
					[test_noise_len[j]], snr_ph: [snr]}) # mapped a priori SNR dB estimate.
				
				xi_hat_out = sess.run(xi_hat, feed_dict={xi_db_mapped_hat_ph: xi_db_mapped_hat_out}) # a priori SNR output.
				spio.savemat(out_path + '/' + test_fnames[j] + '_' + str(snr) + 'dB', {'xi_hat':xi_hat_out})

				print("Inference: %3.2f%% complete for %g dB.       " % (100*(j/len(test_clean_len)), snr), end="\r")
	print('Inference complete.')
