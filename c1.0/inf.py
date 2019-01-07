# FILE:           inf.py
# DATE:           2018
# AUTHOR:         Aaron Nicolson
# AFFILIATION:    Signal Processing Laboratory, Griffith University
# BRIEF:          Inference for the causal ResLSTM a-priori SNR estimator, 'DeepXi'.

import tensorflow as tf
from tensorflow.python.data import Dataset, Iterator
import numpy as np
from datetime import datetime
import scipy.io.wavfile
import scipy.io as spio
import feat, os, batch, residual_legacy, argparse, random, math, time, sys, pickle
np.set_printoptions(threshold=np.nan)

## UPPER ENDPOINT SCALING FACTOR VALUES
# INT	RHO
#  10   0.2944
#  20   0.1472
#  30   0.0981
#  40   0.0736

def DeepXi(test_noisy, out_path, model_path, epoch, gpu, opt):

	print('DeepXi: ResLSTM a priori SNR estimator')

	## OPTIONS
	version = 'c1.0' # model version.
	scaling = 0.0981 # scaling factor for SNR dB intertest size.
	print("Scaling factor: %g." % (scaling))
	print("%s on GPU:%s for epoch %d." % (version, gpu, epoch)) # print version.
	if opt is 'xi_hat': out_path = out_path + '/xi_hat'
	if opt is 'y': 	out_path = out_path + '/y'
	par_iter = 256 # dynamic_rnn/bidirectional_dynamic_rnn parallel iterations.
	if not os.path.exists(out_path): os.makedirs(out_path) # make output directory.

	## NETWORK PARAMETERS
	blocks = ['I2'] + ['B3']*5 + ['O'] # residual blocks. e.g.: ['I2'] + ['B3']*5 + ['O'].
	cell_size = 512 # cell size of forward & backward cells.
	cell_proj = None # output size of the cell projection weight (None for no projection).
	cell_type = 'LSTMCell' # RNN cell type.
	bidi = False # use a Bidirectional Recurrent Neural Network.
	layer_norm = False # layer normalisation in LSTM block.
	res_con = 'add' # residual connection either by addition ('add') or concatenation ('concat').
	res_proj = None # output size of the residual projection weight (None for no projection).
	peep = False # use peephole connections.
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
	x_ph = tf.placeholder(tf.int16, shape=[None, None]) # noisy speech placeholder.
	s_len_ph = tf.placeholder(tf.int32, shape=[None]) # clean speech sequence length placeholder.
	d_len_ph = tf.placeholder(tf.int32, shape=[None]) # noise sequence length placeholder.
	x_len_ph = tf.placeholder(tf.int32, shape=[None]) # noisy speech sequence length placeholder.
	snr_ph = tf.placeholder(tf.float32, shape=[None]) # SNR placeholder.
	x_MS_ph = tf.placeholder(tf.float32, shape=[None, None, input_dim]) # noisy speech MS placeholder.
	x_MS_2D_ph = tf.placeholder(tf.float32, shape=[None, input_dim]) # noisy speech MS placeholder.
	x_PS_ph = tf.placeholder(tf.float32, shape=[None, input_dim]) # noisy speech PS placeholder.
	x_MS_len_ph = tf.placeholder(tf.int32, shape=[None]) # noisy speech MS sequence length placeholder.
	mapped_local_xi_dB_ph = tf.placeholder(tf.float32, shape=[None, input_dim]) # mapped local a priori SNR dB placeholder.

	## TEST SET
	test_noisy, test_noisy_len, test_snr, test_fnames = batch._test_set(test_noisy, '*.wav', [0]) # noisy speech test waveforms and lengths.

	## LOG_10
	def log10(x):
	  numerator = tf.log(x)
	  denominator = tf.constant(np.log(10), dtype=numerator.dtype)
	  return tf.div(numerator, denominator)

	## FEATURE EXTRACTION FUNCTION
	def feat_extr(s, d, s_len, d_len, Q, Nw, Ns, NFFT, fs, P, nconst, scaling):
		'''
		Computes Magnitude Spectrum (MS) input features, and the local a priori SNR 
		dB target. The sequences are padded, with seq_len providing the length of 
		each sequence without padding.

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
			scaling - scaling factor for SNR dB interval size.

		Outputs:
			s_MS - padded noisy single-sided magnitude spectrum.
			mapped_local_xi_dB- mapped local a priori SNR dB.	
			seq_len - length of each sequence without padding.
		'''
		(s, x, d) = tf.map_fn(lambda z: feat.addnoisepad(z[0], z[1], z[2], z[3], z[4],
			P, nconst), (s, d, s_len, d_len, Q), dtype=(tf.float32, tf.float32,
			tf.float32)) # padded noisy waveform, and padded clean waveform.
		seq_len = feat.nframes(s_len, Ns) # length of each sequence.
		s_MS = feat.stms(s, Nw, Ns, NFFT) # clean speech magnitude spectrum.
		d_MS = feat.stms(d, Nw, Ns, NFFT) # noise magnitude spectrum.
		x_MS = feat.stms(x, Nw, Ns, NFFT) # noisy speech magnitude spectrum.
		x_PS = feat.stps(x, Nw, Ns, NFFT) # noisy speech phase spectrum.
		local_xi = tf.div(tf.square(s_MS), tf.add(tf.square(d_MS), 1e-12)) # local a priori SNR.
		local_xi_dB = tf.multiply(10.0, tf.add(log10(local_xi), 1e-12)) # local a priori SNR dB.
		mapped_local_xi_dB = tf.div(1.0, tf.add(1.0, tf.exp(tf.multiply(-scaling, local_xi_dB)))) # mapped local a priori SNR dB.
		mapped_local_xi_dB = tf.boolean_mask(mapped_local_xi_dB, tf.sequence_mask(seq_len)) # convert to 2D.
		return (x_MS, mapped_local_xi_dB, seq_len, x_PS)

	## FEATURE GRAPH
	print('Preparing graph...')
	P = tf.reduce_max(s_len_ph) # padded waveform length.
	feature = feat_extr(s_ph, d_ph, s_len_ph, d_len_ph, snr_ph, Nw, Ns, NFFT, 
		fs, P, nconst, scaling) # feature graph.

	## RESNET
	parser = argparse.ArgumentParser()
	parser.add_argument('--blocks', default=blocks, type=list, help='Residual blocks.')
	parser.add_argument('--cell_size', default=cell_size, type=int, help='BLSTM cell size.')
	parser.add_argument('--cell_proj', default=cell_proj, type=int, help='Output size of the cell projection matrix (None for no projection).')
	parser.add_argument('--cell_type', default=cell_type, type=str, help='RNN cell type.')
	parser.add_argument('--peep', default=peep, type=bool, help='Use peephole connections.')
	parser.add_argument('--bidi', default=bidi, type=bool, help='Use a Bidirectional Recurrent Neural Network.')
	parser.add_argument('--layer_norm', default=layer_norm, type=bool, help='Layer normalisation in LSTM block.')
	parser.add_argument('--res_con', default=res_con, type=str, help='Residual connection. Either addition or concatenation.')
	parser.add_argument('--res_proj', default=res_proj, type=int, help='Residual projection size (None for no projection).')
	parser.add_argument('--input_size', default=input_size, type=int, help='Input layer output size.')
	parser.add_argument('--conv_caus', default=True, type=bool, help='Causal convolution.')
	parser.add_argument('--verbose', default=True, type=bool, help='Print network.')
	parser.add_argument('--par_iter', default=par_iter, type=int, help='Number of parallel iterations.')
	args = parser.parse_args()
	y_hat = residual_legacy.Residual(x_MS_ph, x_MS_len_ph, None, num_outputs, args)

	## LOSS & OPTIMIZER
	loss = residual_legacy.loss(mapped_local_xi_dB_ph, y_hat, 'sigmoid_xentropy')
	total_loss = tf.reduce_mean(loss)
	trainer, _ = residual_legacy.optimizer(loss, optimizer='adam')
	
	## A PRIORI SNR ESTIMATE GRAPH
	xi_db_mapped_hat = tf.sigmoid(mapped_local_xi_dB_ph) # mapped a priori SNR dB estimate.
	xi_dB_hat = tf.subtract(tf.div(1.0, xi_db_mapped_hat), 1.0) # a priori SNR dB estimate.
	xi_dB_hat = tf.negative(tf.div(tf.log(tf.maximum(xi_dB_hat, 1e-12)), scaling)) # a priori SNR dB estimate.
	xi_hat = tf.pow(10.0, tf.div(xi_dB_hat, 10.0)) # a priori SNR estimate.
	
	## ANALYSIS
	x = tf.div(tf.cast(tf.slice(tf.squeeze(x_ph), [0], [tf.squeeze(x_len_ph)]), tf.float32), nconst) # remove padding and normalise.
	x_DFT = feat.stft(x, Nw, Ns, NFFT) # noisy speech single-sided short-time Fourier transform.
	x_MS_3D = tf.expand_dims(tf.abs(x_DFT), 0) # noisy speech single-sided magnitude spectrum.
	x_MS = tf.abs(x_DFT) # noisy speech single-sided magnitude spectrum.
	x_PS = tf.angle(x_DFT) # noisy speech single-sided phase spectrum.
	x_seq_len = feat.nframes(x_len_ph, Ns) # length of each sequence.

	## ENHANCEMENT
	G = tf.div(xi_hat, tf.add(xi_hat, 1.0)) # WF gain function.
	s_hat_MS = tf.multiply(x_MS_2D_ph, G) # enhanced speech single-sided magnitude spectrum.
	
	## SYNTHESIS GRAPH
	y_DFT = tf.cast(s_hat_MS, tf.complex64) * tf.exp(1j * tf.cast(x_PS_ph, tf.complex64)) # enhanced speech single-sided short-time Fourier transform.
	y = tf.contrib.signal.inverse_stft(y_DFT, Nw, Ns, NFFT, 
		tf.contrib.signal.inverse_stft_window_fn(Ns, 
		forward_window_fn=tf.contrib.signal.hamming_window)) # synthesis.

	## SAVE VARIABLES
	saver = tf.train.Saver()

	## INFERENCE
	with tf.Session(config=config) as sess:

		## LOAD MODEL
		saver.restore(sess, model_path + '/' + version + '/epoch-' + str(epoch)) # load model from epoch.
		for j in range(len(test_noisy_len)):
			x_MS_out = sess.run(x_MS, feed_dict={x_ph: [test_noisy[j]], 
				x_len_ph: [test_noisy_len[j]]}) 
			x_MS_3D_out = sess.run(x_MS_3D, feed_dict={x_ph: [test_noisy[j]], 
				x_len_ph: [test_noisy_len[j]]}) 
			x_PS_out = sess.run(x_PS, feed_dict={x_ph: [test_noisy[j]], 
				x_len_ph: [test_noisy_len[j]]}) 
			x_seq_len_out = sess.run(x_seq_len, feed_dict={x_len_ph: [test_noisy_len[j]]}) 

			mapped_local_xi_dB_out = sess.run(y_hat, feed_dict={x_MS_ph: x_MS_3D_out, x_MS_len_ph: x_seq_len_out}) # mapped a priori SNR dB estimate.
				
			if opt is 'xi_hat':
				xi_hat_out = sess.run(xi_hat, feed_dict={mapped_local_xi_dB_ph: mapped_local_xi_dB_out}) # a priori SNR output.
				spio.savemat(out_path + '/' + test_fnames[j], {'xi_hat':xi_hat_out})

			if opt is 'y':
				y_out = sess.run(y, feed_dict={x_MS_2D_ph: x_MS_out, x_PS_ph: x_PS_out, 
					x_MS_len_ph: x_seq_len_out, mapped_local_xi_dB_ph: mapped_local_xi_dB_out}) # enhanced speech output.		
				scipy.io.wavfile.write(out_path + '/' + test_fnames[j] + '.wav', fs, y_out)
			print("Inference: %3.2f%% complete.       " % (100*(j/len(test_noisy_len))), end="\r")
	print('Inference complete.')
