## FILE:           utils.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          General utility functions/modules.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from os.path import expanduser
import argparse, os, string
import numpy as np
import tensorflow as tf

## CHARACTER DICTIONARIES
def char_dict():
	chars = list(string.ascii_lowercase + " " + ">" + "%") # 26 alphabetic characters + space + EOS + blank = 29 classes.
	char2idx = dict(zip(chars, [i for i in range(29)])) 
	idx2char = dict((y,x) for x,y in char2idx.items())
	return char2idx, idx2char

## STRING TO BOOLEAN
def str2bool(s): return s.lower() in ("yes", "true", "t", "1")

## NUMPY SIGMOID FUNCTION
def np_sigmoid(x): return np.divide(1, np.add(1, np.exp(np.negative(x))))

## GPU CONFIGURATION
def gpu_config(gpu_selection):
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selection)
	config = tf.ConfigProto()
	config.allow_soft_placement=True
	config.gpu_options.allow_growth=True
	config.log_device_placement=False
	return config

## CREATE A SPARSE REPRESENTATION
def sparse_tuple_from(sequences, dtype=np.int32):
	"""
	Create a sparse representention of the input sequences.
	
	Input/s:

		sequences: a list of lists of type dtype where each element is a sequence
		dtype: data type.

	Output/s:

		a tuple with (indices, values, shape)
	"""

	indices = []
	values = []

	for n, seq in enumerate(sequences):
		indices.extend(zip([n]*len(seq), range(len(seq))))
		values.extend(seq)

	indices = np.asarray(indices, dtype=np.int64)
	values = np.asarray(values, dtype=dtype)
	shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

	return indices, values, shape

# def sparse_tuple_from(sequences, dtype=np.int32):
#     """Creates a sparse representention of ``sequences``.
#     Args:
        
#         * sequences: a list of lists of type dtype where each element is a sequence
    
#     Returns a tuple with (indices, values, shape)
#     """
#     indices = []
#     values = []

#     for n, seq in enumerate(sequences):
#         indices.extend(zip([n]*len(seq), range(len(seq))))
#         values.extend(seq)

#     indices = np.asarray(indices, dtype=np.int64)
#     values = np.asarray(values, dtype=dtype)
#     shape = np.asarray([len(sequences), indices.max(0)[1]+1], dtype=np.int64)

#     return tf.SparseTensor(indices=indices, values=values, dense_shape=shape) 

## GENERAL ARGUMENTS
def args():
	parser = argparse.ArgumentParser()

	## OPTIONS (GENERAL)
	parser.add_argument('--gpu', default='0', type=str, help='GPU selection')
	parser.add_argument('--ver', type=str, help='Model version')
	parser.add_argument('--par_iter', default=256, type=int, help='dynamic_rnn/bidirectional_dynamic_rnn parallel iterations')
	parser.add_argument('--epoch', type=int, help='Epoch to use/retrain from')
	parser.add_argument('--train', default=False, type=str2bool, help='Training flag')
	parser.add_argument('--test', default=False, type=str2bool, help='Testing flag')
	parser.add_argument('--infer', default=False, type=str2bool, help='Inference flag')
	parser.add_argument('--verbose', default=False, type=str2bool, help='Verbose')
	parser.add_argument('--save_plot', default=False, type=bool, help='Save plot of feature input')

	## OPTIONS (TRAIN)
	parser.add_argument('--cont', default=False, type=str2bool, help='Continue testing from last epoch')
	parser.add_argument('--train_s_ver', default='v1', type=str, help='Clean speech training set version')
	parser.add_argument('--train_d_ver', default='v1', type=str, help='Noise training set version')
	parser.add_argument('--mbatch_size', default=10, type=int, help='Mini-batch size')
	parser.add_argument('--sample_size', default=250, type=int, help='Sample size')
	parser.add_argument('--max_epochs', default=250, type=int, help='Maximum number of epochs')

	## OPTIONS (TEST)
	# parser.add_argument('--val', default=False, type=str2bool, help='Find validation error for "test_epoch"')
	parser.add_argument('--test_x_ver', default='v2', type=str, help='Noisy speech test set version')
	parser.add_argument('--test_cer', default=False, type=str2bool, help='Test character error rate')
	parser.add_argument('--test_perplex', default=False, type=str2bool, help='Test set perplexity')
	parser.add_argument('--test_output', default=False, type=str2bool, help='Test output of LM')

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
	parser.add_argument('--model_path', default='model', type=str, help='Model save path')
	parser.add_argument('--set_path', default='set', type=str, help='Path to datasets')
	parser.add_argument('--stats_path', default='stats', 
		type=str, help='Path to training set statistics')
	parser.add_argument('--test_x_path', default='set/test_noisy_speech', 
		type=str, help='Path to the noisy speech test set')
	parser.add_argument('--out_path', default='out', 
		type=str, help='Output path')

	## ARTIFICIAL NEURAL NETWORK PARAMETERS
	parser.add_argument('--blocks', type=list, help='Residual blocks')
	parser.add_argument('--cell_size', default=256, type=int, help='Cell size')
	parser.add_argument('--cell_proj', default=None, type=int, help='Cell projection size (None for no proj.)')
	parser.add_argument('--cell_type', default=None, type=str, help='RNN cell type')
	parser.add_argument('--peep', default=None, type=str2bool, help='Use peephole connections')
	parser.add_argument('--bidi', default=None, type=str2bool, help='Bidirectional recurrent neural network flag')
	parser.add_argument('--bidi_con', default=None, type=str, help='Forward and backward cell activation connection')
	parser.add_argument('--res_con', default='add', type=str, help='Residual connection (add or concat)')
	parser.add_argument('--res_proj', default=None, type=int, help='Output size of the residual projection weight (None for no projection)')
	parser.add_argument('--block_unit', default=None, type=str, help='Residual unit')
	parser.add_argument('--coup_unit', default='CU1', type=str, help='Coupling unit')
	parser.add_argument('--coup_conv_filt', default=256, type=int, help='Number of filters for coupling unit')
	parser.add_argument('--conv_size', default=3, type=int, help='Convolution kernel size')
	parser.add_argument('--conv_filt', default=64, type=int, help='Number of convolution filters')
	parser.add_argument('--conv_caus', default=True, type=str2bool, help='Causal convolution flag')
	parser.add_argument('--max_dilation_rate', default=16, type=int, help='Maximum dilation rate')
	parser.add_argument('--dropout', default=False, type=str2bool, help='Use droput during training flag')
	parser.add_argument('--keep_prob', default=False, type=float, help='Keep probability during training (0.75 typically)')
	parser.add_argument('--context', default=None, type=int, help='Input context (no. of frames)')
	parser.add_argument('--depth', default=None, type=int, help='Temporal residual-dense lattice (tRDL) depth')
	parser.add_argument('--dilation_strat', default=None, type=str, help='tRDL dilation strategy (height)')
	parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')

	## FEATURES
	parser.add_argument('--min_snr', default=-10, type=int, help='Minimum trained SNR level')
	parser.add_argument('--max_snr', default=20, type=int, help='Maximum trained SNR level')
	parser.add_argument('--input_dim', default=257, type=int, help='Number of inputs')
	parser.add_argument('--num_outputs', default=257, type=int, help='Number of outputs')
	parser.add_argument('--fs', default=16000, type=int, help='Sampling frequency (Hz)')
	parser.add_argument('--Tw', default=32, type=int, help='Window length (ms)')
	parser.add_argument('--Ts', default=16, type=int, help='Window shift (ms)')
	parser.add_argument('--nconst', default=32768.0, type=float, help='Normalisation constant (see feat.addnoisepad())')

	## LANGUAGE MODEL
	parser.add_argument('--librispeech_data_path', default=expanduser("~") + '/data/librispeech', type=str, help='Path to store librispeech data')
	parser.add_argument('--seed', default=43, type=int, help='Random number seed')
	parser.add_argument('--max_words', default=50, type=int, help='Maximum number of words in a sentence')
	parser.add_argument('--word2vec_size', default=64, type=int, help='Word2vec map size')
	parser.add_argument('--num_train_sent', default=int(1e5), type=int, help='Number of training sentences')
	parser.add_argument('--num_val_sent', default=int(1e4), type=int, help='Number of validation sentences')
	parser.add_argument('--num_test_sent', default=int(1e4), type=int, help='Number of test sentences')

	args = parser.parse_args()
	return args
