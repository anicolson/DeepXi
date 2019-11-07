## FILE:           args.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Get command line arguments.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import argparse
import numpy as np
import os
from dev.se_batch import Train_list, Batch
from os.path import expanduser

## ADD ADDITIONAL ARGUMENTS
def add_args(args, modulation=False):

	## DEPENDANT OPTIONS
	args.model_path = args.model_path + '/' + args.ver # model save path.
	args.train_s_path = args.set_path + '/train_clean_speech' # path to the clean speech training set.
	args.train_d_path = args.set_path + '/train_noise' # path to the noise training set.
	args.val_s_path = args.set_path + '/val_clean_speech' # path to the clean speech validation set.
	args.val_d_path = args.set_path + '/val_noise' # path to the noise validation set.
	args.out_path = args.out_path + '/' + args.ver + '/' + 'e' + str(args.epoch) # output path.
	args.N_w = int(args.f_s*args.T_w*0.001) # window length (samples).
	args.N_s = int(args.f_s*args.T_s*0.001) # window shift (samples).
	args.NFFT = int(pow(2, np.ceil(np.log2(args.N_w)))) # number of DFT components.

	## DATASETS
	if args.train: ## TRAINING AND VALIDATION CLEAN SPEECH AND NOISE SET
		args.train_s_list = Train_list(args.train_s_path, '*.wav', 'clean_speech_' + args.set_path.rsplit('/', 1)[-1], args.data_path) # clean speech training list.
		args.train_d_list = Train_list(args.train_d_path, '*.wav', 'noise_' + args.set_path.rsplit('/', 1)[-1], args.data_path) # noise training list.
		if not os.path.exists(args.model_path): os.makedirs(args.model_path) # make model path directory.
		args.val_s, args.val_s_len, args.val_snr, _ = Batch(args.val_s_path, '*.wav', 
			list(range(args.min_snr, args.max_snr + 1))) # clean validation waveforms and lengths.
		args.val_d, args.val_d_len, _, _ = Batch(args.val_d_path, '*.wav', 
			list(range(args.min_snr, args.max_snr + 1))) # noise validation waveforms and lengths.
		args.train_steps=int(np.ceil(len(args.train_s_list)/args.mbatch_size))
		args.val_steps=int(np.ceil(args.val_s.shape[0]/args.mbatch_size))

	## INFERENCE
	if args.infer: args.test_x, args.test_x_len, args.test_snr, args.test_fnames = Batch(args.test_x_path, '*.wav', []) # noisy speech test waveforms and lengths.
	return args

## STRING TO BOOLEAN
def str2bool(s): return s.lower() in ("yes", "true", "t", "1")

## GET COMMAND LINE ARGUMENTS
def get_args():
	parser = argparse.ArgumentParser()

	## OPTIONS (GENERAL)
	parser.add_argument('--gpu', default='0', type=str, help='GPU selection')
	parser.add_argument('--ver', default='3d', type=str, help='Model version')
	parser.add_argument('--epoch', default=173, type=int, help='Epoch to use/retrain from')
	parser.add_argument('--train', default=False, type=str2bool, help='Training flag')
	parser.add_argument('--infer', default=False, type=str2bool, help='Inference flag')
	parser.add_argument('--verbose', default=False, type=str2bool, help='Verbose')

	## OPTIONS (TRAIN)
	parser.add_argument('--cont', default=False, type=str2bool, help='Continue testing from last epoch')
	parser.add_argument('--mbatch_size', default=10, type=int, help='Mini-batch size')
	parser.add_argument('--sample_size', default=1000, type=int, help='Sample size')
	parser.add_argument('--max_epochs', default=250, type=int, help='Maximum number of epochs')
	parser.add_argument('--grad_clip', default=True, type=str2bool, help='Gradient clipping')

	# TEST OUTPUT TYPE
	# 'xi_hat' - a priori SNR estimate (.mat),
	# 'y' - enhanced speech (.wav).
	parser.add_argument('--out_type', default='y', type=str, help='Output type for testing')

	## GAIN FUNCTION
	# 'ibm' - Ideal Binary Mask (IBM), 'wf' - Wiener Filter (WF), 'srwf' - Square-Root Wiener Filter (SRWF),
	# 'cwf' - Constrained Wiener Filter (cWF), 'mmse-stsa' - Minimum-Mean Square Error - Short-Time Spectral Amplitude (MMSE-STSA) estimator,
	# 'mmse-lsa' - Minimum-Mean Square Error - Log-Spectral Amplitude (MMSE-LSA) estimator.
	parser.add_argument('--gain', default='srwf', type=str, help='Gain function for testing')

	## PATHS
	parser.add_argument('--model_path', default='model', type=str, help='Model save path')
	parser.add_argument('--set_path', default='set', type=str, help='Path to datasets')
	parser.add_argument('--data_path', default='data', type=str, help='Save data path')
	parser.add_argument('--stats_path', default='stats', 
		type=str, help='Path to training set statistics')
	parser.add_argument('--test_x_path', default='set/test_noisy_speech', 
		type=str, help='Path to the noisy speech test set')
	parser.add_argument('--out_path', default='out', 
		type=str, help='Output path')

	## FEATURES
	parser.add_argument('--min_snr', default=-10, type=int, help='Minimum trained SNR level')
	parser.add_argument('--max_snr', default=20, type=int, help='Maximum trained SNR level')
	parser.add_argument('--f_s', default=16000, type=int, help='Sampling frequency (Hz)')
	parser.add_argument('--T_w', default=32, type=int, help='Window length (ms)')
	parser.add_argument('--T_s', default=16, type=int, help='Window shift (ms)')
	parser.add_argument('--nconst', default=32768.0, type=float, help='Normalisation constant (see feat.addnoisepad())')

	## NETWORK PARAMETERS
	parser.add_argument('--d_in', default=257, type=int, help='Input dimensionality')
	parser.add_argument('--d_out', default=257, type=int, help='Ouput dimensionality')
	parser.add_argument('--d_model', default=256, type=int, help='Model dimensions')
	parser.add_argument('--n_blocks', default=40, type=int, help='Number of blocks')
	parser.add_argument('--d_f', default=64, type=int, help='Number of filters')
	parser.add_argument('--k_size', default=3, type=int, help='Kernel size')
	parser.add_argument('--max_d_rate', default=16, type=int, help='Maximum dilation rate')
	parser.add_argument('--norm_type', default='FrameLayerNorm', type=str, help='Normalisation type')

	args = parser.parse_args()
	return args
