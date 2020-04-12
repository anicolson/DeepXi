## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import argparse

def str_to_bool(s): return s.lower() in ("yes", "true", "t", "1")

def get_args():
	parser = argparse.ArgumentParser()

	## OPTIONS (GENERAL)
	parser.add_argument('--gpu', default='0', type=str, help='GPU selection')
	parser.add_argument('--ver', type=str, help='Model version')
	parser.add_argument('--test_epoch', type=int, help='Epoch to test')
	parser.add_argument('--train', default=False, type=str_to_bool, help='Training flag')
	parser.add_argument('--infer', default=False, type=str_to_bool, help='Inference flag')
	parser.add_argument('--prelim', default=False, type=str_to_bool, help='Preliminary flag')
	parser.add_argument('--verbose', default=False, type=str_to_bool, help='Verbose')
	parser.add_argument('--network', type=str, help='Network type')

	## OPTIONS (TRAIN)
	parser.add_argument('--mbatch_size', type=int, help='Mini-batch size')
	parser.add_argument('--sample_size', type=int, help='Sample size')
	parser.add_argument('--max_epochs', type=int, help='Maximum number of epochs')
	parser.add_argument('--resume_epoch', type=int, help='Epoch to resume training from')
	parser.add_argument('--save_model', default=False, type=str_to_bool, help='Save architecture, weights, and training configuration')
	parser.add_argument('--log_iter', default=False, type=str_to_bool, help='Log loss per training iteration')
	parser.add_argument('--eval_example', default=False, type=str_to_bool, help='Evaluate a mini-batch of training examples')

	# TEST OUTPUT TYPE
	# 'raw' - output from network (.mat), 'xi_hat' - a priori SNR estimate (.mat),
	# 'gain' - gain function (.mat), 'y' - enhanced speech (.wav).
	parser.add_argument('--out_type', default='y', type=str, help='Output type for testing')

	## GAIN FUNCTION
	# 'ibm' - Ideal Binary Mask (IBM), 'wf' - Wiener Filter (WF), 'srwf' - Square-Root Wiener Filter (SRWF),
	# 'cwf' - Constrained Wiener Filter (cWF), 'mmse-stsa' - Minimum-Mean Square Error - Short-Time Spectral Amplitude (MMSE-STSA) estimator,
	# 'mmse-lsa' - Minimum-Mean Square Error - Log-Spectral Amplitude (MMSE-LSA) estimator.
	parser.add_argument('--gain', default='wf', type=str, help='Gain function for testing')

	## PATHS
	parser.add_argument('--model_path', default='model', type=str, help='Model save path')
	parser.add_argument('--set_path', default='set', type=str, help='Path to datasets')
	parser.add_argument('--data_path', default='data', type=str, help='Save data path')
	parser.add_argument('--test_x_path', default='set/test_noisy_speech', type=str, help='Path to the noisy speech test set')
	parser.add_argument('--out_path', default='out', type=str, help='Output path')

	## FEATURES
	parser.add_argument('--min_snr', type=int, help='Minimum trained SNR level')
	parser.add_argument('--max_snr', type=int, help='Maximum trained SNR level')
	parser.add_argument('--f_s', type=int, help='Sampling frequency (Hz)')
	parser.add_argument('--T_d', type=int, help='Window duration (ms)')
	parser.add_argument('--T_s', type=int, help='Window shift (ms)')

	## NETWORK PARAMETERS
	parser.add_argument('--d_in', type=int, help='Input dimensionality')
	parser.add_argument('--d_out', type=int, help='Ouput dimensionality')
	parser.add_argument('--d_model', type=int, help='Model dimensions')
	parser.add_argument('--n_blocks', type=int, help='Number of blocks')
	parser.add_argument('--d_f', type=int, help='Number of filters')
	parser.add_argument('--k', type=int, help='Kernel size')
	parser.add_argument('--max_d_rate', type=int, help='Maximum dilation rate')

	parser.add_argument('--net_height', default=[4], type=list, help='RDL block height')

	args = parser.parse_args()
	return args
