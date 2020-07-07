## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import argparse
import math

def read_dtype(x):
	x = x.replace("neg_", "-")
	if x == 'pi': return math.pi
	elif x == '-pi': return -math.pi
	elif any(map(str.isdigit, x)):
		if '.' in x: return float(x)
		else: return int(x)
	else: return x

def str_to_list(x):
	if (';' in x) and (',' in x): return [[read_dtype(z) for z in y.split(',')] for y in x.split(';')]
	elif ',' in x: return [read_dtype(y) for y in x.split(',')]
	else: return read_dtype(x)

def str_to_bool(s): return s.lower() in ("yes", "true", "t", "1")

def get_args():
	parser = argparse.ArgumentParser()

	## OPTIONS (GENERAL)
	parser.add_argument('--gpu', default='0', type=str, help='GPU selection')
	parser.add_argument('--ver', type=str, help='Model version')
	parser.add_argument('--test_epoch', type=str_to_list, help='Epoch to test')
	parser.add_argument('--train', default=False, type=str_to_bool, help='Perform training')
	parser.add_argument('--infer', default=False, type=str_to_bool, help='Perform inference and save outputs')
	parser.add_argument('--test', default=False, type=str_to_bool, help='Evaluate using objective measures')
	parser.add_argument('--spect_dist', default=False, type=str_to_bool, help='Find spectral distortion')
	parser.add_argument('--prelim', default=False, type=str_to_bool, help='Preliminary flag')
	parser.add_argument('--verbose', default=False, type=str_to_bool, help='Verbose')
	parser.add_argument('--network_type', type=str, help='Network type')
	parser.add_argument('--inp_tgt_type', type=str, help='Input and target type')
	parser.add_argument('--sd_snr_levels', default=[-5, 0, 5, 10, 15], type=str_to_list, help='SNR levels for spectral distortion')

	## OPTIONS (TRAIN)
	parser.add_argument('--mbatch_size', type=int, help='Mini-batch size')
	parser.add_argument('--sample_size', type=int, help='Sample size')
	parser.add_argument('--max_epochs', type=int, help='Maximum number of epochs')
	parser.add_argument('--resume_epoch', type=int, help='Epoch to resume training from')
	parser.add_argument('--save_model', default=False, type=str_to_bool, help='Save architecture, weights, and training configuration')
	parser.add_argument('--log_iter', default=False, type=str_to_bool, help='Log loss per training iteration')
	parser.add_argument('--eval_example', default=False, type=str_to_bool, help='Evaluate a mini-batch of training examples')
	parser.add_argument('--val_flag', default=True, type=str_to_bool, help='Use validation set')

	# INFERENCE OUTPUT TYPE
	# 'xi_hat' - a priori SNR estimate (.mat),
	# 'gain' - gain function (.mat),
	# 'deepmmse' - noise PSD estimate using DeepMMSE (.mat),
	# 'y' - enhanced speech (.wav),
	# 'd_hat' - noise estimate using DeepMMSE (.wav).
	parser.add_argument('--out_type', default='y', type=str, help='Output type for testing')

	## GAIN FUNCTION
	# 'ibm' - ideal binary mask (IBM),
	# 'wf' - Wiener filter (WF),
	# 'srwf' - square-root Wiener filter (SRWF),
	# 'cwf' - constrained Wiener filter (cWF),
	# 'mmse-stsa' - minimum-mean square error short-time spectral smplitude (MMSE-STSA) estimator,
	# 'mmse-lsa' - minimum-mean square error log-spectral amplitude (MMSE-LSA) estimator.
	parser.add_argument('--gain', default='mmse-lsa', type=str_to_list, help='Gain function for testing')

	## PATHS
	parser.add_argument('--model_path', default='model', type=str, help='Model save path')
	parser.add_argument('--set_path', default='set', type=str, help='Path to datasets')
	parser.add_argument('--data_path', default='data', type=str, help='Save data path')
	parser.add_argument('--test_x_path', default='set/test_noisy_speech', type=str, help='Path to the noisy-speech test set')
	parser.add_argument('--test_s_path', default='set/test_clean_speech', type=str, help='Path to the clean-speech test set')
	parser.add_argument('--test_d_path', default='set/test_noise', type=str, help='Path to the noise test set')
	parser.add_argument('--out_path', default='out', type=str, help='Output path')
	parser.add_argument('--saved_data_path', default=None, type=str, help='Path to saved .mat files necessary for speech enhancement')

	## FEATURES
	parser.add_argument('--min_snr', type=int, help='Minimum trained SNR level')
	parser.add_argument('--max_snr', type=int, help='Maximum trained SNR level')
	parser.add_argument('--snr_inter', type=int, help='Interval between SNR levels')
	parser.add_argument('--f_s', type=int, help='Sampling frequency (Hz)')
	parser.add_argument('--T_d', type=int, help='Window duration (ms)')
	parser.add_argument('--T_s', type=int, help='Window shift (ms)')
	parser.add_argument('--n_filters', default=None, type=int, help='Number of filters for subband ideal binary mask (IBM)')

	## NETWORK PARAMETERS
	parser.add_argument('--d_in', type=int, help='Input dimensionality')
	parser.add_argument('--d_out', type=int, help='Ouput dimensionality')
	parser.add_argument('--d_model', type=int, help='Model dimensions')
	parser.add_argument('--n_blocks', type=int, help='Number of blocks')
	parser.add_argument('--n_heads', type=int, help='Number of attention heads')
	parser.add_argument('--d_b', default=None, type=int, help='Bottleneck size')
	parser.add_argument('--d_f', default=None, type=int, help='Number of filters')
	parser.add_argument('--d_ff', default=None, type=int, help='Feed forward size')
	parser.add_argument('--k', default=None, type=int, help='Kernel size')
	parser.add_argument('--max_d_rate', default=None, type=int, help='Maximum dilation rate')
	parser.add_argument('--causal', type=str_to_bool, help='Causal network')
	parser.add_argument('--warmup_steps', type=int, help='Number of warmup steps')
	parser.add_argument('--net_height', default=[4], type=list, help='RDL block height')
	parser.add_argument('--centre', type=str_to_bool, help='Use centre parameter for layer normalisation')
	parser.add_argument('--scale', type=str_to_bool, help='Use scale parameter for layer normalisation')
	parser.add_argument('--unit_type', type=str, help='Block unit type')
	parser.add_argument('--loss_fnc', type=str, help='Loss function')

	## STATISTICS
	parser.add_argument('--xi_map_type', type=str, help='Function used to map the instantaneous a priori SNR')
	parser.add_argument('--xi_map_params', default=None, type=str_to_list, help='Parameters for the instantaneous a priori SNR map')
	parser.add_argument('--gamma_map_type', type=str, help='Function used to map the instantaneous a posteriori SNR')
	parser.add_argument('--gamma_map_params', default=None, type=str_to_list, help='Parameters for the instantaneous a posteriori SNR map')
	parser.add_argument('--s_stps_map_type', type=str, help='Function used to map the phase')
	parser.add_argument('--s_stps_map_params', default=None, type=str_to_list, help='Parameters for the phase map')

	parser.add_argument('--mu_xi_db', default=None, type=float, help='Mean of instantaneous a priori SNR (dB)')
	parser.add_argument('--sigma_xi_db', default=None, type=float, help='Standard deviation of instantaneous a priori SNR (dB)')
	parser.add_argument('--b_gamma', default=None, type=float, help='Upper bounds for truncated double gamma distribution of a posteriori SNR.')
	parser.add_argument('--alpha_gamma', default=None, type=float, help='Shape parameter for the instantaneous a posteriori SNR distribution')
	parser.add_argument('--stdct_mu_xi_db', default=None, type=float, help='Mean of instantaneous a priori SNR (dB) (STDCT)')
	parser.add_argument('--stdct_sigma_xi_db', default=None, type=float, help='Standard deviation of instantaneous a priori SNR (dB) (STDCT)')

	parser.add_argument('--m_gamma_db', default=None, type=float, help='Location parameter for the instantaneous a posteriori SNR (dB) distribution')
	parser.add_argument('--lambda_gamma_db', default=None, type=float, help='Scale parameter for the instantaneous a posteriori SNR (dB) distribution')
	parser.add_argument('--kappa_gamma_db', default=None, type=float, help='Asymmetry parameter for the instantaneous a posteriori SNR (dB) distribution')
	parser.add_argument('--mu_ps_db', default=None, type=float, help='Mean of power spectrum components (dB)')
	parser.add_argument('--sigma_ps_db', default=None, type=float, help='Standard deviation of power spectrum components (dB)')

	args = parser.parse_args()
	return args
