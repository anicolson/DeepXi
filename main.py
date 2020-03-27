## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from deepxi.args import get_args
from deepxi.model import DeepXi
from deepxi.se_batch import Batch, Batch_list
import deepxi.utils as utils
import numpy as np
import os

def add_args(args):
	"""
	"""
	args.model_path = args.model_path + '/' + args.ver # model save path.
	args.train_s_path = args.set_path + '/train_clean_speech' # path to the clean speech training set.
	args.train_d_path = args.set_path + '/train_noise' # path to the noise training set.
	args.val_s_path = args.set_path + '/val_clean_speech' # path to the clean speech validation set.
	args.val_d_path = args.set_path + '/val_noise' # path to the noise validation set.
	args.out_path = args.out_path + '/' + args.ver + '/' + 'e' + str(args.epoch) # output path.
	args.N_w = int(args.f_s*args.T_w*0.001) # window length (samples).
	args.N_s = int(args.f_s*args.T_s*0.001) # window shift (samples).
	args.NFFT = int(pow(2, np.ceil(np.log2(args.N_w)))) # number of DFT components.

	if args.train:
		args.train_s_list = Batch_list(args.train_s_path, 'clean_speech_' + args.set_path.rsplit('/', 1)[-1], args.data_path)
		args.train_d_list = Batch_list(args.train_d_path, 'noise_' + args.set_path.rsplit('/', 1)[-1], args.data_path)
		args.val_s, args.val_s_len, args.val_snr, _ = Batch(args.val_s_path, list(range(args.min_snr, args.max_snr + 1)))
		args.val_d, args.val_d_len, _, _ = Batch(args.val_d_path, list(range(args.min_snr, args.max_snr + 1)))
		args.train_steps=int(np.ceil(len(args.train_s_list)/args.mbatch_size))
		args.val_steps=int(np.ceil(args.val_s.shape[0]/args.mbatch_size))

	if args.infer: args.test_x, args.test_x_len, _, args.test_x_base_names = Batch(args.test_x_path)

	return args

if __name__ == '__main__':

	args = get_args()
	args = add_args(args)
	config = utils.gpu_config(args.gpu)

	deepxi = DeepXi(
		args.N_w, 
		args.N_s, 
		args.NFFT, 
		args.f_s, 
		min_snr=args.min_snr, 
		max_snr=args.max_snr
		)

	if args.train: deepxi.train(
		args.train_s_list, 
		args.train_d_list, 
		model_path=args.model_path,
		val_s=args.val_s,
		val_d=args.val_d,
		val_s_len=args.val_s_len,
		val_d_len=args.val_d_len,
		val_snr=args.val_snr, 
		val_save_path=args.data_path,
		stats_path=args.data_path, 
		sample_size=args.sample_size,
		mbatch_size=args.mbatch_size, 
		max_epochs=args.max_epochs, 
		resume_epoch=args.resume_epoch,
		ver=args.ver
		)
	if args.infer: deepxi.infer(
		args.test_x[0:10], 
		args.test_x_len[0:10],
		args.test_x_base_names[0:10],
		args.epoch,
		model_path=args.model_path,
		out_type=args.out_type,
		gain=args.gain,
		out_path=args.out_path,
		stats_path=args.data_path
		)
