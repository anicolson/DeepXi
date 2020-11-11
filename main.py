## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from deepxi.args import get_args
from deepxi.model import DeepXi
from deepxi.prelim import Prelim
from deepxi.se_batch import Batch
import deepxi.utils as utils
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

	args = get_args()

	print("Arguments:")
	[print(key,val) for key,val in vars(args).items()]

	if args.causal: args.padding = "causal"
	else: args.padding = "same"

	args.model_path = args.model_path + '/' + args.ver # model save path.
	if args.set_path != "set": args.data_path = args.data_path + '/' + args.set_path.rsplit('/', 1)[-1] # data path.
	train_s_path = args.set_path + '/train_clean_speech' # path to the clean speech training set.
	train_d_path = args.set_path + '/train_noise' # path to the noise training set.
	val_s_path = args.set_path + '/val_clean_speech' # path to the clean speech validation set.
	val_d_path = args.set_path + '/val_noise' # path to the noise validation set.
	N_d = int(args.f_s*args.T_d*0.001) # window duration (samples).
	N_s = int(args.f_s*args.T_s*0.001) # window shift (samples).
	K = int(pow(2, np.ceil(np.log2(N_d)))) # number of DFT components.

	if args.train:
		train_s_list = utils.batch_list(train_s_path, 'clean_speech', args.data_path)
		train_d_list = utils.batch_list(train_d_path, 'noise', args.data_path)
		if args.val_flag:
			val_s, val_d, val_s_len, val_d_len, val_snr = utils.val_wav_batch(val_s_path, val_d_path)
		else: val_s, val_d, val_s_len, val_d_len, val_snr = None, None, None, None, None
	else: train_s_list, train_d_list = None, None

	if args.infer or args.test:
		test_x, test_x_len, _, test_x_base_names = Batch(args.test_x_path)
	if args.test: test_s, test_s_len, _, test_s_base_names = Batch(args.test_s_path)

	config = utils.gpu_config(args.gpu)

	print("Version: %s." % (args.ver))

	deepxi = DeepXi(
		N_d=N_d,
		N_s=N_s,
		K=K,
		sample_dir=args.data_path,
		train_s_list=train_s_list,
		train_d_list=train_d_list,
		**vars(args)
		)

	if args.train: deepxi.train(
		train_s_list=train_s_list,
		train_d_list=train_d_list,
		model_path=args.model_path,
		val_s=val_s,
		val_d=val_d,
		val_s_len=val_s_len,
		val_d_len=val_d_len,
		val_snr=val_snr,
		val_save_path=args.data_path,
		val_flag=args.val_flag,
		mbatch_size=args.mbatch_size,
		max_epochs=args.max_epochs,
		resume_epoch=args.resume_epoch,
		eval_example=args.eval_example,
		loss_fnc=args.loss_fnc,
		log_path=args.log_path
		)

	if args.infer: deepxi.infer(
		test_x=test_x,
		test_x_len=test_x_len,
		test_x_base_names=test_x_base_names,
		test_epoch=args.test_epoch,
		model_path=args.model_path,
		out_type=args.out_type,
		gain=args.gain,
		out_path=args.out_path,
		n_filters=args.n_filters,
		saved_data_path=args.saved_data_path,
		)

	if args.test: deepxi.test(
		test_x=test_x,
		test_x_len=test_x_len,
		test_x_base_names=test_x_base_names,
		test_s=test_s,
		test_s_len=test_s_len,
		test_s_base_names=test_s_base_names,
		test_epoch=args.test_epoch,
		model_path=args.model_path,
		gain=args.gain,
		log_path=args.log_path
		)
