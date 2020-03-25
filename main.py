## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

# import os, sys 
# sys.path.insert(0, 'lib')
# from dev.args import add_args, get_args
# from dev.infer import infer
# from dev.sample_stats import get_stats
# from dev.train import train
# import dev.deepxi_net as deepxi_net
# import numpy as np
# import tensorflow as tf
# import dev.utils as utils
# np.set_printoptions(threshold=1e6)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from deepxi.args import get_args
from deepxi.model import DeepXi
from deepxi.se_batch import Batch, Batch_list
import deepxi.utils as utils
import numpy as np
import os

def add_args(args):
	"""
	"""

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
		args.train_s_list = Batch_list(args.train_s_path, 'clean_speech_' + args.set_path.rsplit('/', 1)[-1], args.data_path) # clean speech training list.
		args.train_d_list = Batch_list(args.train_d_path, 'noise_' + args.set_path.rsplit('/', 1)[-1], args.data_path) # noise training list.
		if not os.path.exists(args.model_path): os.makedirs(args.model_path) # make model path directory.
		args.val_s, args.val_s_len, args.val_snr, _ = Batch(args.val_s_path,  
			list(range(args.min_snr, args.max_snr + 1))) # clean validation waveforms and lengths.
		args.val_d, args.val_d_len, _, _ = Batch(args.val_d_path, 
			list(range(args.min_snr, args.max_snr + 1))) # noise validation waveforms and lengths.
		args.train_steps=int(np.ceil(len(args.train_s_list)/args.mbatch_size))
		args.val_steps=int(np.ceil(args.val_s.shape[0]/args.mbatch_size))

	## INFERENCE
	# if args.infer: args.test_x, args.test_x_len, args.test_snr, args.test_fnames = Batch(args.test_x_path, '*', []) # noisy speech test waveforms and lengths.
	if args.infer: args.test_x_list = Batch_list(args.test_x_path, 'test_x', args.data_path, make_new=True)
	return args

if __name__ == '__main__':

	args = get_args()
	args = add_args(args)
	config = utils.gpu_config(args.gpu)

	deepxi = DeepXi(args.N_w, args.N_s, args.NFFT, args.f_s, min_snr=args.min_snr, 
		max_snr=args.max_snr, save_dir=args.model_path)

	if args.train: deepxi.train(
		args.train_s_list, 
		args.train_d_list, 
		stats_path=args.data_path, 
		sample_size=args.sample_size,
		mbatch_size=args.mbatch_size, 
		max_epochs=args.max_epochs, 
		ver=args.ver,
		resume=args.cont,
		start_epoch=args.epoch
		)
	if args.infer: deepxi.infer()
