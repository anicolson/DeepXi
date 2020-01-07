## FILE:           deepxi.py
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
## BRIEF:          'DeepXi' training and testing.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os, sys 
sys.path.insert(0, 'lib')
from dev.args import add_args, get_args
from dev.infer import infer
from dev.sample_stats import get_stats
from dev.train import train
import dev.deepxi_net as deepxi_net
import numpy as np
import tensorflow as tf
import dev.utils as utils
np.set_printoptions(threshold=1e6)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == '__main__':

	## GET COMMAND LINE ARGUMENTS
	args = get_args()

	## TRAINING AND TESTING SET ARGUMENTS
	args = add_args(args)

	## GPU CONFIGURATION
	config = utils.gpu_config(args.gpu)

	## GET STATISTICS
	args = get_stats(args.data_path, args, config)

	## MAKE DEEP XI NNET
	net = deepxi_net.deepxi_net(args)

	with tf.Session(config=config) as sess:
		if args.train: train(sess, net, args)
		if args.infer: infer(sess, net, args)
