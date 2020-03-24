## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from dev.utils import read_wav
from tqdm import tqdm
import dev.gain as gain
import dev.utils as utils
import dev.xi as xi
import numpy as np
import os
import scipy.io as spio

## INFERENCE
def infer(sess, net, args):
	print("Inference...")
	net.saver.restore(sess, args.model_path + '/epoch-' + str(args.epoch)) # load model from epoch.
	
	if args.out_type == 'xi_hat': args.out_path = args.out_path + '/xi_hat'
	elif args.out_type == 'y': args.out_path = args.out_path + '/' + args.gain + '/y'
	elif args.out_type == 'ibm_hat': args.out_path = args.out_path + '/ibm_hat'
	else: ValueError('Incorrect output type.')

	if not os.path.exists(args.out_path): os.makedirs(args.out_path) # make output directory.

	for j in tqdm(args.test_x_list):
		(wav, _) = read_wav(j['file_path']) # read wav from given file path.		
		input_feat = sess.run(net.infer_feat, feed_dict={net.s_ph: [wav], net.s_len_ph: [j['seq_len']]}) # sample of training set.
		xi_bar_hat = sess.run(net.infer_output, feed_dict={net.input_ph: input_feat[0], 
			net.nframes_ph: input_feat[1], net.training_ph: False}) # output of network.
		xi_hat = xi.xi_hat(xi_bar_hat, args.stats['mu_hat'], args.stats['sigma_hat'])

		file_name = j['file_path'].rsplit('/',1)[1].split('.')[0]

		if args.out_type == 'xi_hat':
			spio.savemat(args.out_path + '/' + file_name + '.mat', {'xi_hat':xi_hat})

		elif args.out_type == 'y':
			y_MAG = np.multiply(input_feat[0], gain.gfunc(xi_hat, xi_hat+1, gtype=args.gain))
			y = np.squeeze(sess.run(net.y, feed_dict={net.y_MAG_ph: y_MAG, 
				net.x_PHA_ph: input_feat[2], net.nframes_ph: input_feat[1], net.training_ph: False})) # output of network.
			if np.isnan(y).any(): ValueError('NaN values found in enhanced speech.')
			if np.isinf(y).any(): ValueError('Inf values found in enhanced speech.')
			utils.save_wav(args.out_path + '/' + file_name + '.wav', args.f_s, y)

		elif args.out_type == 'ibm_hat':
			ibm_hat = np.greater(xi_hat, 1.0)
			spio.savemat(args.out_path + '/' + file_name + '.mat', {'ibm_hat':ibm_hat})

	print('Inference complete.')