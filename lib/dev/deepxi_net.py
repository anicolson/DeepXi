## FILE:           deepxi_net.py
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
## BRIEF:          Network employed withing the Deep Xi framework.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

## VARIABLE DESCRIPTIONS
# s - clean speech.
# d - noise.
# x - noisy speech.

from dev.acoustic.analysis_synthesis.polar import synthesis
from dev.acoustic.feat import polar
from dev.ResNet import ResNet
import dev.optimisation as optimisation
import numpy as np
import tensorflow as tf

## ARTIFICIAL NEURAL NETWORK 
class deepxi_net:
	def __init__(self, args):
		print('Preparing graph...')

		## RESNET		
		self.input_ph = tf.placeholder(tf.float32, shape=[None, None, args.d_in], name='input_ph') # noisy speech MS placeholder.
		self.nframes_ph = tf.placeholder(tf.int32, shape=[None], name='nframes_ph') # noisy speech MS sequence length placeholder.
		if args.model == 'ResNet':
			self.output = ResNet(self.input_ph, self.nframes_ph, args.norm_type, n_blocks=args.n_blocks, boolean_mask=True, d_out=args.d_out, 
				d_model=args.d_model, d_f=args.d_f, k_size=args.k_size, max_d_rate=args.max_d_rate)
		elif args.model == 'RDLNet':
			from dev.RDLNet import RDLNet
			self.output = RDLNet(self.input_ph, self.nframes_ph, args.norm_type, n_blocks=args.n_blocks, boolean_mask=True, d_out=args.d_out, 
				d_f=args.d_f, net_height=args.net_height)
		elif args.model == 'ResLSTM':
			from dev.ResLSTM import ResLSTM
			self.output = ResLSTM(self.input_ph, self.nframes_ph, args.norm_type, n_blocks=args.n_blocks, boolean_mask=True, d_out=args.d_out, d_model=args.d_model)

		## TRAINING FEATURE EXTRACTION GRAPH
		self.s_ph = tf.placeholder(tf.int16, shape=[None, None], name='s_ph') # clean speech placeholder.
		self.d_ph = tf.placeholder(tf.int16, shape=[None, None], name='d_ph') # noise placeholder.
		self.s_len_ph = tf.placeholder(tf.int32, shape=[None], name='s_len_ph') # clean speech sequence length placeholder.
		self.d_len_ph = tf.placeholder(tf.int32, shape=[None], name='d_len_ph') # noise sequence length placeholder.
		self.snr_ph = tf.placeholder(tf.float32, shape=[None], name='snr_ph') # SNR placeholder.
		self.train_feat = polar.input_target_xi(self.s_ph, self.d_ph, self.s_len_ph, 
			self.d_len_ph, self.snr_ph, args.N_w, args.N_s, args.NFFT, args.f_s, args.stats['mu_hat'], args.stats['sigma_hat'])

		## INFERENCE FEATURE EXTRACTION GRAPH
		self.infer_feat = polar.input(self.s_ph, self.s_len_ph, args.N_w, args.N_s, args.NFFT, args.f_s)

		## PLACEHOLDERS
		self.x_ph = tf.placeholder(tf.int16, shape=[None, None], name='x_ph') # noisy speech placeholder.
		self.x_len_ph = tf.placeholder(tf.int32, shape=[None], name='x_len_ph') # noisy speech sequence length placeholder.
		self.target_ph = tf.placeholder(tf.float32, shape=[None, args.d_out], name='target_ph') # training target placeholder.
		self.keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph') # keep probability placeholder.
		self.training_ph = tf.placeholder(tf.bool, name='training_ph') # training placeholder.

		## SYNTHESIS GRAPH
		if args.infer:	
			self.infer_output = tf.nn.sigmoid(self.output)
			self.y_MAG_ph = tf.placeholder(tf.float32, shape=[None, None, args.d_in], name='y_MAG_ph') 
			self.x_PHA_ph = tf.placeholder(tf.float32, [None, None, args.d_in], name='x_PHA_ph')
			self.y = synthesis(self.y_MAG_ph, self.x_PHA_ph, args.N_w, args.N_s, args.NFFT)

		## LOSS & OPTIMIZER
		self.loss = optimisation.loss(self.target_ph, self.output, 'mean_sigmoid_cross_entropy', axis=[1])
		self.total_loss = tf.reduce_mean(self.loss, axis=0)
		self.trainer, _ = optimisation.optimiser(self.total_loss, optimizer='adam', grad_clip=True)

		## SAVE VARIABLES
		self.saver = tf.train.Saver(max_to_keep=256)

		## NUMBER OF PARAMETERS
		args.params = (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
