## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import tensorflow as tf
from tensorflow.python.training import moving_averages
from dev.normalisation import Normalisation
import numpy as np
import argparse, math, sys

def ResLSTMBlock(x, d_model, seq_len, block, parallel_iterations=1024):
	'''
	ResLSTM block.

	Input/s:
		x - input to block.
		d_model - cell size.
		seq_len - sequence length.
		block - block number.
		parallel_iterations - number of parrallel iterations.

	Output/s:
		output of residual block.
	'''
	with tf.variable_scope( 'block_' + str(block)):
		cell = tf.contrib.rnn.LSTMCell(d_model)
		# activation, _ = tf.nn.static_rnn(cell, [tf.expand_dims(x[0,0,:],0)]*seq_len[0], dtype=tf.float32)
		activation, _ = tf.nn.dynamic_rnn(cell, x, seq_len, swap_memory=True, 
			parallel_iterations=parallel_iterations, dtype=tf.float32)
		return tf.add(x, activation)

def ResLSTM(x, seq_len, norm_type, training=None, d_out=257, 
	n_blocks=5, d_model=512, out_layer=True, boolean_mask=False):
	'''
	ResLSTM network.

	Input/s:
		x - input.
		seq_len - length of each sequence.
		norm_type - normalisation type.
		training - training flag.		
		d_out - output dimensions.
		n_blocks - number of residual blocks. 
		d_model - cell size.
		out_layer - add an output layer. 
		boolean_mask - convert padded 3D output to unpadded and stacked 2D output. 

	Output/s:
		unactivated output of ResLSTM.
	'''
	blocks = [tf.nn.relu(Normalisation(tf.layers.dense(x, 
		d_model, use_bias=False), norm_type, seq_len))] # (W -> Norm -> ReLU).
	for i in range(n_blocks): blocks.append(ResLSTMBlock(blocks[-1], d_model, seq_len, i))
	if boolean_mask: blocks[-1] = tf.boolean_mask(blocks[-1], tf.sequence_mask(seq_len))
	return tf.layers.dense(blocks[-1], d_out, use_bias=True)
