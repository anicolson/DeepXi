## FILE:           .py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF: ResNet with bottlekneck blocks and 1D causal dlated convolutional units.        .
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import tensorflow as tf
from tensorflow.python.training import moving_averages
from dev.normalisation import Normalisation
import numpy as np
import argparse, math, sys
from dev.add_noise import add_noise_batch

def CausalDilatedConv1d(x, d_f, k_size, d_rate=1, use_bias=True):
	'''
	1D Causal dilated convolutional unit.

	Input/s:
		x - input.
		d_f - filter dimensions. 
		k_size - kernel dimensions. 
		d_rate - dilation rate.
		use_bias - include use bias vector.

	Output/s:
		output of convolutional unit.
	'''

	if k_size > 1: # padding for causality.
		x_shape = tf.shape(x)
		x = tf.concat([tf.zeros([x_shape[0], (k_size - 1)*d_rate, x_shape[2]]), x], 1)
	return tf.layers.conv1d(x, d_f, k_size, dilation_rate=d_rate, 
		activation=None, padding='valid', use_bias=use_bias)

def BottlekneckBlock(x, norm_type, seq_len, d_model, d_f, k_size, d_rate):
	'''
	Bottlekneck block with causal dilated convolutional 
	units, and normalisation.

	Input/s:
		x - input to block.
		norm_type - normalisation type.				
		seq_len - length of each sequence.
		d_out - output dimensions.
		d_f - filter dimensions. 
		k_size - kernel dimensions. 
		d_rate - dilation rate.

	Output/s:
		output of residual block.
	'''

	layer_1 = CausalDilatedConv1d(tf.nn.relu(Normalisation(x, norm_type, seq_len=seq_len)), d_f, 1, 1, False)
	layer_2 = CausalDilatedConv1d(tf.nn.relu(Normalisation(layer_1, norm_type, seq_len=seq_len)), d_f, k_size, d_rate, False)
	layer_3 = CausalDilatedConv1d(tf.nn.relu(Normalisation(layer_2, norm_type, seq_len=seq_len)), d_model, 1, 1, True)
	return tf.add(x, layer_3)

def ResNet(x, seq_len, norm_type, training=None, d_out=257, 
	n_blocks=40, d_model=256, d_f=64, k_size=3, max_d_rate=16, out_layer=True, boolean_mask=False):
	'''
	ResNet with bottlekneck blocks, causal dilated convolutional 
	units, and normalisation. Dilation resets after 
	exceeding 'max_d_rate'.

	Input/s:
		x - input to ResNet.
		norm_type - normalisation type.		
		seq_len - length of each sequence.
		training - training flag.
		d_out - output dimensions.
		n_blocks - number of residual blocks. 
		d_model - model dimensions.
		d_f - filter dimensions. 
		k_size - kernel dimensions. 
		max_d_rate - maximum dilation rate.

	Output/s:
		unactivated output of ResNet.
	'''
	# mask = tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32) # convert mask to float.
	blocks = [tf.nn.relu(Normalisation(tf.layers.dense(x, d_model, use_bias=False), norm_type, seq_len=seq_len))] # (W -> Norm -> ReLU).
	for i in range(n_blocks): blocks.append(BottlekneckBlock(blocks[-1], norm_type, seq_len, 
		d_model, d_f, k_size, int(2**(i%(np.log2(max_d_rate)+1)))))
	if boolean_mask: blocks[-1] = tf.boolean_mask(blocks[-1], tf.sequence_mask(seq_len))
	return tf.layers.dense(blocks[-1], d_out, use_bias=True)
