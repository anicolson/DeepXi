## FILE:           .py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          .
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import tensorflow as tf

#########################
## ENCODER-DECODER CNN ##
#########################

def BlockConv2d(x, d_b, k_size, strides, padding='same'):
	'''
	Encoder bottlekneck block with convolutional 
	units, normalisation, and ReLU activation.

	Input/s:
		x - input to block.
		d_b - number of output channels.
		k_size - kernel dimensions (list of two integers). 
		strides - convolutional stride (list of two integers).
		padding - padding type.

	Output/s:
		output of block.
	'''
	if d_b < 2: bottlekneck = d_b
	else: bottlekneck = int(d_b/2)
	layer_1 = tf.layers.conv2d(tf.nn.relu(tf.contrib.layers.layer_norm(x)), 
		bottlekneck, [1,1], padding=padding, use_bias=False)
	layer_2 = tf.layers.conv2d(tf.nn.relu(tf.contrib.layers.layer_norm(layer_1)), 
		bottlekneck, k_size, strides=strides, padding=padding, use_bias=False)
	return tf.layers.conv2d(tf.nn.relu(tf.contrib.layers.layer_norm(layer_2)), 
		d_b, [1,1], padding=padding, use_bias=True)

def BlockTransConv2d(x, d_b, k_size, strides, padding='same'):
	'''
	Encoder bottlekneck block with transposed convolutional 
	units, normalisation, and ReLU activation.

	Input/s:
		x - input to block.
		d_b - number of output channels.
		k_size - kernel dimensions (list of two integers). 
		strides - convolutional stride (list of two integers).
		padding - padding type.

	Output/s:
		output of block.
	'''

	layer_1 = tf.layers.conv2d(tf.nn.relu(tf.contrib.layers.layer_norm(x)), 
		int(d_b/2), [1,1], padding=padding, use_bias=False)
	layer_2 = tf.layers.conv2d_transpose(tf.nn.relu(tf.contrib.layers.layer_norm(layer_1)), 
		int(d_b/2), k_size, strides=strides, padding=padding, use_bias=False)
	return tf.layers.conv2d(tf.nn.relu(tf.contrib.layers.layer_norm(layer_2)), 
		d_b, [1,1], padding=padding, use_bias=True)

def EncoderDecoder(x, c_in, c_out, d_b, k_size=[3,3], strides=[2, 1]):
	'''
	Encoder-decoder network with residual connections. Striding is used
	to reduce dimensionality, and transposed convolutions are used to
	increase dimensionality.

	Input/s:
		x - input to ResNet.
		c_in - input channels.
		c_out - output channels.
		d_b - list specifying the output size each block.
		k_size - kernel dimensions. 
		strides - convolutional stride (list of two integers).

	Output/s:
		unactivated output of decoder.
	'''

	## FIRST ENCODER BLOCK
	layer_1 = tf.nn.relu(tf.contrib.layers.layer_norm(tf.layers.conv2d(x, c_in, 
		[2,1], strides=[1,1], padding='valid', use_bias=False)))
	layer_2 = tf.layers.conv2d(tf.nn.relu(tf.contrib.layers.layer_norm(layer_1)), 
		int(d_b[0]/2), k_size, strides=strides, padding='same', use_bias=False)
	layer_3 = tf.layers.conv2d(tf.nn.relu(tf.contrib.layers.layer_norm(layer_2)), 
		d_b[0], [1,1], padding='same', use_bias=True)
	encoder = [layer_3]

	## SUBSEQUENT ENCODER BLOCKS
	for i in d_b[1:]: encoder.append(BlockConv2d(encoder[-1], i, k_size, strides)); 

	## DECODER BLOCKS
	decoder = [encoder[-1]]
	decoder.append(BlockTransConv2d(decoder[-1], d_b[-2], k_size, strides));
	for i, j in reversed(list(enumerate(d_b[0:-2]))):
		decoder.append(BlockTransConv2d(tf.add(encoder[i+1], 
			decoder[-1]), j, k_size, strides)); 

	## FINAL DECODER BLOCK 
	layer_1 = tf.layers.conv2d(tf.nn.relu(tf.contrib.layers.layer_norm(tf.add(encoder[0], decoder[-1]))), 
		int(d_b[0]/2), [1,1], padding='same', use_bias=False)
	layer_2 = tf.layers.conv2d_transpose(tf.nn.relu(tf.contrib.layers.layer_norm(layer_1)), 
		int(d_b[0]/2), k_size, strides=strides, padding='same', use_bias=False)
	layer_3 = tf.layers.conv2d_transpose(layer_2, c_out, [2,1], strides=[1,1], padding='valid', use_bias=True)
	return layer_3