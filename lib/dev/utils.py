## FILE:           utils.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          General utility functions/modules.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from os.path import expanduser
import argparse, os, string
import numpy as np
from scipy.io.wavfile import write as wav_write
import tensorflow as tf

def save_wav(save_path, f_s, wav):
	if isinstance(wav[0], np.float32): wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int16)
	wav_write(save_path, f_s, wav)

def log10(x):
    numerator = tf.log(x)
    denominator = tf.constant(np.log(10), dtype=numerator.dtype)
    return tf.truediv(numerator, denominator)

## CHARACTER DICTIONARIES
def char_dict():
	chars = list(" " + string.ascii_lowercase + "'") # 26 alphabetic characters + space + EOS + blank = 29 classes.
	char2idx = dict(zip(chars, [i for i in range(len(chars))])) 
	idx2char = dict((y,x) for x,y in char2idx.items())
	return char2idx, idx2char

## NUMPY SIGMOID FUNCTION
def np_sigmoid(x): return np.divide(1, np.add(1, np.exp(np.negative(x))))

## GPU CONFIGURATION
def gpu_config(gpu_selection, log_device_placement=False):
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selection)
	config = tf.ConfigProto()
	config.allow_soft_placement=True
	config.gpu_options.allow_growth=True
	config.log_device_placement=log_device_placement
	return config

## CREATE A SPARSE REPRESENTATION
def sparse_tuple_from(sequences, dtype=np.int32):
	"""
	Create a sparse representention of the input sequences.
	
	Input/s:

		sequences: a list of lists of type dtype where each element is a sequence
		dtype: data type.

	Output/s:

		a tuple with (indices, values, shape)
	"""

	indices = []
	values = []

	for n, seq in enumerate(sequences):
		indices.extend(zip([n]*len(seq), range(len(seq))))
		values.extend(seq)

	indices = np.asarray(indices, dtype=np.int64)
	values = np.asarray(values, dtype=dtype)
	shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

	return indices, values, shape

# def sparse_tuple_from(sequences, dtype=np.int32):
#     """Creates a sparse representention of ``sequences``.
#     Args:
        
#         * sequences: a list of lists of type dtype where each element is a sequence
    
#     Returns a tuple with (indices, values, shape)
#     """
#     indices = []
#     values = []

#     for n, seq in enumerate(sequences):
#         indices.extend(zip([n]*len(seq), range(len(seq))))
#         values.extend(seq)

#     indices = np.asarray(indices, dtype=np.int64)
#     values = np.asarray(values, dtype=dtype)
#     shape = np.asarray([len(sequences), indices.max(0)[1]+1], dtype=np.int64)

#     return tf.SparseTensor(indices=indices, values=values, dense_shape=shape) 
