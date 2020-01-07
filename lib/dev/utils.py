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
import soundfile as sf

def save_wav(save_path, f_s, wav):
	if isinstance(wav[0], np.float32): wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int16)
	wav_write(save_path, f_s, wav)

def read_wav(path):
	wav, f_s = sf.read(path, dtype='int16')
	return wav, f_s

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