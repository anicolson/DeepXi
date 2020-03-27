## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

# from os.path import expanduser
# import argparse, os, string
# import numpy as np
# from scipy.io.wavfile import write as wav_write
# import tensorflow as tf
# import soundfile as sf

from scipy.io import loadmat, savemat
import numpy as np
import os
import soundfile as sf
import tensorflow as tf

def save_wav(path, wav, f_s):
	"""
	"""
	wav = np.squeeze(wav)
	if isinstance(wav[0], np.float32): wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int16)
	sf.write(path, wav, f_s)

def read_wav(path):
	"""
	"""
	wav, f_s = sf.read(path, dtype='int16')
	return wav, f_s

def save_mat(path, data, name):
	"""
	"""
	if not path.endswith('.mat'): path = path + '.mat'
	savemat(path, {name:data})

def read_mat(path):
	"""
	"""
	if not path.endswith('.mat'): path = path + '.mat'
	return loadmat(path)

def gpu_config(gpu_selection, log_device_placement=False):
	"""
	"""
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selection)
	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)
