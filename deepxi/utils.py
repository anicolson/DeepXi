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

import numpy as np
import os
import soundfile as sf
import tensorflow as tf

def save_wav(save_path, f_s, wav):
	"""
	"""
	if isinstance(wav[0], np.float32): wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int16)
	wav_write(save_path, f_s, wav)

def read_wav(path):
	"""
	"""
	wav, f_s = sf.read(path, dtype='int16')
	return wav, f_s

def gpu_config(gpu_selection, log_device_placement=False):
	"""
	"""
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selection)
	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)