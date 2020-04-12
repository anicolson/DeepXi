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
from soundfile import SoundFile, SEEK_END
import numpy as np
import glob, os, pickle, platform
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
	savemat(path, {name: data})

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

def batch_list(file_dir, list_name, data_path='data', make_new=False):
	"""
	Places the file paths and wav lengths of an audio file into a dictionary, which
	is then appended to a list. 'glob' is used to support Unix style pathname
	pattern expansions. Checks if the training list has already been saved, and loads
	it.

	Argument/s:
		file_dir - directory containing the audio files.
		list_name - name for the list.
		data_path - path to store pickle files.
		make_new - re-create list.

	Returns:
		batch_list - list of file paths and wav length.
	"""
	extension = ['*.wav', '*.flac', '*.mp3']
	if not make_new:
		if os.path.exists(data_path + '/' + list_name + '_list_' + platform.node() + '.p'):
			print('Loading ' + list_name + ' list...')
			with open(data_path + '/' + list_name + '_list_' + platform.node() + '.p', 'rb') as f:
				batch_list = pickle.load(f)
			if batch_list[0]['file_path'].find(file_dir) != -1:
				print(list_name + ' list has a total of %i entries.' % (len(batch_list)))
				return batch_list

	print('Creating ' + list_name + ' list...')
	batch_list = []
	for i in extension:
		for j in glob.glob(os.path.join(file_dir, i)):
			f = SoundFile(j)
			wav_len = f.seek(0, SEEK_END)
			if wav_len == -1:
				wav, _ = read_wav(path)
				wav_len = len(wav)
			batch_list.append({'file_path': j, 'wav_len': wav_len}) # append dictionary.
	if not os.path.exists(data_path): os.makedirs(data_path) # make directory.
	with open(data_path + '/' + list_name + '_list_' + platform.node() + '.p', 'wb') as f:
		pickle.dump(batch_list, f)
	print('The ' + list_name + ' list has a total of %i entries.' % (len(batch_list)))
	return batch_list
