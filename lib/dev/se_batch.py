## FILE:           se_batch.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Generates mini-batches, creates training, and test lists for speech enhancement.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import contextlib, glob, os, pickle, platform, random, sys, wave
import numpy as np
from dev.utils import read_wav
from scipy.io.wavfile import read

def Batch_list(file_dir, list_name, data_path=None, make_new=False):
	from soundfile import SoundFile, SEEK_END
	'''
	Places the file paths and wav lengths of an audio file into a dictionary, which 
	is then appended to a list. SPHERE format cannot be used. 'glob' is used to 
	support Unix style pathname pattern expansions. Checks if the training list 
	has already been pickled, and loads it. If a different dataset is to be 
	used, delete the pickle file.

	Inputs:
		file_dir - directory containing the wavs.
		list_name - name for the list.
		data_path - path to store pickle files.
		make_new - re-create list.

	Outputs:
		batch_list - list of file paths and wav length.
	'''
	file_name = ['*.wav', '*.flac', '*.mp3']
	if data_path == None: data_path = 'data'
	if not make_new:
		if os.path.exists(data_path + '/' + list_name + '_list_' + platform.node() + '.p'):
			print('Loading ' + list_name + ' list from pickle file...')
			with open(data_path + '/' + list_name + '_list_' + platform.node() + '.p', 'rb') as f:
				batch_list = pickle.load(f)
			if batch_list[0]['file_path'].find(file_dir) != -1: 
				print('The ' + list_name + ' list has a total of %i entries.' % (len(batch_list)))
				return batch_list
	print('Creating ' + list_name + ' list, as no pickle file exists...')
	batch_list = [] # list for wav paths and lengths.
	for fn in file_name:
		for file_path in glob.glob(os.path.join(file_dir, fn)):
			f = SoundFile(file_path)
			seq_len = f.seek(0, SEEK_END)
			batch_list.append({'file_path': file_path, 'seq_len': seq_len}) # append dictionary.
	if not os.path.exists(data_path): os.makedirs(data_path) # make directory.
	with open(data_path + '/' + list_name + '_list_' + platform.node() + '.p', 'wb') as f: 		
		pickle.dump(batch_list, f)
	print('The ' + list_name + ' list has a total of %i entries.' % (len(batch_list)))
	return batch_list

def Clean_mbatch(clean_list, mbatch_size, start_idx, end_idx):
	'''
	Creates a padded mini-batch of clean speech wavs.

	Inputs:
		clean_list - training list for the clean speech files.
		mbatch_size - size of the mini-batch.
		version - version name.

	Outputs:
		mbatch - matrix of paded wavs stored as a numpy array.
		seq_len - length of each wavs strored as a numpy array.
		clean_list - training list for the clean files.
	'''
	mbatch_list	= clean_list[start_idx:end_idx] # get mini-batch list from training list.
	maxlen = max([dic['seq_len'] for dic in mbatch_list]) # find maximum length wav in mini-batch.
	seq_len = [] # list of the wavs lengths.
	mbatch = np.zeros([len(mbatch_list), maxlen], np.int16) # numpy array for wav matrix.
	for i in range(len(mbatch_list)):
		(wav, _) = read_wav(mbatch_list[i]['file_path']) # read wav from given file path.		
		mbatch[i,:mbatch_list[i]['seq_len']] = wav # add wav to numpy array.
		seq_len.append(mbatch_list[i]['seq_len']) # append length of wav to list.
	return mbatch, np.array(seq_len, np.int32)

def Noise_mbatch(noise_list, mbatch_size, clean_seq_len):
	'''
	Creates a padded mini-batch of noise speech wavs.

	Inputs:
		noise_list - training list for the noise files.
		mbatch_size - size of the mini-batch.
		clean_seq_len - sequence length of each clean speech file in the mini-batch.

	Outputs:
		mbatch - matrix of paded wavs stored as a numpy array.
		seq_len - length of each wavs strored as a numpy array.
	'''
	
	mbatch_list	= random.sample(noise_list, mbatch_size) # get mini-batch list from training list.
	for i in range(len(clean_seq_len)):
		flag = True
		while flag:
			if mbatch_list[i]['seq_len'] < clean_seq_len[i]:
				mbatch_list[i] = random.choice(noise_list)
			else:
				flag = False
	maxlen = max([dic['seq_len'] for dic in mbatch_list]) # find maximum length wav in mini-batch.
	seq_len = [] # list of the wav lengths.
	mbatch = np.zeros([len(mbatch_list), maxlen], np.int16) # numpy array for wav matrix.
	for i in range(len(mbatch_list)):
		(wav, _) = read_wav(mbatch_list[i]['file_path']) # read wav from given file path.
		mbatch[i,:mbatch_list[i]['seq_len']] = wav # add wav to numpy array.
		seq_len.append(mbatch_list[i]['seq_len']) # append length of wav to list.
	return mbatch, np.array(seq_len, np.int32)

def Batch(fdir, snr_l):
	'''
	REQUIRES REWRITING.

	Places all of the test waveforms from the list into a numpy array. 
	SPHERE format cannot be used. 'glob' is used to support Unix style pathname 
	pattern expansions. Waveforms are padded to the maximum waveform length. The 
	waveform lengths are recorded so that the correct lengths can be sliced 
	for feature extraction. The SNR levels of each test file are placed into a
	numpy array. Also returns a list of the file names.

	Inputs:
		fdir - directory containing the waveforms.
		fnames - filename/s of the waveforms.
		snr_l - list of the SNR levels used.

	Outputs:
		wav_np - matrix of paded waveforms stored as a numpy array.
		len_np - length of each waveform strored as a numpy array.
		snr_test_np - numpy array of all the SNR levels for the test set.
		fname_l - list of filenames.
	'''
	fname_l = [] # list of file names.
	wav_l = [] # list for waveforms.
	snr_test_l = [] # list of SNR levels for the test set.
	# if isinstance(fnames, str): fnames = [fnames] # if string, put into list.
	fnames = ['*.wav', '*.flac', '*.mp3']
	for fname in fnames:
		for fpath in glob.glob(os.path.join(fdir, fname)):
			for snr in snr_l:
				if fpath.find('_' + str(snr) + 'dB') != -1:
					snr_test_l.append(snr) # append SNR level.	
			(wav, _) = read_wav(fpath) # read waveform from given file path.
			if np.isnan(wav).any() or np.isinf(wav).any():
				raise ValueError('Error: NaN or Inf value. File path: %s.' % (file_path))
			wav_l.append(wav) # append.
			fname_l.append(os.path.basename(os.path.splitext(fpath)[0])) # append name.
	len_l = [] # list of the waveform lengths.
	maxlen = max(len(wav) for wav in wav_l) # maximum length of waveforms.
	wav_np = np.zeros([len(wav_l), maxlen], np.int16) # numpy array for waveform matrix.
	for (i, wav) in zip(range(len(wav_l)), wav_l):
		wav_np[i,:len(wav)] = wav # add waveform to numpy array.
		len_l.append(len(wav)) # append length of waveform to list.
	return wav_np, np.array(len_l, np.int32), np.array(snr_test_l, np.int32), fname_l
