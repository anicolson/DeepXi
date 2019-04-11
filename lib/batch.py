## FILE:           batch.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Generates mini-batches. Creates training, and test lists.
##
## Copyright (C) 2019  Aaron Nicolson
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.io.wavfile import read
import glob, os, pickle, random, sys

def _clean_mbatch(clean_list, mbatch_size, start_idx, end_idx):
	'''
	Creates a padded mini-batch of clean speech signals.

	Inputs:
		clean_list - training list for the clean speech files.
		mbatch_size - size of the mini-batch.
		version - version name.

	Outputs:
		mbatch - matrix of paded signals stored as a numpy array.
		seq_len - length of each signals strored as a numpy array.
		clean_list - training list for the clean files.
	'''
	mbatch_list	= clean_list[start_idx:end_idx] # get mini-batch list from training list.
	maxlen = max([dic['seq_len'] for dic in mbatch_list]) # find maximum length signal in mini-batch.
	seq_len = [] # list of the signals lengths.
	mbatch = np.zeros([len(mbatch_list), maxlen], np.int16) # numpy array for signal matrix.
	for i in range(len(mbatch_list)):
		(_, sig) = read(mbatch_list[i]['file_path']) # read signal from given file path.		
		mbatch[i,:mbatch_list[i]['seq_len']] = sig # add signal to numpy array.
		seq_len.append(mbatch_list[i]['seq_len']) # append length of signal to list.
	return mbatch, np.array(seq_len, np.int32)

def _noise_mbatch(noise_list, mbatch_size, clean_seq_len):
	'''
	Creates a padded mini-batch of noise speech signals.

	Inputs:
		noise_list - training list for the noise files.
		mbatch_size - size of the mini-batch.
		clean_seq_len - sequence length of each clean speech file in the mini-batch.

	Outputs:
		mbatch - matrix of paded signals stored as a numpy array.
		seq_len - length of each signals strored as a numpy array.
	'''
	
	mbatch_list	= random.sample(noise_list, mbatch_size) # get mini-batch list from training list.
	for i in range(len(clean_seq_len)):
		flag = True
		while flag:
			if mbatch_list[i]['seq_len'] < clean_seq_len[i]:
				mbatch_list[i] = random.choice(noise_list)
			else:
				flag = False
	maxlen = max([dic['seq_len'] for dic in mbatch_list]) # find maximum length signal in mini-batch.
	seq_len = [] # list of the signal lengths.
	mbatch = np.zeros([len(mbatch_list), maxlen], np.int16) # numpy array for signal matrix.
	for i in range(len(mbatch_list)):
		(_, sig) = read(mbatch_list[i]['file_path']) # read signal from given file path.
		mbatch[i,:mbatch_list[i]['seq_len']] = sig # add signal to numpy array.
		seq_len.append(mbatch_list[i]['seq_len']) # append length of signal to list.
	return mbatch, np.array(seq_len, np.int32)

def _train_list(file_dir, file_name, name):
	'''
	Places the file paths and signal lengths of a file into a dictionary, which 
	is then appended to a list. SPHERE format cannot be used. 'glob' is used to 
	support Unix style pathname pattern expansions. Checks if the training list 
	has already been pickled, and loads it. If a different dataset is to be 
	used, delete the pickle file.

	Inputs:
		file_dir - directory containing the signals.
		file_name - filename of the signals.
		name - name of the training list.

	Outputs:
		train_list - list of file paths and signal length.
	'''
	if os.path.exists('data/' + name + '_list.p'):
		print('Loading ' + name + ' list from pickle file...')
		with open('data/' + name + '_list.p', 'rb') as f:
			train_list = pickle.load(f)
	else:
		print('Creating ' + name + ' list, as no pickle file exists...')
		train_list = [] # list for signal paths and lengths.
		for file_path in glob.glob(os.path.join(file_dir, file_name)):
			(_, sig) = read(file_path) # read signal from given file path.
			if np.isnan(sig).any() or np.isinf(sig).any():
				raise ValueError('Error: NaN or Inf value. File path: %s.' % (file_path))
			train_list.append({'file_path': file_path, 'seq_len': len(sig)}) # append dictionary.
		if not os.path.exists('data'): os.makedirs('data') # make directory.
		with open('data/' + name + '_list.p', 'wb') as f: 		
			pickle.dump(train_list, f)
	print('The ' + name + ' set has a total of %i files.' % (len(train_list)))
	random.shuffle(train_list) # shuffle list.
	return train_list

def _batch(fdir, fnames, snr_l):
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
	if isinstance(fnames, str): fnames = [fnames] # if string, put into list.
	for fname in fnames:
		for fpath in glob.glob(os.path.join(fdir, fname)):
			for snr in snr_l:
				if fpath.find('_' + str(snr) + 'dB') != -1:
					snr_test_l.append(snr) # append SNR level.	
			(_, wav) = read(fpath) # read waveform from given file path.
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


def Mov_avg(avg, new_sample):
	'''
	Moving average.

	Inputs:
		avg - moving average.
		new_sample - new sample.

	Outputs:
		avg - moving average.
	'''
	avg -= avg/16
	avg += new_sample/16
	return avg
