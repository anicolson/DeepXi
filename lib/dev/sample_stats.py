## FILE:           sample_stats.py
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
## BRIEF:          Get statistics from sample of the training set.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
import tensorflow as tf
import os, pickle, random
import dev.se_batch as batch
from dev.acoustic.feat import polar
from tqdm import tqdm
import scipy.io as spio

## GET STATISTICS OF SAMPLE
def get_stats(stats_path, args, config):
	if os.path.exists(stats_path + '/stats.p'):
		print('Loading sample statistics from pickle file...')
		with open(stats_path + '/stats.p', 'rb') as f:
			args.stats = pickle.load(f)
		return args
	elif args.infer:
		raise ValueError('You have not completed training (no stats.p file exsists). In the Deep Xi github repository, data/stats.p is available.')
	else:
		print('Finding sample statistics...')
		random.shuffle(args.train_s_list) # shuffle list.
		s_sample, s_sample_seq_len = batch.Clean_mbatch(args.train_s_list, 
			args.sample_size, 0, args.sample_size) # generate mini-batch of clean training waveforms.
		d_sample, d_sample_seq_len = batch.Noise_mbatch(args.train_d_list, 
			args.sample_size, s_sample_seq_len) # generate mini-batch of noise training waveforms.
		snr_sample = np.random.randint(args.min_snr, args.max_snr + 1, args.sample_size) # generate mini-batch of SNR levels.
		s_ph = tf.placeholder(tf.int16, shape=[None, None], name='s_ph') # clean speech placeholder.
		d_ph = tf.placeholder(tf.int16, shape=[None, None], name='d_ph') # noise placeholder.
		s_len_ph = tf.placeholder(tf.int32, shape=[None], name='s_len_ph') # clean speech sequence length placeholder.
		d_len_ph = tf.placeholder(tf.int32, shape=[None], name='d_len_ph') # noise sequence length placeholder.
		snr_ph = tf.placeholder(tf.float32, shape=[None], name='snr_ph') # SNR placeholder.
		analysis = polar.target_xi(s_ph, d_ph, s_len_ph, d_len_ph, snr_ph, args.N_w, args.N_s, args.NFFT, args.f_s)
		sample_graph = analysis[0]
		samples = []
		with tf.Session(config=config) as sess:
			for i in tqdm(range(s_sample.shape[0])):
				sample = sess.run(sample_graph, feed_dict={s_ph: [s_sample[i]], d_ph: [d_sample[i]], s_len_ph: [s_sample_seq_len[i]], 
					d_len_ph: [d_sample_seq_len[i]], snr_ph: [snr_sample[i]]}) # sample of training set.
				samples.append(sample)
		samples = np.vstack(samples)
		if len(samples.shape) != 2: ValueError('Incorrect shape for sample.')
		args.stats = {'mu_hat': np.mean(samples, axis=0), 'sigma_hat': np.std(samples, axis=0)}
		if not os.path.exists(stats_path): os.makedirs(stats_path) # make directory.
		with open(stats_path + '/stats.p', 'wb') as f: 		
			pickle.dump(args.stats, f)
		spio.savemat(stats_path + '/stats.m', mdict={'mu_hat': args.stats['mu_hat'], 'sigma_hat': args.stats['sigma_hat']})
		print('Sample statistics saved to pickle file.')
	return args
