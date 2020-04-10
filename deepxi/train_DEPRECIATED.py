## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import dev.se_batch as batch
import numpy as np
import math, os, random
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm

## TRAINING
def train(sess, net, args):
	print("Training...")
	random.shuffle(args.train_s_list) # shuffle training list.

	## CONTINUE FROM LAST EPOCH
	if args.cont:
		epoch_size = len(args.train_s_list); epoch_comp = args.epoch; start_idx = 0; 
		end_idx = args.mbatch_size; val_error = float("inf") # create epoch parameters.
		net.saver.restore(sess, args.model_path + '/epoch-' + str(args.epoch)) # load model from last epoch.

	## TRAIN RAW NETWORK
	else:
		epoch_size = len(args.train_s_list); epoch_comp = 0; start_idx = 0;
		end_idx = args.mbatch_size; val_error = float("inf") # create epoch parameters.
		if args.mbatch_size > epoch_size: raise ValueError('Error: mini-batch size is greater than the epoch size.')
		sess.run(tf.global_variables_initializer()) # initialise model variables.
		net.saver.save(sess, args.model_path + '/epoch', global_step=epoch_comp) # save model.

	## TRAINING LOG
	if not os.path.exists('log'): os.makedirs('log') # create log directory.
	with open("log/" + args.ver + ".csv", "a") as results: results.write("'"'Validation error'"', '"'Training error'"', '"'Epoch'"', '"'D/T'"'\n")

	train_err = 0; mbatch_count = 0
	while args.train:

		print('Training E%d (ver=%s, gpu=%s, params=%g)...' % (epoch_comp + 1, args.ver, args.gpu, args.params))
		for _ in tqdm(range(args.train_steps)):

			## MINI-BATCH GENERATION
			mbatch_size_iter = end_idx - start_idx # number of examples in mini-batch for the training iteration.
			s_mbatch, s_mbatch_seq_len = batch.Clean_mbatch(args.train_s_list, 
				mbatch_size_iter, start_idx, end_idx) # generate mini-batch of clean training waveforms.
			d_mbatch, d_mbatch_seq_len = batch.Noise_mbatch(args.train_d_list, 
				mbatch_size_iter, s_mbatch_seq_len) # generate mini-batch of noise training waveforms.
			snr_mbatch = np.random.randint(args.min_snr, args.max_snr + 1, end_idx - start_idx) # generate mini-batch of SNR levels.

			## TRAINING ITERATION
			mbatch = sess.run(net.train_feat, feed_dict={net.s_ph: s_mbatch, net.d_ph: d_mbatch, 
				net.s_len_ph: s_mbatch_seq_len, net.d_len_ph: d_mbatch_seq_len, net.snr_ph: snr_mbatch}) # mini-batch.
			[_, mbatch_err] = sess.run([net.trainer, net.total_loss], feed_dict={net.input_ph: mbatch[0], net.target_ph: mbatch[1], 
			net.nframes_ph: mbatch[2], net.training_ph: True}) # training iteration.						
			if not math.isnan(mbatch_err):
				train_err += mbatch_err; mbatch_count += 1

			## UPDATE EPOCH PARAMETERS
			start_idx += args.mbatch_size; end_idx += args.mbatch_size # start and end index of mini-batch.
			if end_idx > epoch_size: end_idx = epoch_size # if less than the mini-batch size of examples is left.

		## VALIDATION SET ERROR
		start_idx = 0; end_idx = args.mbatch_size # reset start and end index of mini-batch.
		random.shuffle(args.train_s_list) # shuffle list.
		start_idx = 0; end_idx = args.mbatch_size; frames = 0; val_error = 0; # validation variables.
		print('Validation error for E%d...' % (epoch_comp + 1))
		for _ in tqdm(range(args.val_steps)):
			mbatch = sess.run(net.train_feat, feed_dict={net.s_ph: args.val_s[start_idx:end_idx], 
				net.d_ph: args.val_d[start_idx:end_idx], net.s_len_ph: args.val_s_len[start_idx:end_idx], 
				net.d_len_ph: args.val_d_len[start_idx:end_idx], net.snr_ph: args.val_snr[start_idx:end_idx]}) # mini-batch.
			val_error_mbatch = sess.run(net.loss, feed_dict={net.input_ph: mbatch[0], 
				net.target_ph: mbatch[1], net.nframes_ph: mbatch[2], net.training_ph: False}) # validation error for each frame in mini-batch.
			val_error += np.sum(val_error_mbatch)
			frames += mbatch[1].shape[0] # total number of frames.
			print("Validation error for Epoch %d: %3.3f%% complete.       " % 
				(epoch_comp + 1, 100*(end_idx/args.val_s_len.shape[0])), end="\r")
			start_idx += args.mbatch_size; end_idx += args.mbatch_size
			if end_idx > args.val_s_len.shape[0]: end_idx = args.val_s_len.shape[0]
		val_error /= frames # validation error.
		epoch_comp += 1 # an epoch has been completed.
		net.saver.save(sess, args.model_path + '/epoch', global_step=epoch_comp) # save model.
		print("E%d: train err=%3.3f, val err=%3.3f.           " % 
			(epoch_comp, train_err/mbatch_count, val_error))
		with open("log/" + args.ver + ".csv", "a") as results:
			results.write("%g, %g, %d, %s\n" % (val_error, train_err/mbatch_count,
			epoch_comp, datetime.now().strftime('%Y-%m-%d/%H:%M:%S')))
		train_err = 0; mbatch_count = 0; start_idx = 0; end_idx = args.mbatch_size

		if epoch_comp >= args.max_epochs:
			args.train = False
			print('\nTraining complete. Validation error for epoch %d: %g.                 ' % 
				(epoch_comp, val_error))