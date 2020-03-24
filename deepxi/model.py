## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

# from dev.acoustic.analysis_synthesis.polar import synthesis
# from dev.acoustic.feat import polar
# from dev.ResNet import ResNet
# import dev.optimisation as optimisation
# import numpy as np
# import tensorflow as tf

class DeepXi(DeepXiInput):
	"""
	Deep Xi
	"""
	def __init__(self, N_w, N_s, NFFT, f_s, mu=None, sigma=None, save_dir=None):
		"""
		Argument/s
			Nw - window length (samples).
			Ns - window shift (samples).
			NFFT - number of DFT bins.
			f_s - sampling frequency.
			mu - sample mean of each instantaneous a priori SNR in dB frequency component.
			sigma - sample standard deviation of each instantaneous a priori SNR in dB frequency component.
			save_dir - directory to save model.
		"""
		super().__init__(N_w, N_s, NFFT, f_s, mu, sigma)
		self.save_dir = save_dir
		self.n_inp = math.ceil(self.NFFT/2 + 1)
		self.n_outp = self.n_inp
		self.inp = Input(name='inp', shape=[None, self.n_inp], dtype='float32')

		if args.network == 'TCN': self.network = TCN(self.inp, self.n_outp, B=40, d_model=256, d_f=64, k=3, max_d_rate=16, softmax=False)
		else: raise ValueError('Invalid network type.')

		self.opt = Adam()
		self.model = Model(inputs=self.inp, outputs=self.network)
		self.model.summary()
		if self.save_dir == None: self.save_dir = 'model'
		if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
		with open(self.save_dir + "/model.json", "w") as json_file: json_file.write(self.model.to_json())

	def compile():
		"""
		"""
		self.model.compile(loss='binary_crossentropy', optimizer=self.opt)

	def save_weights(self, epoch):
		""" 
		"""
		self.model.save_weights(self.save_dir + "/epoch-" + str(epoch))

	def load_weights(self, epoch):
		""" 
		"""
		self.model.load_weights(self.save_dir + "/epoch-" + str(epoch))

	def get_stats(stats_path, args, config):
		"""
		"""
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

	def train():
		"""
		"""

		train_file_path_list, max_frame_len, max_char_len = examples_list(args.train_list, feat,
			'train_clean_100_' + str(args.blank_idx) + '_' + args.feat_type + '_' + str(args.M), args.data_path)
		x_val, y_val, x_val_len, y_val_len = batch(args.val_list, feat, 
			'dev_clean_' + str(args.blank_idx) + '_' + args.feat_type + '_' + str(args.M), args.data_path)
		# train_file_path_list, max_frame_len, max_char_len = examples_list(args.val_list, feat,
		# 	'dev_clean_' + str(args.blank_idx) + '_' + args.feat_type + '_' + str(args.M), args.data_path)

		train_dataset = model.dataset(train_file_path_list, args.mbatch_size, max_frame_len, max_char_len)

		if args.cont: 
			model.load_weights(args.model_path, args.epoch)
			start_epoch = args.epoch + 1
		else: start_epoch = 1

		if args.lr_scheduler: callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler)]
		else: callbacks = None

		if not os.path.exists('log'): os.makedirs('log') # create log directory.
		with open("log/" + args.ver + ".csv", "a") as results:
			results.write("Epoch, Train loss, Val. loss, Val. CER, D/T\n")

		pbar = tqdm(total=args.max_epochs, desc='Training E' + str(start_epoch))
		pbar.update(start_epoch-1)
		for i in range(start_epoch, args.max_epochs+1):
			history = model.fit(train_dataset, initial_epoch=i-1, epochs=i, steps_per_epoch=model.n_iter)
			val_loss = model.loss(x_val, y_val, x_val_len, y_val_len, batch_size=args.mbatch_size)
			likelihood = model.output([x_val], batch_size=args.mbatch_size)
			_, cer = model.greedy_decode_metrics(likelihood, y_val, x_val_len, y_val_len, args.idx2char)
			pbar.set_description_str("E%d train|val loss: %.2f|%.2f, val CER: %.2f%%" % (i, 
				history.history['loss'][0], val_loss, 100*cer))
			pbar.update(); pbar.refresh()
			with open("log/" + args.ver + ".csv", "a") as results:
				results.write("%d, %.2f, %.2f, %.2f, %s\n" % (i, 
					history.history['loss'][0], val_loss, 100*cer,
					datetime.now().strftime('%Y-%m-%d/%H:%M:%S')))
			model.save_weights(args.model_path, i)
			

	def infer(): 
		"""
		"""
		pass

	def mbatch_gen(self, file_path_list, mbatch_size, max_frame_len, max_char_len): 
		"""
		"""
		random.shuffle(file_path_list)
		start_idx = 0; end_idx = mbatch_size
		for _ in range(self.n_iter):
			mbatch_file_path_list = file_path_list[start_idx:end_idx]
			n_examples_mbatch = end_idx - start_idx
			x_batch = np.zeros((n_examples_mbatch, max_frame_len, self.n_feat), np.float32)
			y_batch = np.zeros((n_examples_mbatch, max_char_len), np.int32)
			x_batch_len = np.zeros((n_examples_mbatch, 1), np.int32)
			y_batch_len = np.zeros((n_examples_mbatch, 1), np.int32)
			y_dummy =  np.zeros((n_examples_mbatch, 1), np.int32)
			for (i,j) in zip(mbatch_file_path_list, range(n_examples_mbatch)): 
				with np.load(i) as data:
					x_batch[j,:,:] = data['x_example']
					y_batch[j,:] = data['y_example']
					x_batch_len[j,0] = data['x_example_len']
					y_batch_len[j,0] = data['y_example_len']
			start_idx += mbatch_size; end_idx += mbatch_size
			if end_idx > self.n_examples_batch: end_idx = self.n_examples_batch
			yield {"inp": x_batch, "tgt": y_batch, "inp_len": x_batch_len, "tgt_len": y_batch_len}, y_dummy




		# ## RESNET		
		# self.input_ph = tf.placeholder(tf.float32, shape=[None, None, args.d_in], name='input_ph') # noisy speech MS placeholder.
		# self.nframes_ph = tf.placeholder(tf.int32, shape=[None], name='nframes_ph') # noisy speech MS sequence length placeholder.
		# if args.model == 'ResNet':
		# 	self.output = ResNet(self.input_ph, self.nframes_ph, args.norm_type, n_blocks=args.n_blocks, boolean_mask=True, d_out=args.d_out, 
		# 		d_model=args.d_model, d_f=args.d_f, k_size=args.k_size, max_d_rate=args.max_d_rate)
		# elif args.model == 'RDLNet':
		# 	from dev.RDLNet import RDLNet
		# 	self.output = RDLNet(self.input_ph, self.nframes_ph, args.norm_type, n_blocks=args.n_blocks, boolean_mask=True, d_out=args.d_out, 
		# 		d_f=args.d_f, net_height=args.net_height)
		# elif args.model == 'ResLSTM':
		# 	from dev.ResLSTM import ResLSTM
		# 	self.output = ResLSTM(self.input_ph, self.nframes_ph, args.norm_type, n_blocks=args.n_blocks, boolean_mask=True, d_out=args.d_out, d_model=args.d_model)

		# ## TRAINING FEATURE EXTRACTION GRAPH
		# self.s_ph = tf.placeholder(tf.int16, shape=[None, None], name='s_ph') # clean speech placeholder.
		# self.d_ph = tf.placeholder(tf.int16, shape=[None, None], name='d_ph') # noise placeholder.
		# self.s_len_ph = tf.placeholder(tf.int32, shape=[None], name='s_len_ph') # clean speech sequence length placeholder.
		# self.d_len_ph = tf.placeholder(tf.int32, shape=[None], name='d_len_ph') # noise sequence length placeholder.
		# self.snr_ph = tf.placeholder(tf.float32, shape=[None], name='snr_ph') # SNR placeholder.
		# self.train_feat = polar.input_target_xi(self.s_ph, self.d_ph, self.s_len_ph, 
		# 	self.d_len_ph, self.snr_ph, args.N_w, args.N_s, args.NFFT, args.f_s, args.stats['mu_hat'], args.stats['sigma_hat'])

		# ## INFERENCE FEATURE EXTRACTION GRAPH
		# self.infer_feat = polar.input(self.s_ph, self.s_len_ph, args.N_w, args.N_s, args.NFFT, args.f_s)

		# ## PLACEHOLDERS
		# self.x_ph = tf.placeholder(tf.int16, shape=[None, None], name='x_ph') # noisy speech placeholder.
		# self.x_len_ph = tf.placeholder(tf.int32, shape=[None], name='x_len_ph') # noisy speech sequence length placeholder.
		# self.target_ph = tf.placeholder(tf.float32, shape=[None, args.d_out], name='target_ph') # training target placeholder.
		# self.keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph') # keep probability placeholder.
		# self.training_ph = tf.placeholder(tf.bool, name='training_ph') # training placeholder.

		# ## SYNTHESIS GRAPH
		# if args.infer:	
		# 	self.infer_output = tf.nn.sigmoid(self.output)
		# 	self.y_MAG_ph = tf.placeholder(tf.float32, shape=[None, None, args.d_in], name='y_MAG_ph') 
		# 	self.x_PHA_ph = tf.placeholder(tf.float32, [None, None, args.d_in], name='x_PHA_ph')
		# 	self.y = synthesis(self.y_MAG_ph, self.x_PHA_ph, args.N_w, args.N_s, args.NFFT)

		# ## LOSS & OPTIMIZER
		# self.loss = optimisation.loss(self.target_ph, self.output, 'mean_sigmoid_cross_entropy', axis=[1])
		# self.total_loss = tf.reduce_mean(self.loss, axis=0)
		# self.trainer, _ = optimisation.optimiser(self.total_loss, optimizer='adam', grad_clip=True)

		# ## SAVE VARIABLES
		# self.saver = tf.train.Saver(max_to_keep=256)

		# ## NUMBER OF PARAMETERS
		# args.params = (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
