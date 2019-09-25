## FILE:           optimistion.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Loss functions and algorithms for gradient descent optimisation.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import tensorflow as tf

## LOSS FUNCTIONS
def loss(target, estimate, loss_fnc, axis=1):
	'loss functions for gradient descent.'
	with tf.name_scope(loss_fnc + '_loss'):
		if loss_fnc == 'quadratic':
			loss = tf.reduce_sum(tf.square(tf.subtract(target, estimate)), axis=axis)
		if loss_fnc == 'sigmoid_cross_entropy':
			loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=estimate), axis=axis)
		if loss_fnc == 'mean_quadratic':
			loss = tf.reduce_mean(tf.square(tf.subtract(target, estimate)), axis=axis)
		if loss_fnc == 'mean_sigmoid_cross_entropy':
			loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=estimate), axis=axis)
		return loss

## GRADIENT DESCENT OPTIMISERS
def optimiser(loss, lr=None, epsilon=None, var_list=None, optimizer='adam', grad_clip=False):
	'optimizers for training.'
	with tf.name_scope(optimizer + '_opt'):
		if optimizer == 'adam':
			if lr == None: lr = 0.001 # default.
			if epsilon == None: epsilon = 1e-8 # default.
			optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
		if optimizer == 'lazyadam':
			if lr == None: lr = 0.001 # default.
			if epsilon == None: epsilon = 1e-8 # default.
			optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=lr, epsilon=epsilon)
		if optimizer == 'nadam':
			if lr == None: lr = 0.001 # default.
			if epsilon == None: epsilon = 1e-8 # default.
			optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, epsilon=epsilon)
		if optimizer == 'sgd':
			if lr == None: lr = 0.5 # default.
			optimizer = tf.train.GradientDescentOptimizer(lr)
		grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
		if grad_clip: grads_and_vars = [(tf.clip_by_value(gv[0], -1., 1.), gv[1]) for gv in grads_and_vars]
		trainer = optimizer.apply_gradients(grads_and_vars)

	return trainer, optimizer