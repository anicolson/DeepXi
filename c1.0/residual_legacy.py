import argparse
import tensorflow as tf
 
def Residual(input, seq_len, keep_prob, num_outputs, args):
    '''
    Residual Network. Can have CNN-based blocks, or LSTM-based blocks.
    '''
 
    ## RESNET
    network = [input] # input features.
    blocks = 0; # number of blocks.
    if args.conv_caus: args.padding = 'valid'
    else: args.padding = 'same'
 
    ## RESIDUAL BLOCKS
    for i in range(len(args.blocks)):   
        start_layer = len(network) - 1 # starting index of block.
 
        if args.blocks[i] == 'I1': # CNN input layer.
            with tf.variable_scope('I1'):
                c = network[-1]
                if args.conv_caus:
                    c = tf.concat([tf.zeros([tf.shape(c)[0], args.conv_size - 1, tf.shape(c)[2]]), c], 1)
                network.append(tf.multiply(tf.layers.conv1d(c, args.conv_filt, args.conv_size, 
                    padding=args.padding), tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32))) 
 
        elif args.blocks[i] == 'I2': # FC input layer.
            with tf.variable_scope('I2'):
                network.append(tf.multiply(tf.layers.dense(network[-1], args.input_size, tf.nn.relu, 
                    bias_initializer=tf.constant_initializer([0.1]*args.input_size)), 
                    tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32))) # FC.
 
        elif args.blocks[i] == 'C1': # 1x1 conv adapter.
            with tf.variable_scope('C1'):
                network.append(tf.multiply(tf.layers.conv1d(masked_layer_norm(tf.nn.relu(network[-1]), 
                    seq_len), args.cell_size, 1), tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32))) # 1x1 conv adapter.
 
        elif args.blocks[i] == 'C2': # jump residual connection.
            with tf.variable_scope('C2'):
                network.append(tf.concat([masked_layer_norm(tf.nn.relu(network[-1]), 
                    seq_len), network[0]], 2)) # jump residual connection.
 
        elif args.blocks[i] == 'C3': # ReLU + LN.
            with tf.variable_scope('C3'):
                network.append(masked_layer_norm(tf.nn.relu(network[-1]), seq_len))
 
        elif args.blocks[i] == 'O': # Output layer.
            with tf.variable_scope('O'):
                O = tf.boolean_mask(network[-1], tf.sequence_mask(seq_len))
                network.append(tf.layers.dense(O, num_outputs))
 
        elif args.blocks[i] == 'B1': # RNN block type 1.
            blocks += 1
            with tf.variable_scope('B1_' + str(blocks)):
                B1 = network[-1]
                B1 = masked_layer_norm(B1, seq_len) # LN.
                args.name = args.cell_type + '1'
                B1 = rnn_layer(B1, seq_len, args) # RNN 1.
                if args.conv_caus:
                    B1 = tf.concat([tf.zeros([tf.shape(B1)[0], args.conv_size - 1, 
                        tf.shape(B1)[2]]), B1], 1)
                B1 = tf.multiply(tf.layers.conv1d(B1, args.conv_filt, 
                    args.conv_size, padding=args.padding), 
                    tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), 
                    tf.float32)) # W1.
                B1 = masked_layer_norm(B1, seq_len) # LN.
                args.name = args.cell_type + '2'
                B1 = rnn_layer(B1, seq_len, args) # RNN 2.
                network.append(tf.multiply(tf.layers.conv1d(B1, args.cell_size, 1), 
                    tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32))) # 1x1 conv.
 
        elif args.blocks[i] == 'B2': # 1D CNN block.
            blocks += 1
            with tf.variable_scope('B2_' + str(blocks)):
                B2 = masked_layer_norm(tf.nn.relu(network[-1]), seq_len)
                if args.conv_caus:
                    B2 = tf.concat([tf.zeros([tf.shape(B2)[0], args.conv_size - 1, 
                        tf.shape(B2)[2]]), B2], 1)
                B2 = tf.multiply(tf.layers.conv1d(B2, args.conv_filt, args.conv_size, 
                    bias_initializer=tf.constant_initializer(0.01), padding=args.padding), 
                    tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32))
                B2 = masked_layer_norm(tf.nn.relu(B2), seq_len)
                if args.dropout:
                    B2 = tf.nn.dropout(B2, keep_prob)
                if args.conv_caus:
                    B2 = tf.concat([tf.zeros([tf.shape(B2)[0], args.conv_size - 1, 
                        tf.shape(B2)[2]]), B2], 1)
                network.append(tf.multiply(tf.layers.conv1d(B2, args.conv_filt, args.conv_size, 
                    bias_initializer=tf.constant_initializer(0.01), padding=args.padding),
                    tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32)))
 
        ## LSTM BLOCK
        elif args.blocks[i] == 'B3': # RNN block type 2.
            blocks += 1
            with tf.variable_scope('B3_' + str(blocks)):
                B3 = network[-1]
                if args.layer_norm:
                    B3 = masked_layer_norm(B3, seq_len) # LN.
                args.name = args.cell_type
                network.append(rnn_layer(B3, seq_len, args)) # RNN.
 
 
        else: # block type does not exist.
            raise ValueError('Block type does not exist: %s.' % (args.blocks[i]))
 
        ## RESIDUAL CONNECTION 
        if args.res_con == 'add':
            if network[-1].get_shape().as_list()[-1] == network[start_layer].get_shape().as_list()[-1]:
                with tf.variable_scope('add_L' + str(len(network)-1) + '_L' + str(start_layer)):
                    network.append(tf.add(network[-1], network[start_layer])) # residual connection.
        elif args.res_con == 'concat':
            with tf.variable_scope('concat_L' + str(len(network)-1) + '_L' + str(start_layer)):
                network.append(tf.concat([network[-1], network[start_layer]], 2)) # residual connection.
        elif args.res_con == 'proj_concat':
            with tf.variable_scope('concat_L' + str(len(network)-1) + '_L' + str(start_layer)):
                p = tf.multiply(tf.layers.dense(network[start_layer], args.res_proj, use_bias = False), 
                    tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32)) # projected residual connection.
                network.append(tf.concat([network[-1], p], 2)) # residual connection.
        elif args.res_con == 'concat_proj':
            with tf.variable_scope('concat_L' + str(len(network)-1) + '_L' + str(start_layer)):
                p = tf.concat([network[-1], network[start_layer]], 2) # residual connection.
                network.append(tf.multiply(tf.layers.dense(p, args.res_proj, use_bias = False), 
                    tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32))) # projection.
 
    ## SUMMARY  
    if args.verbose:
        for I, i in enumerate(network):
            print(i.get_shape().as_list(), end="")
            print("%i:" % (I), end="")
            print(str(i.name))
 
    return network[-1]
 
## RNN LAYER
def rnn_layer(input, seq_len, args):
    with tf.variable_scope(args.name):
        if args.cell_type == 'IndRNNCell':
            cell_fw = tf.contrib.rnn.IndRNNCell(args.cell_size, activation=None) # forward IndRNNCell.
            if args.bidi:
                cell_bw = tf.contrib.rnn.IndRNNCell(args.cell_size, activation=None) # backward IndRNNCell.
        elif args.cell_type == 'IndyLSTMCell':
            cell_fw = tf.contrib.rnn.IndyLSTMCell(args.cell_size) # forward IndyLSTMCell.
            if args.bidi:
                cell_bw = tf.contrib.rnn.IndyLSTMCell(args.cell_size) # backward IndyLSTMCell.
        elif args.cell_type == 'LSTMCell':
            cell_fw = tf.contrib.rnn.LSTMCell(args.cell_size, args.peep, num_proj=args.cell_proj) # forward LSTMCell.
            if args.bidi:
                cell_bw = tf.contrib.rnn.LSTMCell(args.cell_size, args.peep, num_proj=args.cell_proj) # backward LSTMCell.
        if args.bidi:
            output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input, seq_len, 
                swap_memory=True, parallel_iterations=args.par_iter, dtype=tf.float32) # bidirectional recurrent neural network.
        else:
            output, _ = tf.nn.dynamic_rnn(cell_fw, input, seq_len, swap_memory=True,
                parallel_iterations=args.par_iter, dtype=tf.float32) # recurrent neural network.
        if args.bidi:
            if args.bidi_connect == 'concat':
                output = tf.concat(output, 2)
            elif args.bidi_connect == 'add':
                output = tf.add(output[0][2], output[1][2])
            else:
                raise ValueError('Incorrect args.bidi_connect specification.')
        if args.cell_type == 'IndRNNCell':
            output = tf.nn.relu(output)
        return output
 
## RNN LAYER WITH LSTMP CELLS
def lstmp_layer(input, seq_len, args):
    with tf.variable_scope('LSTMP_layer'):
        cell = tf.contrib.rnn.LSTMCell(args.cell_size, args.peep, num_proj=args.cell_proj) # BasicLSTMCell.
        output, _ = tf.nn.dynamic_rnn(cell, input, seq_len, swap_memory=True, 
            parallel_iterations=args.par_iter, dtype=tf.float32) # Recurrent Neural Network.
        return output
 
## BRNN LAYER WITH LSTMP CELLS
def blstmp_layer(input, seq_len, args):
    with tf.variable_scope('BLSTMP_layer'):
        cell_fw = tf.contrib.rnn.LSTMCell(args.cell_size, args.peep, num_proj=args.cell_proj) # forward BasicLSTMCell.
        cell_bw = tf.contrib.rnn.LSTMCell(args.cell_size, args.peep, num_proj=args.cell_proj) # backward BasicLSTMCell.
        output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input, seq_len, 
            swap_memory=True, parallel_iterations=args.par_iter, dtype=tf.float32) # Bidirectional Recurrent Neural Network.
        return tf.concat(output, 2) # concatenate forward and backward outputs.
 
## LOSS FUNCTIONS
def loss(target, estimate, loss_fnc):
    'loss functions for gradient descent.'
    with tf.name_scope(loss_fnc + '_loss'):
        if loss_fnc == 'mse':
            loss = tf.losses.mean_squared_error(labels=target, predictions=estimate)
        if loss_fnc == 'softmax_xentropy':
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=estimate)
        if loss_fnc == 'sigmoid_xentropy':
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=estimate)
    return loss
 
## GRADIENT DESCENT OPTIMISERS
def optimizer(loss, lr=None, epsilon=None, var_list=None, optimizer='adam'):
    'optimizers for training.'
    with tf.name_scope(optimizer + '_opt'):
        if optimizer == 'adam':
            if lr == None: lr = 0.001 # default.
            if epsilon == None: epsilon = 1e-8 # default.
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
            trainer = optimizer.minimize(loss, var_list=var_list) 
        if optimizer == 'nadam':
            if lr == None: lr = 0.001 # default.
            if epsilon == None: epsilon = 1e-8 # default.
            optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, epsilon=epsilon)
            trainer = optimizer.minimize(loss, var_list=var_list) 
        if optimizer == 'sgd':
            if lr == None: lr = 0.5 # default.
            optimizer = tf.train.GradientDescentOptimizer(lr)
            trainer = optimizer.minimize(loss, var_list=var_list) 
    return trainer, optimizer
 
## LAYER NORM FOR 3D TENSORS
def masked_layer_norm(input, seq_len):
    with tf.variable_scope('Layer_Norm'):
        mask = tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32) # convert mask to float.
        dims = input.get_shape().as_list()[-1] # get number of input dimensions.
        den = tf.multiply(tf.reduce_sum(mask, axis=1, keepdims=True), dims) # inverse of the number of input dimensions.
        mean = tf.divide(tf.reduce_sum(tf.multiply(input, mask), axis=[1, 2], keepdims=True), den) # mean over the input dimensions.
        var = tf.divide(tf.reduce_sum(tf.multiply(tf.square(tf.subtract(input, mean)), mask), axis=[1, 2], 
            keepdims=True), den) # variance over the input dimensions.
        beta = tf.Variable(tf.constant(0.0, shape=[dims]), trainable=True,name='beta')
        gamma = tf.Variable(tf.constant(1.0, shape=[dims]), trainable=True,name='Gamma')
        norm = tf.nn.batch_normalization(input, mean, var, offset=beta, scale=gamma, 
            variance_epsilon=1e-12) # normalise batch.
        norm = tf.multiply(norm, mask)
        return norm
 
# def e_swish(input):
#   beta = tf.Variable(tf.constant(1.0, shape=shape))
#   input = tf.multiply(input, beta)
#   tf.nn.sigmoid()2d
