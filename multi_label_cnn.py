import tensorflow as tf
import numpy as np
import random
import ops
import heapq
lower = str.lower
class model:
    image_shape = None
    learning_rate = 0.001
    model_restore = False
    working_dir = None
    batch_size = 64
    epochs = 10
    test_result = []
    train_result = []

    no_of_classes = None
    cnn_type = 'mydensenet'
    dropout_rate=0.5

    loss_type = 'binary_crossentropy'
    regularization_type = None
    regularization_coefficient = 0.0001
    logits=None
    optimizer=None

    def __init__(self):
        return

    def setup(self):
        tf.reset_default_graph()
        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[None, self.image_shape[0], self.image_shape[1], self.image_shape[2]],
                                name="input")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.no_of_classes], name="labels")
        self.lr = tf.placeholder("float", shape=[])
        self.is_train = tf.placeholder(tf.bool, shape=[])

        if self.logits==None:
            self.logits=self.get_model_by_name(self.x,self.is_train)
        else:
            self.logits=self.logits(self.x,self.is_train)
        with tf.name_scope('Output'):
            self.cross_entropy = ops.get_loss(self.logits, self.y, self.loss_type)
            if self.regularization_type != None:
                self.cross_entropy = ops.get_regularization(self.cross_entropy, self.regularization_type,
                                                            self.regularization_coefficient)
            self.prediction = tf.round(tf.nn.sigmoid(self.logits, name="sigmoid"))
            correct_prediction = tf.equal(self.prediction, self.y, name='Correct_prediction')
            print(correct_prediction)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')
            tf.summary.scalar("Cross_Entropy", self.cross_entropy)
            tf.summary.scalar("Accuracy", self.accuracy)

        with tf.name_scope('Optimizer'):
            if self.optimizer==None:
                # learningRate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=1,
                #                                          decay_steps=shape[0], decay_rate=0.97, staircase=True,
                #                                          name='Learning_Rate')
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
                # optimizer = tf.train.MomentumOptimizer(lr, .9, use_nesterov=True).minimize(cross_entropy)
                self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
                # optimizer = tf.train.AdadeltaOptimizer(lr).minimize(cross_entropy)
        self.session = tf.InteractiveSession()
        return

    def get_model_by_name(self,x,is_training):
        if self.cnn_type == 'lenet':
            return self.lenet(x,is_training)
        elif self.cnn_type == 'alexnet':
            return self.alexnet(x,is_training)
        elif self.cnn_type == 'vgg16':
            return self.vgg16(x,is_training)
        elif self.cnn_type == 'vgg19':
            return self.vgg19(x,is_training)
        elif self.cnn_type == 'mydensenet':
            return self.mydensenet(x,is_training)
        elif self.cnn_type == 'densenet':
            return self.densenet(x,is_training)
        elif self.cnn_type == 'densenet121':
            return self.densenet121(x,is_training)
        elif self.cnn_type == 'densenet161':
            return self.densenet161(x,is_training)
        elif self.cnn_type == 'densenet169':
            return self.densenet169(x,is_training)
        elif self.cnn_type == 'resnet':
            return self.resnet(x,is_training)
        elif self.cnn_type == 'resnet18':
            return self.resnet18(x,is_training)
        elif self.cnn_type == 'resnet32':
            return self.resnet32(x,is_training)
        elif self.cnn_type == 'resnet50':
            return self.resnet50(x,is_training)
        elif self.cnn_type == 'resnet101':
            return self.resnet101(x,is_training)
        elif self.cnn_type == 'resnet152':
            return self.resnet152(x,is_training)
        elif self.cnn_type == 'inceptionv2':
            return self.inception_v2(x,is_training)

    def get_paramter_count(self):
        return ops.get_no_of_parameter()

    def clear(self):
        tf.reset_default_graph()
        self.session.close()

    def lenet(self, x, is_training):
        x_shape = x.get_shape().as_list()[1:]
        kernel = {'c1': [5, 5, x_shape[2], 20], 'c2': [5, 5, 20, 50]}
        strides = {'1': [1, 1, 1, 1], '2': [1, 2, 2, 1]}
        pool_win_size = {'2': [1, 2, 2, 1]}

        with tf.variable_scope('Conv_1') as scope:
            conv = ops.conv2d(x,'conv1', kernel['c1'], strides['1'], 'SAME')
            conv = tf.nn.lrn(conv, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            conv = ops.max_pool(conv, pool_win_size['2'], strides['2'])
        with tf.variable_scope('Conv_2') as scope:
            conv = ops.conv2d(conv,'conv2', kernel['c2'], strides['1'], 'SAME')
            conv = tf.nn.lrn(conv, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            conv = ops.max_pool(conv, pool_win_size['2'], strides['2'])
        with tf.variable_scope('Flatten_layer') as scope:
            conv=ops.flatten(conv)
        with tf.variable_scope('Hidden_layer_1') as scope:
            conv = ops.get_hidden_layer(conv,'Hidden_layer_1',120, initializer='xavier')
        with tf.variable_scope('Hidden_layer_2') as scope:
            conv = ops.get_hidden_layer(conv,'Hidden_layer_2', 84, initializer='xavier')
        with tf.variable_scope('Output_layer') as scope:
            conv = ops.get_hidden_layer(conv,'output_layer', self.no_of_classes, activation="none", initializer='xavier')
        return conv

    def alexnet(self, x, is_training):
        x_shape = x.get_shape().as_list()[1:]
        kernel = {'c1': [11, 11, x_shape[2], 96], 'c2': [5, 5, 96, 256],
                  'c3': [3, 3, 256, 384], 'c4': [3, 3, 384, 384],
                  'c5': [3, 3, 384, 256]}
        strides = {'1': [1, 1, 1, 1], '2': [1, 2, 2, 1], '3': [1, 3, 3, 1], '4': [1, 4, 4, 1]}
        pool_win_size = {'1': [1, 1, 1, 1], '2': [1, 2, 2, 1], '3': [1, 3, 3, 1], '4': [1, 4, 4, 1]}

        with tf.variable_scope('Conv_1') as scope:
            conv = ops.conv2d(x,'conv_1', kernel['c1'], strides['4'], 'VALID')
            conv = tf.nn.lrn(conv, depth_radius=2, bias=1.0, alpha=1e-05, beta=0.75)
            conv = ops.max_pool(conv, pool_win_size['3'], strides['2'], "VALID")
        with tf.variable_scope('Conv_2') as scope:
            conv = ops.conv2d(conv,'conv_2', kernel['c2'], strides['1'], padding='SAME', groups=2)
            conv = tf.nn.lrn(conv, depth_radius=2, bias=1.0, alpha=1e-05, beta=0.75)
            conv = ops.max_pool(conv, pool_win_size['3'], strides['2'], 'VALID')
        with tf.variable_scope('Conv_3') as scope:
            conv = ops.conv2d(conv,'conv_3', kernel['c3'], strides['1'], 'SAME')
        with tf.variable_scope('Conv_4') as scope:
            conv = ops.conv2d(conv,'conv_4', kernel['c4'], strides['1'], 'SAME', groups=2)
        with tf.variable_scope('Conv_5') as scope:
            conv = ops.conv2d(conv,'conv_5', kernel['c5'], strides['1'], 'SAME', groups=2)
            conv = ops.max_pool(conv, pool_win_size['3'], strides['2'], 'VALID')
        with tf.variable_scope('Flatten_layer') as scope:
            conv=ops.flatten(conv)
        with tf.variable_scope('Hidden_layer_1') as scope:
            conv = ops.get_hidden_layer(conv,'Hidden_layer_1', 4096, activation=['relu', 'dropout'], initializer='xavier')
        with tf.variable_scope('Hidden_layer_2') as scope:
            conv = ops.get_hidden_layer(conv,'Hidden_layer_2', 4096, activation=['relu', 'dropout'], initializer='xavier')
        with tf.variable_scope('Output_layer') as scope:
            conv = ops.get_hidden_layer(conv,'output_layer',self.no_of_classes, activation='none', initializer='xavier')
        return conv

    def vgg16(self, x, is_training):
        x_shape = x.get_shape().as_list()[1:]
        kernel = {
            'c1_1': [3, 3, x_shape[2], 64], 'c1_2': [3, 3, 64, 64],
            'c2_1': [3, 3, 64, 128], 'c2_2': [3, 3, 128, 128],
            'c3_1': [3, 3, 128, 256], 'c3_2': [3, 3, 256, 256],
            'c3_3': [3, 3, 256, 256],
            'c4_1': [3, 3, 256, 512], 'c4_2': [3, 3, 512, 512],
            'c4_3': [3, 3, 512, 512],
            'c5_1': [3, 3, 512, 512], 'c5_2': [3, 3, 512, 512],
            'c5_3': [3, 3, 512, 512]}
        strides = {'c': [1, 1, 1, 1], 'p': [1, 2, 2, 1]}
        pool_win_size = [1, 2, 2, 1]
        conv = x

        with tf.variable_scope('Conv_1') as scope:
            conv = ops.conv2d(conv,'Conv_1_1', kernel['c1_1'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_1_2', kernel['c1_2'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.max_pool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_2') as scope:
            conv = ops.conv2d(conv,'Conv_2_1', kernel['c2_1'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_2_2', kernel['c2_2'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.max_pool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_3') as scope:
            conv = ops.conv2d(conv,'Conv_3_1', kernel['c3_1'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_3_2', kernel['c3_2'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_3_3', kernel['c3_3'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.max_pool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_4') as scope:
            conv = ops.conv2d(conv,'Conv_4_1', kernel['c4_1'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_4_2', kernel['c4_2'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_4_3', kernel['c4_3'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.max_pool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_5') as scope:
            conv = ops.conv2d(conv,'Conv_5_1', kernel['c5_1'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_5_2', kernel['c5_2'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_5_3', kernel['c5_3'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.max_pool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Flatten_layer') as scope:
            conv = ops.flatten(conv)
        with tf.variable_scope('Hidden_layer_1') as scope:
            conv = ops.get_hidden_layer(conv,'Hidden_layer_1', 4096, activation='relu', initializer='xavier')
        with tf.variable_scope('Hidden_layer_2') as scope:
            conv = ops.get_hidden_layer(conv,'Hidden_layer_2', 4096, activation='relu', initializer='xavier')
        with tf.variable_scope('Output_layer') as scope:
            conv = ops.get_hidden_layer(conv,'output_layer', self.no_of_classes, activation="none", initializer='xavier')
        return conv

    def vgg19(self, x, is_training):
        x_shape = x.get_shape().as_list()[1:]
        kernel = {'c1_1': [3, 3, x_shape[2], 64], 'c1_2': [3, 3, 64, 64],
                  'c2_1': [3, 3, 64, 128], 'c2_2': [3, 3, 128, 128],
                  'c3_1': [3, 3, 128, 256], 'c3_2': [3, 3, 256, 256],
                  'c3_3': [3, 3, 256, 256], 'c3_4': [3, 3, 256, 256],
                  'c4_1': [3, 3, 256, 512], 'c4_2': [3, 3, 512, 512],
                  'c4_3': [3, 3, 512, 512], 'c4_4': [3, 3, 512, 512],
                  'c5_1': [3, 3, 512, 512], 'c5_2': [3, 3, 512, 512],
                  'c5_3': [3, 3, 512, 512], 'c5_4': [3, 3, 512, 512]}
        strides = {'c': [1, 1, 1, 1], 'p': [1, 2, 2, 1]}
        pool_win_size = [1, 2, 2, 1]
        with tf.variable_scope('Conv_1') as scope:
            conv = ops.conv2d(x,'Conv_1_1', kernel['c1_1'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_1_2', kernel['c1_2'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.max_pool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_2') as scope:
            conv = ops.conv2d(conv,'Conv_2_1', kernel['c2_1'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_2_2', kernel['c2_2'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.max_pool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_3') as scope:
            conv = ops.conv2d(conv,'Conv_3_1', kernel['c3_1'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_3_2', kernel['c3_2'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_3_3', kernel['c3_3'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_3_4', kernel['c3_4'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.max_pool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_4') as scope:
            conv = ops.conv2d(conv,'Conv_4_1', kernel['c4_1'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_4_2', kernel['c4_2'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_4_3', kernel['c4_3'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_4_4', kernel['c4_4'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.max_pool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_5') as scope:
            conv = ops.conv2d(conv,'Conv_5_1', kernel['c5_1'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_5_2', kernel['c5_2'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_5_3', kernel['c5_3'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.conv2d(conv,'Conv_5_4', kernel['c5_4'], strides['c'], 'SAME')
            conv = tf.nn.relu(conv)
            conv = ops.max_pool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Flatten_layer') as scope:
            conv = ops.flatten(conv)
        with tf.variable_scope('Hidden_layer_1') as scope:
            conv = ops.get_hidden_layer(conv,'Hidden_layer_1', 4096, activation='relu', initializer='xavier')
        with tf.variable_scope('Hidden_layer_2') as scope:
            conv = ops.get_hidden_layer(conv,'Hidden_layer_2', 4096, activation='relu', initializer='xavier')
        with tf.variable_scope('Output_layer') as scope:
            conv = ops.get_hidden_layer(conv, 'output_layer',self.no_of_classes, activation="none", initializer='xavier')
        return conv

    def inception_v2(self, input, is_training):
        input_shape = input.get_shape().as_list()[1:]
        conv = ops.conv2d(input,'conv1',kernel_size=[7, 7, input_shape[2], 64], strides=[1, 2, 2, 1])
        conv = tf.nn.relu(conv)
        conv = ops.max_pool(conv, size=[1, 3, 3, 1], strides=[1, 2, 2, 1])
        conv = tf.nn.local_response_normalization(conv, depth_radius=2, alpha=2e-05, beta=0.75)

        conv = ops.conv2d(conv,'conv2', kernel_size=[1, 1, 64, 64], strides=[1, 1, 1, 1], padding='VALID')
        conv = tf.nn.relu(conv)

        conv_shape = conv.get_shape().as_list()[1:]
        conv = ops.conv2d(conv,'conv3', kernel_size=[3, 3, conv_shape[2], 192], strides=[1, 1, 1, 1])
        conv = tf.nn.relu(conv)

        conv = tf.nn.local_response_normalization(conv, depth_radius=2, alpha=2e-05, beta=0.75)
        conv = ops.max_pool(conv, size=[1, 3, 3, 1], strides=[1, 2, 2, 1])

        conv = ops.inception_v2_block(conv,'Block_1',is_training, out_channel={'1': 64, '3': 128, '5': 32},
                                      reduced_out_channel={'3': 96, '5': 16, 'p': 32})
        conv = ops.batch_normalization(conv, is_training)
        conv = tf.nn.relu(conv)

        conv = ops.inception_v2_block(conv,'Block_2', is_training, out_channel={'1': 128, '3': 192, '5': 96},
                                      reduced_out_channel={'3': 128, '5': 32, 'p': 64})
        conv = ops.batch_normalization(conv, is_training)
        conv = tf.nn.relu(conv)

        conv = ops.max_pool(conv, size=[1, 3, 3, 1], strides=[1, 2, 2, 1])

        conv = ops.inception_v2_block(conv,'Block_3', is_training, out_channel={'1': 192, '3': 208, '5': 48},
                                      reduced_out_channel={'3': 96, '5': 16, 'p': 64})
        conv = ops.batch_normalization(conv, is_training)
        conv = tf.nn.relu(conv)

        conv = ops.inception_v2_block(conv,'Block_4', is_training, out_channel={'1': 160, '3': 224, '5': 64},
                                      reduced_out_channel={'3': 112, '5': 24, 'p': 64})
        conv = ops.batch_normalization(conv, is_training)
        conv = tf.nn.relu(conv)

        conv = ops.inception_v2_block(conv,'Block_5', is_training, out_channel={'1': 128, '3': 256, '5': 64},
                                      reduced_out_channel={'3': 128, '5': 24, 'p': 64})
        conv = ops.batch_normalization(conv, is_training)
        conv = tf.nn.relu(conv)

        conv = ops.inception_v2_block(conv,'Block_6', is_training, out_channel={'1': 112, '3': 228, '5': 64},
                                      reduced_out_channel={'3': 144, '5': 32, 'p': 64})
        conv = ops.batch_normalization(conv, is_training)
        conv = tf.nn.relu(conv)

        conv = ops.inception_v2_block(conv,'Block_7', is_training, out_channel={'1': 256, '3': 320, '5': 128},
                                      reduced_out_channel={'3': 160, '5': 32, 'p': 128})
        conv = ops.batch_normalization(conv, is_training)
        conv = tf.nn.relu(conv)

        conv = ops.max_pool(conv, size=[1, 3, 3, 1], strides=[1, 2, 2, 1])

        conv = ops.inception_v2_block(conv,'Block_8', is_training, out_channel={'1': 256, '3': 320, '5': 128},
                                      reduced_out_channel={'3': 160, '5': 32, 'p': 128})
        conv = ops.batch_normalization(conv, is_training)
        conv = tf.nn.relu(conv)

        conv = ops.inception_v2_block(conv,'Block_9', is_training, out_channel={'1': 384, '3': 384, '5': 128},
                                      reduced_out_channel={'3': 192, '5': 48, 'p': 128})
        conv = ops.batch_normalization(conv, is_training)
        conv = tf.nn.relu(conv)

        conv = ops.global_avg_pool(conv)
        conv = ops.flatten(conv)

        conv = tf.nn.dropout(conv, 0.4)
        conv = ops.get_hidden_layer(conv,'output_layer',1000, 'none', 'xavier')
        return conv

    def densenet(self, x, is_training, no_of_blocks=3, block_layers=7, first_conv_op_channel=16, block_op_channel=12,
                 kernal_size=3):

        strides = {'1': [1, 1, 1, 1], '2': [1, 2, 2, 1], '3': [1, 3, 3, 1], '4': [1, 4, 4, 1], '8': [1, 8, 8, 1]}
        pool_win_size = {'1': [1, 1, 1, 1], '2': [1, 2, 2, 1], '3': [1, 3, 3, 1], '4': [1, 4, 4, 1], '8': [1, 8, 8, 1]}
        x_shape = x.get_shape().as_list()[1:]

        kernel = [kernal_size, kernal_size, x_shape[2], first_conv_op_channel]
        conv = ops.conv2d(x, kernel, strides['1'], 'SAME', initial='xavier', with_bias=False)
        if isinstance(block_layers, int):
            with tf.variable_scope('Dense_Block_1') as scope:
                kernel = [kernal_size, kernal_size, first_conv_op_channel, block_op_channel]
                conv = ops.conv2d_dense_block(conv,'Dense_Block_1', is_training, kernel, layers=block_layers,dropout_rate=self.dropout_rate)
                op_channel = first_conv_op_channel + block_layers * block_op_channel
            for _ in range(1, no_of_blocks):
                with tf.variable_scope('transition_layer_' + str(_ - 1)) as scope:
                    conv = tf.contrib.layers.batch_norm(conv, scale=True, is_training=is_training,
                                                        updates_collections=None)
                    conv = tf.nn.relu(conv)
                    kernel = [kernal_size, kernal_size, op_channel, op_channel]
                    conv = ops.conv2d(conv,'transition_layer_' + str(_ - 1), kernel, strides=[1, 1, 1, 1], padding='SAME', initial='xavier',
                                       with_bias=False)
                    conv = tf.nn.dropout(conv, self.dropout_rate)
                    conv = ops.avg_pool(conv, pool_win_size['2'], strides['2'], 'VALID')
                with tf.variable_scope('Dense_Block_' + str(_)) as scope:
                    kernel = [kernal_size, kernal_size, op_channel, block_op_channel]
                    conv = ops.conv2d_dense_block(conv,'Dense_Block_'+str(_),is_training, kernel, layers=block_layers,dropout_rate=self.dropout_rate)
                    op_channel += block_layers * block_op_channel
        elif isinstance(block_layers, list):
            no_of_blocks = len(block_layers)

            with tf.variable_scope('Dense_Block_1') as scope:
                kernel = [kernal_size, kernal_size, first_conv_op_channel, block_op_channel]
                conv = ops.conv2d_dense_block(conv,'Dense_Block_1', is_training, kernel, layers=block_layers[0],dropout_rate=self.dropout_rate)
                op_channel = first_conv_op_channel + block_layers[0] * block_op_channel

            for _ in range(1, no_of_blocks):
                with tf.variable_scope('transition_layer_' + str(_)) as scope:
                    conv = tf.contrib.layers.batch_norm(conv, scale=True, is_training=is_training,
                                                        updates_collections=None)
                    conv = tf.nn.relu(conv)
                    kernel = [kernal_size, kernal_size, op_channel, op_channel]
                    conv = ops.conv2d(conv,'transition_layer_' + str(_), kernel, strides=[1, 1, 1, 1], padding='SAME', initial='xavier',
                                       with_bias=False)
                    conv = tf.nn.dropout(conv, self.dropout_rate)
                    conv = ops.avg_pool(conv, pool_win_size['2'], strides['2'], 'VALID')
                with tf.variable_scope('Dense_Block_' + str(_ + 1)) as scope:
                    kernel = [kernal_size, kernal_size, op_channel, block_op_channel]
                    conv = ops.conv2d_dense_block(conv,'Dense_Block_'+str(_), is_training, kernel, layers=block_layers[_],dropout_rate=self.dropout_rate)
                    op_channel += block_layers[_] * block_op_channel
        with tf.variable_scope('Global_Average_Pooling') as scope:
            conv = tf.contrib.layers.batch_norm(conv, scale=True, is_training=is_training, updates_collections=None)
            conv = tf.nn.relu(conv)
            conv = ops.avg_pool(conv, pool_win_size['8'], strides['8'], 'VALID')

        with tf.variable_scope('Flatten_layer') as scope:
            conv= ops.flatten(conv)

        with tf.variable_scope('Output_layer') as scope:
            conv = ops.get_hidden_layer(conv,'output_layer',self.no_of_classes, activation='none', initializer='xavier')

        return conv


    def resnet_without_bottleneck(self, input, is_training,layer_from_2=[2,2,2,2],first_kernel=7,first_stride=2,first_pool=True,stride=2):

        input_shape = input.get_shape().as_list()[1:]
        conv=ops.conv2d(input,'initial_conv',[first_kernel,first_kernel,input_shape[2],64],[1,first_stride,first_stride,1])
        if first_pool:
            conv=ops.max_pool(conv, [1, 3, 3, 1], [1, 2, 2, 1])

        for i in range(layer_from_2[0]):
            conv=ops.residual_block(conv,'Block_1_'+str(i),is_training,64,kernel=3,first_block=True,stride=stride)

        for i in range(layer_from_2[1]):
            conv=ops.residual_block(conv,'Block_2_'+str(i),is_training,128,kernel=3,first_block=True,stride=stride)

        for i in range(layer_from_2[2]):
            conv=ops.residual_block(conv,'Block_3_'+str(i),is_training,256,kernel=3,first_block=True,stride=stride)

        for i in range(layer_from_2[3]):
            conv=ops.residual_block(conv,'Block_4_'+str(i),is_training,512,kernel=3,first_block=True,stride=stride)

        with tf.variable_scope('unit'):
            conv = ops.batch_normalization(conv,is_training)
            conv = tf.nn.relu(conv)
            conv = ops.global_avg_pool(conv)
            conv =ops.flatten(conv)
        with tf.variable_scope('logit'):
            conv = ops.get_hidden_layer(conv,'output_layer',self.no_of_classes,'none')
        return conv

    def resnet_with_bottleneck(self,input,is_training,layer_from_2=[3,4,6,3],first_kernel=7,first_stride=2,first_pool=True,stride=2):

        input_shape = input.get_shape().as_list()[1:]
        conv=ops.conv2d(input,'initial_conv',[first_kernel,first_kernel,input_shape[2],64],[1,first_stride,first_stride,1])
        if first_pool:
            conv=ops.max_pool(conv, [1, 3, 3, 1], [1, 2, 2, 1])

        for i in range(layer_from_2[0]):
            conv=ops.residual_bottleneck_block(conv,'Block_1_'+str(i),is_training,256,kernel=3,first_block=True,stride=stride)

        for i in range(layer_from_2[1]):
            conv=ops.residual_bottleneck_block(conv,'Block_2_'+str(i),is_training,512,kernel=3,first_block=True,stride=stride)

        for i in range(layer_from_2[2]):
            conv=ops.residual_bottleneck_block(conv,'Block_3_'+str(i),is_training,1024,kernel=3,first_block=True,stride=stride)

        for i in range(layer_from_2[3]):
            conv=ops.residual_bottleneck_block(conv,'Block_4_'+str(i),is_training,2048,kernel=3,first_block=True,stride=stride)
        with tf.variable_scope('unit'):
            conv = ops.batch_normalization(conv,is_training)
            conv = tf.nn.relu(conv)
            conv = ops.global_avg_pool(conv)
            conv =ops.flatten(conv)
        with tf.variable_scope('logit'):
            conv = ops.get_hidden_layer(conv,'output',self.no_of_classes,'none')
        return conv

    def myensenet(self,input,is_training):
        return self.densenet(input, is_training, no_of_blocks=3, block_layers=[7, 7, 7], first_conv_op_channel=32,
                      block_op_channel=16, kernal_size=3)

    def densenet121(self,input,is_training):
        return self.densenet(input,is_training, block_layers=[6, 12, 24, 16], first_conv_op_channel=64, block_op_channel=32,
                                   kernal_size=7)

    def densenet161(self, input, is_training):
        return self.densenet(input, is_training, block_layers=[6, 12, 36, 24], first_conv_op_channel=96, block_op_channel=48,
                                   kernal_size=7)

    def densenet169(self, input, is_training):
        return self.densenet(input, is_training, block_layers=[6, 12, 32, 32], first_conv_op_channel=64, block_op_channel=32,
                                   kernal_size=7)
    def mydensenet(self,input,is_training):
        return self.densenet(input, is_training, no_of_blocks=3, block_layers=[7,7,7], first_conv_op_channel=32,
                 block_op_channel=16, kernal_size=3)

    def resnet(self,x,is_training):
        return self.resnet_with_bottleneck(x, is_training,first_pool=False,layer_from_2=[2,2,2,2],first_stride=1,stride=1)

    def resnet18(self,input,is_training,first_kernel=7,first_stride=2,first_pool=True,stride=2):
        return self.resnet_without_bottleneck(input,is_training,[2,2,2,2],first_kernel,first_stride,first_pool,stride)

    def resnet32(self,input,is_training,first_kernel=7,first_stride=2,first_pool=True,stride=2):
        return self.resnet_without_bottleneck(input,is_training,[3,4,6,3],first_kernel,first_stride,first_pool,stride)

    def resnet50(self,input,is_training,first_kernel=7,first_stride=2,first_pool=True,stride=2):
        return self.resnet_with_bottleneck(input,is_training,[3,4,6,3],first_kernel,first_stride,first_pool,stride)

    def resnet101(self,input,is_training,first_kernel=7,first_stride=2,first_pool=True,stride=2):
        return self.resnet_with_bottleneck(input,is_training,[3,4,23,3],first_kernel,first_stride,first_pool,stride)

    def resnet152(self,input,is_training,first_kernel=7,first_stride=2,first_pool=True,stride=2):
        return self.resnet_with_bottleneck(input,is_training,[3,4,36,3],first_kernel,first_stride,first_pool,stride)

    def train(self,train,val_data=None,max_keep=100,shuffle=False):
        init = tf.global_variables_initializer()
        self.session.run(init)
        epoch_offset=0

        saver = tf.train.Saver(max_to_keep=max_keep)
        if self.model_restore == True and self.working_dir!= None:
            name = ops.look_for_last_checkpoint(self.working_dir + "/model/")
            if name is not None:
                saver.restore(self.session, self.working_dir + "/model/" + name)
                print('Model Succesfully Loaded : ', name)
                epoch_offset=int(name[6:])

        if self.working_dir != None:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.working_dir + '/train', self.session.graph)
            test_writer = tf.summary.FileWriter(self.working_dir + '/test')

        for epoch in range(epoch_offset+1,epoch_offset+self.epochs+1):
            if shuffle==True:
                ind_list = [i for i in range(len(train['x']))]
                random.shuffle(ind_list)
                train['x']=train['x'][ind_list]
                train['y']=train['y'][ind_list]
            epoch_loss = 0
            acc = 0
            i = 0
            batch_iteration = 0
            while i < len(train['x']):
                start = i
                end = i + self.batch_size
                if (end > len(train['x'])): end = len(train['x'])
                batch_x = train['x'][start:end]
                batch_y = train['y'][start:end]
                if self.working_dir != None:
                    summary, _, loss,batch_acc= self.session.run([merged, self.optimizer,self.cross_entropy,self.accuracy],
                                                              feed_dict={self.x: batch_x, self.y: batch_y,self.lr:self.learning_rate,self.is_train:True})
                else:
                    _, loss,batch_acc= self.session.run([self.optimizer, self.cross_entropy,self.accuracy],
                                                     feed_dict={self.x: batch_x,self.y: batch_y,self.lr: self.learning_rate,self.is_train:True})
                epoch_loss += loss
                acc += batch_acc
                batch_iteration += 1
                i += self.batch_size
                print('Training: Accuracy={} loss={}\r '.format(round(batch_acc,4),round(epoch_loss/batch_iteration,4)),)
            if self.working_dir != None:
                train_writer.add_summary(summary, epoch)
            self.train_result.append([epoch, epoch_loss/batch_iteration, acc / batch_iteration])
            if val_data != None:
                epoch_loss = 0
                acc = 0
                i = 0
                batch_iteration = 0
                while i < len(val_data['x']):
                    start = i
                    end = i + self.batch_size
                    if (end > len(val_data['x'])): end = len(val_data['x'])
                    batch_x = val_data['x'][start:end]
                    batch_y = val_data['y'][start:end]
                    if self.working_dir != None:
                        summary,loss,batch_acc= self.session.run([merged,self.cross_entropy,self.accuracy],
                                                              feed_dict={self.x: batch_x, self.y: batch_y,self.lr:self.learning_rate,self.is_train:False})
                    else:
                        loss,batch_acc= self.session.run([self.cross_entropy,self.accuracy],
                                                     feed_dict={self.x: batch_x,self.y: batch_y,self.lr: self.learning_rate,self.is_train:False})
                    epoch_loss += loss

                    acc += batch_acc
                    batch_iteration += 1
                    i += self.batch_size
                    print('Validation: Accuracy={} loss={}\r '.format(round(batch_acc, 4),round(epoch_loss / batch_iteration, 4)),)
                if self.working_dir != None:
                    test_writer.add_summary(summary, epoch)
                self.test_result.append([epoch, epoch_loss/batch_iteration, acc / batch_iteration])

                print("Training:", self.train_result[len(self.train_result) - 1], "Val:", self.test_result[len(self.test_result) - 1])
            else:
                print("Training :", self.train_result[len(self.train_result) - 1])

            if self.working_dir != None:
                save_path = saver.save(self.session, self.working_dir + "/model/" + 'model',global_step=epoch)
        print('Training Succesfully Complete')

    def check_restore(self):
        saver = tf.train.Saver()
        print(self.working_dir+'/model/')
        try:
            saver.restore(self.session, tf.train.latest_checkpoint(self.working_dir+'/model/'))
            return True
        except:
            return False

    def predict(self,test):
        saver = tf.train.Saver()
        saver.restore(self.session, tf.train.latest_checkpoint(self.working_dir+'/model/'))
        print ('Retored model',ops.look_for_last_checkpoint(self.working_dir + "/model/"))
        merged = tf.summary.merge_all()
        if 'x' in test and test['x'].shape[0] > 0:
            i = 0
            iteration = 0
            acc = 0
            test_prediction=[]
            j=0
            while i < len(test['x']):
                start = i
                end = i + self.batch_size
                if (end > len(test['x'])): end = len(test['x'])
                batch_x = test['x'][start:end]
                if 'y' in test and test['y'].shape[0] > 0:
                    batch_y = test['y'][start:end]
                    pred,batch_acc = self.session.run([self.prediction,self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y,self.is_train:False})
                    acc += batch_acc
                else:
                    pred= self.session.run([self.prediction], feed_dict={self.x: batch_x,self.is_train:False})
                iteration += 1
                i += self.batch_size
                if isinstance(pred,list):
                    test_prediction+=pred[0].tolist()
                else:
                    test_prediction += pred.tolist()
            if 'y' in test and test['y'].shape[0] > 0:
                return np.array(test_prediction), acc/iteration
            else:
                return np.array(test_prediction)
