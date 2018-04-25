import tensorflow as tf
import numpy as np

lower = str.lower


def look_for_last_checkpoint(mode_dir):
    try:
        fr = open(mode_dir + 'checkpoint', "r")
    except:
        return None
    f_line = fr.readline()
    start = f_line.find('"')
    end = f_line.rfind('"')
    return f_line[start + 1:end]


def get_variable(name, shape, initializer='xavier', dtype=tf.float32):
    if lower(initializer) == 'random':
        initial_weight = tf.random_normal_initializer(stddev=0.05, dtype=dtype)
    elif lower(initializer) == 'truncated':
        initial_weight = tf.truncated_normal_initializer(stddev=0.05, dtype=dtype)
    elif lower(initializer) == 'uniform':
        initial_weight = tf.random_uniform_initializer()
    elif lower(initializer) == 'xavier':
        initial_weight = tf.contrib.layers.xavier_initializer()
    elif lower(initializer) == 'xavier_conv2d':
        initial_weight = tf.contrib.layers.xavier_initializer_conv2d()
    else:
        initial_weight = initializer
    return tf.get_variable(name, shape=shape, initializer=initial_weight, dtype=dtype)


def bias_variable(shape, value=None):
    if value == None:
        initial = tf.constant(0.001, shape=shape)
    else:
        initial = tf.constant(value, shape=shape)
    return tf.Variable(initial)


def conv2d(input, name, kernel_size, strides, padding='SAME', initial='xavier', groups=1, with_bias=True):
    if groups == 1:
        W = get_variable(name, kernel_size, initial)
        conv = tf.nn.conv2d(input, W, strides, padding=padding, )
        if with_bias:
            return tf.nn.bias_add(conv, bias_variable([kernel_size[3]]))
        return conv
    else:
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides=strides, padding=padding)
        W = get_variable(name, kernel_size, initial)
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=input)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=W)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
        conv = tf.concat(axis=3, values=output_groups)
        if with_bias:
            tf.nn.bias_add(conv, bias_variable([kernel_size[3]]))
        return conv


def flatten(input):
    op_shape = input.get_shape().as_list()[1:]
    dim = 1
    for value in op_shape:
        dim = dim * value
    return tf.reshape(input, [-1, dim])


def avg_pool(input, size, strides, padding='SAME'):
    return tf.nn.avg_pool(input, size, strides, padding)


def max_pool(input, size, strides, padding='SAME'):
    return tf.nn.max_pool(input, size, strides, padding)


def global_avg_pool(input, dim=[1, 2]):
    assert input.get_shape().ndims == 4
    return tf.reduce_mean(input, dim)


def batch_normalization(input, is_training):
    return tf.contrib.layers.batch_norm(input, scale=True, is_training=is_training, updates_collections=None)


def get_hidden_layer(input, name, size=50, activation='relu', initializer='xavier', dtype=tf.float32):
    node_shape = input.get_shape().as_list()[1:]
    weight = get_variable('hidden_' + str(name), [node_shape[0], size], initializer, dtype)
    bias = get_variable('Baises_' + str(name), [1, size], initializer, dtype)
    output = tf.add(tf.matmul(input, weight), bias)
    if isinstance(activation, int):
        output = get_activation_function(output, activation)
    elif isinstance(activation, str):
        output = get_activation_function(output, activation)
    elif isinstance(activation, list):
        for a in activation:
            output = get_activation_function(output, a)
    elif activation == None:
        output = get_activation_function(output, 'relu')
    return output

def get_n_hidden_layers(input,name,hidden_sizes=None,activation_function_list=None,initializer='xavier'):
    try:
        no_of_layers = len(hidden_sizes)
    except:
        no_of_layers = 0
    output=input
    for i in range(no_of_layers):
        with tf.name_scope('Hidden_Layer_' + str(i + 1)):
            output = get_hidden_layer(output,name+'_'+str(i + 1),hidden_sizes[i], activation=activation_function_list[i],initializer=initializer)
    return output


def leaky_relu(node, parameter=0.1):
    shape = node.get_shape().as_list()[1:]
    const = tf.constant(value=parameter, shape=shape)
    return tf.maximum(node * const, node)


def squash(vector):
    vector += 0.00001  # Workaround for the squashing function ...
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm)
    vec_squashed = scalar_factor * vector  # element-wise
    return vec_squashed


def get_activation_function(input, choice=2, value=None):
    if choice == 0 or lower(str(choice)) == 'none':
        return input
    if choice == 1 or lower(str(choice)) == 'relu':
        return tf.nn.relu(input)
    if choice == 2 or lower(str(choice)) == 'leaky_relu':
        if value == None:
            value = 0.1
        return leaky_relu(input, value)
    if choice == 3 or lower(str(choice)) == 'crelu':
        return tf.nn.crelu(input)
    if choice == 4 or lower(str(choice)) == 'relu6':
        return tf.nn.relu6(input)
    if choice == 5 or lower(str(choice)) == 'elu':
        return tf.nn.elu(input)
    if choice == 6 or lower(str(choice)) == 'sigmoid':
        return tf.nn.sigmoid(input)
    if choice == 7 or lower(str(choice)) == 'tanh':
        return tf.nn.tanh(input)
    if choice == 8 or lower(str(choice)) == 'softplus':
        return tf.nn.softplus(input)
    if choice == 9 or lower(str(choice)) == 'softsign':
        return tf.nn.softsign(input)
    if choice == 10 or lower(str(choice)) == 'softmax':
        return tf.nn.softmax(logits=input)
    if choice == 11 or lower(str(choice)) == 'squash':
        return tf.nn.softmax(logits=input)
    if choice == 12 or lower(str(choice)) == 'dropout':
        if value == None:
            value = 0.5
        return tf.nn.dropout(input, value)


def multi_layer_bank(input, out_channel={'1': 32, '3': 32, '5': 32}, strides=[1, 1, 1, 1]):
    input_shape = input.get_shape().as_list()[1:]
    con_1x1 = conv2d(input, 'MLB_Conv_0_1', kernel_size=[1, 1, input_shape[2], out_channel['1']], strides=strides,
                     initial='xavier')
    con_3x3 = conv2d(input, 'MLB_Conv_0_2', kernel_size=[3, 3, input_shape[2], out_channel['3']], strides=strides,
                     initial='xavier')
    con_5x5 = conv2d(input, 'MLB_Conv_0_3', kernel_size=[5, 5, input_shape[2], out_channel['5']], strides=strides,
                     initial='xavier')
    pool = max_pool(input, size=[1, 3, 3, 1], strides=strides)
    output = tf.concat([input, con_1x1, con_3x3, con_5x5, pool], 3)
    return output


def get_loss(logits, labels, loss_type='softmax'):
    if lower(loss_type) == "softmax":
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='Cross_entropy'))
    elif lower(loss_type) == "hinge":
        cross_entropy = tf.reduce_mean(tf.losses.hinge_loss(logits=logits, labels=labels))
    elif lower(loss_type) == "huber":
        cross_entropy = tf.reduce_mean(tf.losses.huber_loss(labels=labels, predictions=logits))
    elif lower(loss_type) == "log":
        cross_entropy = tf.reduce_mean(tf.losses.log_loss(labels=labels, predictions=logits))
    elif lower(loss_type) == "absolute":
        cross_entropy = tf.reduce_mean(tf.losses.absolute_difference(labels=labels, predictions=logits))
    elif lower(loss_type) == "mse":
        cross_entropy = tf.losses.mean_squared_error(labels=labels, predictions=logits)
    elif lower(loss_type) == "mpse":
        cross_entropy = tf.losses.mean_pairwise_squared_error(labels=labels, predictions=logits)
    elif lower(loss_type) == "sigmoid":
        cross_entropy = tf.losses.sigmoid_cross_entropy(labels, logits)
    elif lower(loss_type) == "binary_crossentropy":
        cross_entropy = tf.reduce_mean(tf.keras.losses.binary_crossentropy(logits, labels))
    else:
        one=tf.constant(1.0)
        cross_entropy = -tf.reduce_mean(labels * tf.log(logits) + (one - labels) * tf.log(one - logits))
    return cross_entropy


def get_regularization(input_loss, regularization_type='l2', regularization_coefficient=0.0001):
    if regularization_type == 'l2':
        beta = tf.constant(regularization_coefficient)
        regularized_loss = input_loss + beta * tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()], 'L2_regurlization')
        return regularized_loss
    if regularization_type == 'l1':
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=regularization_coefficient, scope=None)
        weights = tf.trainable_variables()
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
        regularized_loss = input_loss + regularization_penalty
        return regularized_loss
    if regularization_type == 'elastic_net':
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=regularization_coefficient[0], scope=None)
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=regularization_coefficient[1], scope=None)
        weights = tf.trainable_variables()
        l1_regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
        l2_regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
        regularized_loss = input_loss + l1_regularization_penalty + l2_regularization_penalty
        return regularized_loss


def conv2d_dense_block(input, name, is_training, kernel=[3, 3, 16, 12], strides=[1, 1, 1, 1], layers=5,
                       dropout_rate=0.5, activation='relu'):
    current = input
    features = kernel[2]
    for _ in range(layers):
        with tf.variable_scope('Bottleneck_layer_' + name + str(_)) as scope:
            tmp = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training,
                                               updates_collections=None)
            tmp = get_activation_function(tmp, 'relu')
            tmp = conv2d(tmp, 'Bottleneck_layer_' + name + str(_) + 'conv', [kernel[0], kernel[0], features, kernel[3]],
                         strides=strides, padding='SAME',
                         initial='xavier', with_bias=False)
            tmp = tf.nn.dropout(tmp, dropout_rate)
            current = tf.concat([current, tmp], 3)
            features += kernel[3]
    return current


def residual_block(input, name, is_training, output_channel, kernel=3, stride=1, first_block=False, padding_option=0,
                   pad_or_conv=False):
    original_shape = input.get_shape().as_list()[1:]
    original_input = input

    if first_block:
        input = conv2d(input, name + 'conv_1', kernel_size=[kernel, kernel, original_shape[2], output_channel],
                       strides=[1, stride, stride, 1], padding='SAME', initial='xavier', with_bias=False)
    else:
        input = batch_normalization(input, is_training)
        input = tf.nn.relu(input)
        input = conv2d(input, name + 'conv_1', kernel_size=[kernel, kernel, original_shape[2], output_channel],
                       strides=[1, stride, stride, 1], padding='SAME', initial='xavier', with_bias=False)

    input = batch_normalization(input, is_training)
    input = tf.nn.relu(input)
    input = conv2d(input, name + 'conv_2', kernel_size=[kernel, kernel, output_channel, output_channel],
                   strides=[1, 1, 1, 1],
                   padding='SAME', initial='xavier', with_bias=False)

    if stride != 1:
        original_input = avg_pool(original_input, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
    input_shape = input.get_shape().as_list()[1:]

    if original_shape[2] != input_shape[2] and original_shape[2] < input_shape[2]:
        if pad_or_conv == True:
            original_input = conv2d(original_input, name + 'conv_3', [1, 1, original_shape[2], input_shape[2]],
                                    [1, 1, 1, 1],
                                    with_bias=False)
        else:
            original_input = tf.pad(original_input,
                                    [[padding_option, padding_option], [padding_option, padding_option],
                                     [padding_option, padding_option],
                                     [(input_shape[2] - original_shape[2]) // 2,
                                      (input_shape[2] - original_shape[2]) // 2]])
    elif original_shape[2] != input_shape[2] and original_shape[2] > input_shape[2]:
        original_input = conv2d(original_input, name + 'conv_3', [1, 1, original_shape[2], input_shape[2]],
                                [1, 1, 1, 1],
                                with_bias=False)
    return input + original_input


def residual_bottleneck_block(input, name, is_training, output_channel, kernel=3, stride=1, first_block=False,
                              padding_option=0, pad_or_conv=False):
    original_shape = input.get_shape().as_list()[1:]
    original_input = input

    if first_block:
        input = conv2d(input, name + 'conv_1', kernel_size=[1, 1, original_shape[2], output_channel / 4],
                       strides=[1, stride, stride, 1], padding='SAME', initial='xavier', with_bias=False)
    else:
        input = batch_normalization(input, is_training)
        input = tf.nn.relu(input)
        input = conv2d(input, name + 'conv_1', kernel_size=[1, 1, original_shape[2], output_channel / 4],
                       strides=[1, stride, stride, 1], padding='SAME', initial='xavier', with_bias=False)

    input = batch_normalization(input, is_training)
    input = tf.nn.relu(input)
    input = conv2d(input, name + 'conv_2', kernel_size=[kernel, kernel, output_channel / 4, output_channel / 4],
                   strides=[1, 1, 1, 1],
                   padding='SAME', initial='xavier', with_bias=False)

    input = batch_normalization(input, is_training)
    input = tf.nn.relu(input)
    input = conv2d(input, name + 'conv_3', kernel_size=[kernel, kernel, output_channel / 4, output_channel],
                   strides=[1, 1, 1, 1],
                   padding='SAME', initial='xavier', with_bias=False)

    if stride != 1:
        original_input = avg_pool(original_input, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
    input_shape = input.get_shape().as_list()[1:]

    if original_shape[2] != input_shape[2] and original_shape[2] < input_shape[2]:
        if pad_or_conv == True:
            original_input = conv2d(original_input, name + 'conv_4', [1, 1, original_shape[2], input_shape[2]],
                                    [1, 1, 1, 1], with_bias=False)
        else:
            original_input = tf.pad(original_input, [[padding_option, padding_option], [padding_option, padding_option],
                                                     [padding_option, padding_option],
                                                     [(input_shape[2] - original_shape[2]) // 2,
                                                      (input_shape[2] - original_shape[2]) // 2]])
    elif original_shape[2] != input_shape[2] and original_shape[2] > input_shape[2]:
        original_input = conv2d(original_input, name + 'conv_4', [1, 1, original_shape[2], input_shape[2]],
                                [1, 1, 1, 1], with_bias=False)

    return input + original_input


def residual_wide_block(input, name, is_training, output_channel, kernel=3, stride=1, first_block=False,
                        padding_option=0, pad_or_conv=False, dropout_rate=0.5):
    original_shape = input.get_shape().as_list()[1:]
    original_input = input
    if first_block:
        input = conv2d(input, name + 'conv_1', kernel_size=[kernel, kernel, original_shape[2], output_channel],
                       strides=[1, stride, stride, 1], padding='SAME', initial='xavier', with_bias=False)
    else:
        input = batch_normalization(input, is_training)
        input = tf.nn.relu(input)
        input = conv2d(input, name + 'conv_1', kernel_size=[kernel, kernel, original_shape[2], output_channel],
                       strides=[1, stride, stride, 1], padding='SAME', initial='xavier', with_bias=False)

    input = tf.nn.dropout(input, dropout_rate)

    input = batch_normalization(input, is_training)
    input = tf.nn.relu(input)
    input = conv2d(input, name + 'conv_2', kernel_size=[kernel, kernel, output_channel, output_channel],
                   strides=[1, 1, 1, 1],
                   padding='SAME', initial='xavier', with_bias=False)

    if stride != 1:
        original_input = avg_pool(original_input, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
    input_shape = input.get_shape().as_list()[1:]

    if original_shape[2] != input_shape[2] and original_shape[2] < input_shape[2]:
        if pad_or_conv == True:
            original_input = conv2d(original_input, name + 'conv_3', [1, 1, original_shape[2], input_shape[2]],
                                    [1, 1, 1, 1],
                                    with_bias=False)
        else:
            original_input = tf.pad(original_input,
                                    [[padding_option, padding_option], [padding_option, padding_option],
                                     [padding_option, padding_option],
                                     [(input_shape[2] - original_shape[2]) // 2,
                                      (input_shape[2] - original_shape[2]) // 2]])
    elif original_shape[2] != input_shape[2] and original_shape[2] > input_shape[2]:
        original_input = conv2d(original_input, name + 'conv_3', [1, 1, original_shape[2], input_shape[2]],
                                [1, 1, 1, 1],
                                with_bias=False)

    return input + original_input


def inception_naive_block(input, name, is_training, out_channel={'1': 32, '3': 32, '5': 32}, strides=[1, 1, 1, 1]):
    input_shape = input.get_shape().as_list()[1:]
    con_1x1 = conv2d(input, name + '_conv1X1', kernel_size=[1, 1, input_shape[2], out_channel['1']], strides=strides,
                     initial='xavier')
    con_3x3 = conv2d(input, name + '_conv3X3', kernel_size=[3, 3, input_shape[2], out_channel['3']], strides=strides,
                     initial='xavier')
    con_5x5 = conv2d(input, name + '_conv5X5', kernel_size=[5, 5, input_shape[2], out_channel['5']], strides=strides,
                     initial='xavier')
    pool = max_pool(input, size=[1, 3, 3, 1], strides=strides)
    output = tf.concat([con_1x1, con_3x3, con_5x5, pool], 3)
    return output


def inception_v2_block(input, name, is_training, strides=[1, 1, 1, 1], out_channel={'1': 32, '3': 32, '5': 32},
                       reduced_out_channel={'3': 32, '5': 32, 'p': 32}):
    input_shape = input.get_shape().as_list()[1:]
    con_1x1 = conv2d(input, name + '_conv1X1', kernel_size=[1, 1, input_shape[2], out_channel['1']], strides=strides,
                     initial='xavier')
    for_3x3_con_1x1 = conv2d(input, name + '_conv1X1_3X3', kernel_size=[1, 1, input_shape[2], reduced_out_channel['3']],
                             strides=strides, initial='xavier')
    con_3x3 = conv2d(for_3x3_con_1x1, name + '_conv3X3', kernel_size=[3, 3, reduced_out_channel['3'], out_channel['3']],
                     strides=strides,
                     initial='xavier')
    for_5x5_con_1x1 = conv2d(input, name + '_conv1X1_5X5', kernel_size=[1, 1, input_shape[2], reduced_out_channel['5']],
                             strides=strides, initial='xavier')
    con_5x5 = conv2d(for_5x5_con_1x1, name + '_conv5X5', kernel_size=[5, 5, reduced_out_channel['5'], out_channel['5']],
                     strides=strides,
                     initial='xavier')
    pool = max_pool(input, size=[1, 3, 3, 1], strides=strides)
    max_1x1 = conv2d(pool, name + '_conv1X1_max', kernel_size=[1, 1, input_shape[2], reduced_out_channel['p']],
                     strides=strides,
                     initial='xavier')
    output = tf.concat([con_1x1, con_3x3, con_5x5, max_1x1], 3)
    return output


def inception_v3_block(self, input, name, is_training, strides=[1, 1, 1, 1],
                       out_channel={'1': 32, '3_1': 32, '3_2': 32},
                       reduced_out_channel={'3_1': 32, '3_2': 32, 'p': 32}):
    input_shape = input.get_shape().as_list()[1:]
    con_1x1 = self.conv2d(input, name + '_conv1X1', kernel_size=[1, 1, input_shape[2], out_channel['1']],
                          strides=strides,
                          initial='xavier')
    for_3x3_con_1x1 = self.conv2d(input, name + '_conv1X1_3X3_1',
                                  kernel_size=[1, 1, input_shape[2], reduced_out_channel['3_1']],
                                  strides=strides, initial='xavier')
    con_3x3 = self.conv2d(for_3x3_con_1x1, name + '_conv_3X3_1',
                          kernel_size=[3, 3, reduced_out_channel['3_1'], out_channel['3_1']],
                          strides=strides,
                          initial='xavier')
    for_3x3_2_con_1x1 = self.conv2d(input, name + '_conv1X1_3X3_2',
                                    kernel_size=[1, 1, input_shape[2], reduced_out_channel['3_2']],
                                    strides=strides, initial='xavier')
    con_3x3_2 = self.conv2d(for_3x3_2_con_1x1, name + '_conv3X3_2',
                            kernel_size=[3, 3, reduced_out_channel['3_2'], out_channel['3_2']],
                            strides=strides,
                            initial='xavier')
    con_3x3_2 = self.conv2d(con_3x3_2, name + '_conv3X3_3', kernel_size=[3, 3, out_channel['3_2'], out_channel['3_2']],
                            strides=strides,
                            initial='xavier')
    pool = self.max_pool(input, size=[1, 3, 3, 1], strides=strides)
    max_1x1 = self.conv2d(pool, name + '_conv1X1max', kernel_size=[1, 1, input_shape[2], reduced_out_channel['p']],
                          strides=strides,
                          initial='xavier')
    output = tf.concat([con_1x1, con_3x3, con_3x3_2, max_1x1], 3)
    return output


def helixnet_block(input,is_training,first_block=False,output_channel=[16,16,16,16]):
    output = []
    if first_block:
        input_shape = input.get_shape().as_list()[1:]
        temp=conv2d_dense_block(input, is_training, kernel=[3, 3, input_shape[2], output_channel[0]], strides=[1, 1, 1, 1], layers=5)
        output.append(residual_bottleneck_block(input, output_channel[1], kernel=3, stride=1, first_block=first_block,pad_or_conv=True))
        output.append(inception_v3_block(input, is_training, strides=[1, 1, 1, 1],
                        out_channel={'1': output_channel[2], '3_1': output_channel[2], '3_2': output_channel[2]},
                           reduced_out_channel={'3_1': output_channel[2], '3_2': output_channel[2], 'p': output_channel[2]}))
        output.append(conv2d(input,[1,1,input_shape[2],output_channel[3]],[1,1,1,1],with_bias=False))
    else:

        input1=tf.concat([input[0],input[1]],axis=3)
        input1 = batch_normalization(input1, is_training)
        input1 = tf.nn.relu(input1)
        input_shape = input1.get_shape().as_list()[1:]
        temp=conv2d_dense_block(input1, is_training, kernel=[3, 3, input_shape[2], output_channel[0]], strides=[1, 1, 1, 1], layers=5)

        input2 = tf.concat([input[1], input[2]], axis=3)
        input2 = batch_normalization(input2, is_training)
        input2 = tf.nn.relu(input2)
        output.append(residual_bottleneck_block(input2, output_channel[1], kernel=3, stride=1, first_block=first_block,pad_or_conv=True))

        input3 = tf.concat([input[2], input[3]], axis=3)
        input3 = batch_normalization(input3, is_training)
        input3 = tf.nn.relu(input3)
        output.append(inception_v3_block(input3, is_training, strides=[1, 1, 1, 1],
                        out_channel={'1': output_channel[2], '3_1': output_channel[2], '3_2': output_channel[2]},
                           reduced_out_channel={'3_1': output_channel[2], '3_2': output_channel[2], 'p': output_channel[2]}))

        input4 = tf.concat([input[3], input[0]], axis=3)
        input4 = batch_normalization(input4, is_training)
        input4 = tf.nn.relu(input4)
        input_shape = input4.get_shape().as_list()[1:]
        output.append(conv2d(input4,[1,1,input_shape[2],output_channel[3]],[1,1,1,1],with_bias=False))
    output.append(temp)


def get_no_of_parameter():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def pre_process(images, is_training):
    def pre_process_image(image, is_training):
        # This function takes a single image as input,
        # and a boolean whether to build the training or testing graph.
        img_size_cropped = 32
        if is_training:
            # For training, add the following to the TensorFlow graph.

            # Randomly crop the input image.
            image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, 3])

            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)

            # Randomly adjust hue, contrast and saturation.
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

            # Some of these functions may overflow and result in pixel
            # values beyond the [0, 1] range. It is unclear from the
            # documentation of TensorFlow 0.10.0rc0 whether this is
            # intended. A simple solution is to limit the range.

            # Limit the image pixels between [0, 1] in case of overflow.
            image = tf.minimum(image, 1.0)
            image = tf.maximum(image, 0.0)
        else:
            # For training, add the following to the TensorFlow graph.

            # Crop the input image around the centre so it is the same
            # size as images that are randomly cropped during training.
            image = tf.image.resize_image_with_crop_or_pad(image,
                                                           target_height=img_size_cropped,
                                                           target_width=img_size_cropped)
        return image

    images = tf.map_fn(lambda image: pre_process_image(image, is_training), images)
    return images
