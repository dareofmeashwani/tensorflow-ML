import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0,'../')

import ops
import cnn
import datasets

def model_fun(x,is_training):
    x_shape = x.get_shape().as_list()[1:]
    kernel = {'c1': [5, 5, x_shape[2], 20], 'c2': [5, 5, 20, 50]}
    strides = {'1': [1, 1, 1, 1], '2': [1, 2, 2, 1]}
    pool_win_size = {'2': [1, 2, 2, 1]}

    conv = ops.conv2d(x, 'conv1', kernel['c1'], strides['1'], 'SAME')

    conv=ops.residual_bottleneck_block(conv,'ins_block',is_training,64)

    with tf.variable_scope('Flatten_layer') as scope:
        conv = ops.flatten(conv)
    with tf.variable_scope('Output_layer') as scope:
        conv = ops.get_hidden_layer(conv, 'output_layer',10, activation="none", initializer='xavier')
    return conv



#data=read_mnist("H:/datasets/mnist/")
data=datasets.read_mnist('/media/batman/ent/datasets/mnist/',one_hot=True)

train_x=data['train_x']
train_y=data['train_y']
val_x=data['test_x']
val_y=data['test_y']
mean=np.mean(train_x)
#mean=117.07463261370678
print(mean)
std=np.std(train_x)
#std=78.52984152948856
print(std)

train_x=(train_x-mean)/std
val_x=(val_x-mean)/std


print(train_x.shape,train_y.shape,val_x.shape,val_y.shape)


train={}
val={}
train['x']=train_x.reshape([-1,28,28,1])
train['y']=train_y#convert_to_onehot(train_y,10)
val['x']=val_x.reshape([-1,28,28,1])
val['y']=val_y#convert_to_onehot(val_y,10)


model=cnn.cnn()
model.batch_size=128
model.epochs=10
model.image_shape=[28,28,1]
model.learning_rate=0.0005
model.no_of_classes=10
model.working_dir='result_res18'
model.cnn_type='alexnet'
model.model_restore=False
model.regularization_type = 'l2'
model.logits=model_fun
model.regularization_coefficient = 0.0001
model.setup()
#model.predict(val)
model.train(train,val,max_keep=100)
model.clear()
