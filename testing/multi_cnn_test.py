import h5py
from sklearn.model_selection import train_test_split
import sys
import numpy as np
sys.path.insert(0,'../')
import multi_label_cnn,ops,tensorflow as tf
f = h5py.File("dataset.h5")
x = f['x'].value
y = f['y'].value
f.close()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

x_train /= 255
x_test /= 255

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

train={}
val={}
train['x']=np.transpose(x_train,(0,2,3,1))#np.array(train_x).reshape([-1,100,100,3])
train['y']=y_train#utils.convert_to_onehot(np.array(train_y),17)
val['x']=np.transpose(x_test,(0,2,3,1))#np.array(val_x).reshape([-1,128,1,3])
val['y']=y_test#utils.convert_to_onehot(np.array(val_y),17)

print(train['x'].shape,train['y'].shape,val['x'].shape,val['y'].shape)


def model_fun(x,is_training):
    x_shape = x.get_shape().as_list()[1:]
    kernel = {'c1': [5, 5, x_shape[2], 64], 'c2': [5, 5, 20, 50]}
    strides = {'1': [1, 1, 1, 1], '2': [1, 2, 2, 1]}
    pool_win_size = {'2': [1, 2, 2, 1]}

    conv = ops.conv2d(x, 'conv1', kernel['c1'], strides['1'], 'SAME')

    conv =ops.max_pool(conv,[1,3,3,1],[1,1,1,1])

    conv=ops.residual_bottleneck_block(conv,'ins_block',is_training,64)

    with tf.variable_scope('Flatten_layer') as scope:
        conv = ops.flatten(conv)
    with tf.variable_scope('Output_layer') as scope:
        conv = ops.get_hidden_layer(conv, 'output_layer',5, activation="none", initializer='xavier')
    return conv

import multi_label_cnn
model=multi_label_cnn.model()
model.batch_size=64
model.epochs=10
model.image_shape=[100,100,3]
model.learning_rate=0.00001
model.no_of_classes=5
model.working_dir='result_res18'
model.model_restore=False
model.cnn_type='lenet'
#model.logits=model_fun
model.regularization_type='l2'
model.regularization_coefficient=0.001

model.type='lenet'
model.setup()
#model.predict(val)
model.train(train,val,max_keep=100)
model.clear()
