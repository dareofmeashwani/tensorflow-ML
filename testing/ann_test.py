import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0,'../')

import ops
import ann_classifier
import multi_class_logistic_regression
import rnn_classifier
import datasets
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
train['x']=train_x.reshape([-1,28,28])
train['y']=train_y#convert_to_onehot(train_y,10)
val['x']=val_x.reshape([-1,28,28])
val['y']=val_y#convert_to_onehot(val_y,10)


model=rnn_classifier.model()
model.batch_size=256
model.epochs=10
model.sequence_dimensions=28
model.sequence_length=28
model.cell_size=[32,32]
model.no_of_cell=2
model.learning_rate=0.0005
model.no_of_classes=10
model.working_dir='result_res18'
model.model_restore=False

model.setup()
#model.predict(val)
model.train(train,val,max_keep=100)
model.clear()
