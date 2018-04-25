import os,sys,random
import tensorflow as tf
import numpy as np
sys.path.insert(0,'../')
import image_ops,utils,cnn,ops
image_handler=image_ops.data_augmentor()

def read_flower(path='/media/batman/ent/datasets/flower',size=64):
    files_path=os.path.join(path,'files.txt')
    files=open(files_path,'r').readlines()
    data={}
    i=0
    index=0
    temp=[]
    for file in files:
        file=file.replace('\n','')
        file=os.path.join(path,os.path.join('jpg',file))
        img=image_handler.read_image(file)
        img=image_handler.resize_image_with_aspect_ratio(img,size)
        temp.append(img)
        i+=1
        if i%80==0:
            data[str(index)]=temp
            temp=[]
            index+=1
            i=0
    temp=None
    return data


#data_path='/media/batman/ent/datasets/flower'
#data=read_flower(data_path,128)
#utils.save_model(data,'flower.pickle')
data=utils.load_encoding_model('flower.pickle','bytes')
train_x=[]
train_y=[]
val_x=[]
val_y=[]
for k in data.keys():
    #random.shuffle(data[k])
    for i in range(len(data[k])):
        if i<(len(data[k])*0.85):
            train_x.append(data[k][i])
            train_y.append(int(k))
        else:
            val_x.append(data[k][i])
            val_y.append(int(k))


train={}
val={}
train['x']=np.array(train_x).reshape([-1,128,128,3])
train['y']=utils.convert_to_onehot(np.array(train_y),17)
val['x']=np.array(val_x).reshape([-1,128,128,3])
val['y']=utils.convert_to_onehot(np.array(val_y),17)

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
        conv = ops.get_hidden_layer(conv, 'output_layer',17, activation="none", initializer='xavier')
    return conv

import multi_label_cnn
model=multi_label_cnn.model()
model.batch_size=64
model.epochs=10
model.image_shape=[128,128,3]
model.learning_rate=0.00001
model.no_of_classes=17
model.working_dir='result_res18'
model.model_restore=False
model.logits=model_fun
model.regularization_type='l2'
model.regularization_coefficient=0.001

model.type='lenet'
model.setup()
#model.predict(val)
#model.train(train,val,max_keep=100)
model.clear()

