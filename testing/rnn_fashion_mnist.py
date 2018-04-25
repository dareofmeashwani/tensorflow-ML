import numpy as np
import pandas as pd
import sys
sys.path.insert(0,'../')
import utils,rnn_classifier

data_train_file='/media/batman/ent/datasets/fashionmnist/fashion-mnist_train.csv'
data_test_file='/media/batman/ent/datasets/fashionmnist/fashion-mnist_test.csv'
train_df=pd.read_csv(data_train_file)
test_df=pd.read_csv(data_test_file)

text_labels=['T-shirt/top'
'Trouser'
'Pullover'
'Dress'
'Coat'
'Sandal'
'Shirt'
'Sneaker'
'Bag'
'Ankle boot']

train={}
val={}
train['y']=utils.convert_to_onehot(np.array(train_df['label']),10)
del train_df['label']
train['x'] = np.reshape(np.array(train_df), [-1,28,28])

val['y']=utils.convert_to_onehot(np.array(test_df['label']),10)
del test_df['label']
val['x'] = np.reshape(np.array(test_df), [-1, 28,28])

print train['x'].shape,train['y'].shape,val['x'].shape,val['y'].shape

model=rnn_classifier.model()
model.batch_size=128
model.epochs=10
model.learning_rate=0.0001
model.sequence_dimensions=28
model.sequence_length=28
model.no_of_cell=2
model.cell_size=32
model.no_of_classes=10
model.model_restore=True
model.hidden_layers=[50]
model.working_dir='rnn_test_f_mnist'
model.activation_list=['leaky_relu']
model.setup()
model.train(train,val)
model.clear()