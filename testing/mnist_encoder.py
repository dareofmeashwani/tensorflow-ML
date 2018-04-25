import numpy as np
import sys
sys.path.insert(0,'../')
import datasets,autoencoder,temp

data=datasets.read_mnist("H:/datasets/mnist/")
#data=datasets.read_mnist('/media/batman/ent/datasets/mnist/',one_hot=True)

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



model=temp.model()
model.batch_size=32
model.epochs=10
model.no_of_features=28*28
model.learning_rate=0.001
model.working_dir='encoder_mnist'
model.model_restore=False
model.encoder_hidden_layers=[400,150]
model.encoder_activation_list=['leaky_relu','leaky_relu']
model.decoder_hidden_layers=[150,400]
model.decoder_activation_list=['leaky_relu','leaky_relu']
model.setup()
model.train({'x':train_x,'y':train_y},{'x':val_x,'y':val_y},max_keep=100)
#print (
#model.get_encoded_vector(val_x)
model.clear()