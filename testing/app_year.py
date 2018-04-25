import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'../')
import ann_regressor

#df=pd.read_csv('/media/batman/ent/datasets/YearPredictionMSD.txt',header=None)
df=pd.read_csv('H:\datasets\YearPredictionMSD.txt',header=None)
y=np.array(df[0])-1922
print('min year',np.min(y))
del df[0]
x=np.array(df)

mean=116.281729335#np.mean(x)
std=508.480934755#np.std(x)
print(mean,std)
x=(x-mean)/std

train={}
val={}
train['x']=x[0:int(len(x)*.8)]
val['x']=x[int(len(x)*.8):]
train['y']=y[0:int(len(x)*.8)].reshape([-1,1])
val['y']=y[int(len(x)*.8):].reshape([-1,1])


print(train['x'].shape,train['y'].shape,val['x'].shape,val['y'].shape)



model=ann_regressor.model()
model.no_of_features=90
model.batch_size=160
model.epochs=500
model.learning_rate=0.0001
model.model_restore=False
model.working_dir='nnr_wine'
model.hidden_layers=[50,50,10]
model.activation_list=['leaky_relu','leaky_relu','leaky_relu']
model.setup()
#model.train(train,val,shuffle=True)
pred=model.predict(val)
for i in range(len(pred[0:10])):
    print(pred[i],val['y'][i])

model.clear()
