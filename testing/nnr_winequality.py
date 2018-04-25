import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'../')
import ann_regressor,linear_regression
df=pd.read_csv('/media/batman/ent/datasets/winequality-white.csv',delimiter=';')


train={}
val={}
y=np.array(df['quality'])
del df['quality']
x=np.array(df)

train['x']=x[0:int(len(x)*.8)]
val['x']=x[int(len(x)*.8):]
train['y']=y[0:int(len(x)*.8)].reshape([-1,1])
val['y']=y[int(len(x)*.8):].reshape([-1,1])

print(train['x'].shape,train['y'].shape,val['x'].shape,val['y'].shape)



model=linear_regression.model()
model.no_of_features=11
model.batch_size=160
model.epochs=500
model.learning_rate=0.0001
model.model_restore=False
model.working_dir='nnr_wine'
#model.hidden_layers=[10,5]
#model.activation_list=['relu','relu']
model.setup()
model.train(train,val,shuffle=True)
pred=model.predict(val)
for i in range(len(pred)):
    print(pred[i],val['y'][i])

model.clear()
