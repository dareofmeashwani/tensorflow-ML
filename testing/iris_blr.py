import numpy as np
import sys
sys.path.insert(0,'../')
import binary_logistic_regression,datasets

data=datasets.read_iris(2)
train={}
train['x']=data['train_x']
train['y']=data['train_y']
val={}
val['x']=data['test_x']
val['y']=data['test_y']
print(train['x'].shape,train['y'].shape,val['x'].shape,val['y'].shape)

model=binary_logistic_regression.model()
model.no_of_features=4
model.batch_size=160
model.epochs=250
model.learning_rate=0.009
model.model_restore=False
model.working_dir='iris_blr'
model.setup()
model.train(train,val,shuffle=True)
pred,acc=model.predict(val)
print pred
print acc
model.clear()

