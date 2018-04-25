import autoencoder
import numpy as np

train=None


model=autoencoder.autoencoder()
model.no_of_features=len(train[0])
model.learning_rate=0.001
model.batch_size=128
model.epochs=100
model.model_restore=False
model.working_dir='encoder_result'

model.setup()
model.train(train)


encoded_data,loss,sim=model.get_encoded_vector(train)

model.clear()


