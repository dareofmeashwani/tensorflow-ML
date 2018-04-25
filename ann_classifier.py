import tensorflow as tf
import numpy as np
import random
import ops
import heapq
lower = str.lower
class model:
    no_of_features = None
    learning_rate = 0.001
    model_restore = False
    working_dir = None
    batch_size = 64
    epochs = 10
    test_result = []
    train_result = []
    activation_list=[]
    hidden_layers=[]

    no_of_classes = None
    dropout_rate=0.5

    loss_type = 'softmax'
    regularization_type = None
    regularization_coefficient = 0.0001
    logits=None
    optimizer=None

    def __init__(self):
        return

    def setup(self):
        tf.reset_default_graph()
        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[None, self.no_of_features],
                                name="input")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.no_of_classes], name="labels")
        self.lr = tf.placeholder("float", shape=[])
        self.is_train = tf.placeholder(tf.bool, shape=[])

        if self.logits==None:
            self.logits=self.get_model(self.x,self.is_train)
        else:
            self.logits=self.logits(self.x,self.is_train)
        with tf.name_scope('Output'):
            self.cross_entropy = ops.get_loss(self.logits, self.y, self.loss_type)
            if self.regularization_type != None:
                self.cross_entropy = ops.get_regularization(self.cross_entropy, self.regularization_type,
                                                            self.regularization_coefficient)
            self.probability = tf.nn.softmax(self.logits, name="softmax")
            self.prediction = tf.argmax(self.probability, 1, name='Prediction')
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.y, 1), name='Correct_prediction')
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')
            tf.summary.scalar("Cross_Entropy", self.cross_entropy)
            tf.summary.scalar("Accuracy", self.accuracy)

        with tf.name_scope('Optimizer'):
            if self.optimizer==None:
                # learningRate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=1,
                #                                          decay_steps=shape[0], decay_rate=0.97, staircase=True,
                #                                          name='Learning_Rate')
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
                # optimizer = tf.train.MomentumOptimizer(lr, .9, use_nesterov=True).minimize(cross_entropy)
                self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
                # optimizer = tf.train.AdadeltaOptimizer(lr).minimize(cross_entropy)
        self.session = tf.InteractiveSession()
        return

    def get_model(self,x,is_training):
        op=ops.get_n_hidden_layers(x,'',self.hidden_layers,self.activation_list,'xavier')
        return ops.get_hidden_layer(op,'output_layer',self.no_of_classes,'none','xavier')


    def get_paramter_count(self):
        return ops.get_no_of_parameter()

    def clear(self):
        tf.reset_default_graph()
        self.session.close()

    def train(self,train,val_data=None,max_keep=100,shuffle=False):
        init = tf.global_variables_initializer()
        self.session.run(init)
        epoch_offset=0

        saver = tf.train.Saver(max_to_keep=max_keep)
        if self.model_restore == True and self.working_dir!= None:
            name = ops.look_for_last_checkpoint(self.working_dir + "/model/")
            if name is not None:
                saver.restore(self.session, self.working_dir + "/model/" + name)
                print('Model Succesfully Loaded : ', name)
                epoch_offset=int(name[6:])

        if self.working_dir != None:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.working_dir + '/train', self.session.graph)
            test_writer = tf.summary.FileWriter(self.working_dir + '/test')

        for epoch in range(epoch_offset+1,epoch_offset+self.epochs+1):
            if shuffle==True:
                ind_list = [i for i in range(len(train['x']))]
                random.shuffle(ind_list)
                train['x']=train['x'][ind_list]
                train['y']=train['y'][ind_list]
            epoch_loss = 0
            acc = 0
            i = 0
            batch_iteration = 0
            while i < len(train['x']):
                start = i
                end = i + self.batch_size
                if (end > len(train['x'])): end = len(train['x'])
                batch_x = train['x'][start:end]
                batch_y = train['y'][start:end]
                if self.working_dir != None:
                    summary, _, loss,batch_acc= self.session.run([merged, self.optimizer,self.cross_entropy,self.accuracy],
                                                              feed_dict={self.x: batch_x, self.y: batch_y,self.lr:self.learning_rate,self.is_train:True})
                else:
                    _, loss,batch_acc= self.session.run([self.optimizer, self.cross_entropy,self.accuracy],
                                                     feed_dict={self.x: batch_x,self.y: batch_y,self.lr: self.learning_rate,self.is_train:True})
                epoch_loss += loss
                acc += batch_acc
                batch_iteration += 1
                i += self.batch_size
                print('Training: Accuracy={} loss={}\r '.format(round(batch_acc,4),round(epoch_loss/batch_iteration,4)),)
            if self.working_dir != None:
                train_writer.add_summary(summary, epoch)
            self.train_result.append([epoch, epoch_loss/batch_iteration, acc / batch_iteration])
            if val_data != None:
                epoch_loss = 0
                acc = 0
                i = 0
                batch_iteration = 0
                while i < len(val_data['x']):
                    start = i
                    end = i + self.batch_size
                    if (end > len(val_data['x'])): end = len(val_data['x'])
                    batch_x = val_data['x'][start:end]
                    batch_y = val_data['y'][start:end]
                    if self.working_dir != None:
                        summary,loss,batch_acc= self.session.run([merged,self.cross_entropy,self.accuracy],
                                                              feed_dict={self.x: batch_x, self.y: batch_y,self.lr:self.learning_rate,self.is_train:False})
                    else:
                        loss,batch_acc= self.session.run([self.cross_entropy,self.accuracy],
                                                     feed_dict={self.x: batch_x,self.y: batch_y,self.lr: self.learning_rate,self.is_train:False})
                    epoch_loss += loss

                    acc += batch_acc
                    batch_iteration += 1
                    i += self.batch_size
                    print('Validation: Accuracy={} loss={}\r '.format(round(batch_acc, 4),round(epoch_loss / batch_iteration, 4)),)
                if self.working_dir != None:
                    test_writer.add_summary(summary, epoch)
                self.test_result.append([epoch, epoch_loss/batch_iteration, acc / batch_iteration])

                print("Training:", self.train_result[len(self.train_result) - 1], "Val:", self.test_result[len(self.test_result) - 1])
            else:
                print("Training :", self.train_result[len(self.train_result) - 1])

            if self.working_dir != None:
                save_path = saver.save(self.session, self.working_dir + "/model/" + 'model',global_step=epoch)
        print('Training Succesfully Complete')

    def check_restore(self):
        saver = tf.train.Saver()
        print(self.working_dir+'/model/')
        try:
            saver.restore(self.session, tf.train.latest_checkpoint(self.working_dir+'/model/'))
            return True
        except:
            return False

    def predict(self,test):
        saver = tf.train.Saver()
        saver.restore(self.session, tf.train.latest_checkpoint(self.working_dir+'/model/'))
        print ('Retored model',ops.look_for_last_checkpoint(self.working_dir + "/model/"))
        merged = tf.summary.merge_all()
        if 'x' in test and test['x'].shape[0] > 0:
            i = 0
            iteration = 0
            acc = 0
            test_prediction=[]
            j=0
            while i < len(test['x']):
                start = i
                end = i + self.batch_size
                if (end > len(test['x'])): end = len(test['x'])
                batch_x = test['x'][start:end]
                if 'y' in test and test['y'].shape[0] > 0:
                    batch_y = test['y'][start:end]
                    pred,batch_acc = self.session.run([self.prediction,self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y,self.is_train:False})
                    acc += batch_acc
                else:
                    pred= self.session.run([self.prediction], feed_dict={self.x: batch_x,self.is_train:False})
                iteration += 1
                i += self.batch_size
                if isinstance(pred,list):
                    test_prediction+=pred[0].tolist()
                else:
                    test_prediction += pred.tolist()
            if 'y' in test and test['y'].shape[0] > 0:
                return np.array(test_prediction), acc/iteration
            else:
                return np.array(test_prediction)
