import numpy as np
import tensorflow as tf
class rnnr:
    __batch_size=None
    __datatype=tf.float32
    __epochs=10
    __hidden_layers=None
    __activation_choice=None
    __lstm_unit_size=32
    __lstm_units_count=1
    __learning_rate = 0.1
    __dropout_rate=.8
    __sequence_length=None
    __sequence_dimensions = 10
    __working_dir='./'
    __model_saving=False
    __model_restore=False
    __shape=None
    training_loss_history=None
    test_loss_history = None
    def __init__(self):
        return
    def set_batch_size(self,value):
        self.__batch_size=value
    def set_datatype(self,value):
        self.__datatype=value
    def set_epochs(self,value):
        self.__epochs=value
    def set_hidden_layers(self,list,activation_choice_list):
        self.__hidden_layers=list
        self.__activation_choice=activation_choice_list
    def set_lstm_defination(self,size,count):
        self.__lstm_unit_size=size
        self.__lstm_units_count=count
    def set_learningrate(self,value):
        self.__learning_rate=value
    def set_dropout(self,value):
        self.__dropout_rate=value
    def set_sequence_defination(self,length,dimension):
        self.__sequence_length=length
        self.__sequence_dimensions=dimension
    def set_save_summary_path(self,path):
        self.__working_dir=path
        print 'run this command on terminal'
        print 'tensorboard --logdir='+path
        self.__model_saving = True
    def set_model_restore_state(self,state):
        self.__model_restore=state
    def set_train_shape(self,shape):
        self.__shape=shape
    def __choose_activation_function(self,node, choice):
        if choice == 0:
            return node
        if choice == 1:
            return tf.nn.relu(node)
        if choice == 2:
            return tf.nn.sigmoid(node)
        if choice == 3:
            return tf.nn.tanh(node)
        if choice == 4:
            return tf.nn.dropout(node,self.__dropout_rate)
        if choice == 5:
            return tf.nn.crelu(node)
        if choice == 6:
            return tf.nn.relu6(node)
        if choice == 7:
            return tf.nn.elu(node)
        if choice == 8:
            return tf.nn.softplus(node)
        if choice == 9:
            return tf.nn.softsign(node)
        if choice == 10:
            return tf.nn.softmax(logits=node)
    def get_activation_function_name(self,choice):
        if choice == 0:
            return 'None'
        if choice == 1:
            return 'Relu'
        if choice == 2:
            return 'Sigmoid'
        if choice == 3:
            return 'Tanh'
        if choice == 4:
            return 'dropout'
        if choice == 5:
            return 'crelu6'
        if choice == 6:
            return 'Relu6'
        if choice == 7:
            return 'Elu'
        if choice == 8:
            return 'softplus'
        if choice == 9:
            return 'softsign'
        if choice == 10:
            return "softmax"
    def __look_for_last_checkpoint(self,mode_dir):
        try:
            fr=open(mode_dir+'checkpoint',"r")
        except:
            return None
        f_line=fr.readline()
        start=f_line.find('"')
        end=f_line.rfind('"')
        return f_line[start+1:end]


    def __variable_summaries(self,var,name):
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
              stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
    def __lstm(self,x):
        with tf.name_scope('LSTM_Layer' ):
            lstm_cells=[]
            for i in range(self.__lstm_units_count):
                lstm_cells.append(tf.contrib.rnn.BasicLSTMCell(self.__lstm_unit_size, forget_bias=1.0, state_is_tuple=True))
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)
            value, state = tf.nn.dynamic_rnn(stacked_lstm,x, dtype=tf.float32)
            value = tf.transpose(value, [1, 0, 2])
            self.__variable_summaries(value, "LSTM_output")
            #last = tf.gather(value, int(value.get_shape()[0]) - 1)
            last=value[-1]
            self.__variable_summaries(last, "LSTM_Last")
        return last
    def __fc(self,x):
        hidden_layer_weights_biases = {}
        layer_output = {}
        try:
            no_of_layers=len(self.__hidden_layers)
        except:
            no_of_layers=0
        for i in range(no_of_layers):
            with tf.name_scope('Hidden_Layer_'+str(i+1)):
                if i == 0:
                    hidden_layer_weights_biases.update(
                        {'weights' + str(i): tf.Variable(tf.truncated_normal([self.__lstm_unit_size, self.__hidden_layers[i]]),name='weight'),
                         'biases' + str(i): tf.Variable(tf.truncated_normal([1, self.__hidden_layers[i]]),name='biases')})
                    self.__variable_summaries(hidden_layer_weights_biases['weights' + str(i)],"weights")
                    self.__variable_summaries(hidden_layer_weights_biases['biases' + str(i)],"biases")
                    mul=tf.matmul(x, hidden_layer_weights_biases['weights' + str(i)])
                    add=tf.add(mul,hidden_layer_weights_biases['biases' + str(i)])
                    act=self.__choose_activation_function(add,self.__activation_choice[i])
                    layer_output.update({str(i):act})
                    self.__variable_summaries(mul,"Multiply")
                    self.__variable_summaries( add,"Add")
                    self.__variable_summaries(act,self.get_activation_function_name(self.__activation_choice[i]))

                else:
                    hidden_layer_weights_biases.update(
                        {'weights' + str(i): tf.Variable(tf.truncated_normal([self.__hidden_layers[i - 1], self.__hidden_layers[i]]),name='weight'),
                         'biases' + str(i): tf.Variable(tf.truncated_normal([1, self.__hidden_layers[i]]),name='biases')})
                    self.__variable_summaries(hidden_layer_weights_biases['weights' + str(i)],"weights")
                    self.__variable_summaries(hidden_layer_weights_biases['biases' + str(i)],"biases")
                    mul=tf.matmul(layer_output[str(i - 1)], hidden_layer_weights_biases['weights' + str(i)])
                    add=tf.add(mul,hidden_layer_weights_biases['biases' + str(i)])
                    act=self.__choose_activation_function(add, self.__activation_choice[i])
                    layer_output.update({str(i):act})
                    self.__variable_summaries(mul,"Multiply")
                    self.__variable_summaries( add,"Add")
                    self.__variable_summaries(act, self.get_activation_function_name(self.__activation_choice[i]))

        with tf.name_scope('Output_Layer'):
            if self.__hidden_layers is not None and no_of_layers!=0:
                output_layer = {'weights': tf.Variable(tf.random_normal([self.__hidden_layers[len(self.__hidden_layers) - 1],1]),name='weight'),
                                'biases': tf.Variable(tf.random_normal([1]), name='biases')}
                self.__variable_summaries(output_layer['weights'], "weights")
                self.__variable_summaries(output_layer['biases'], "biases")
                mul=tf.matmul(layer_output[str(len(self.__hidden_layers) - 1)], output_layer['weights'])
            else:
                output_layer = {'weights': tf.Variable(tf.random_normal([self.__lstm_unit_size,1]),name='weight'),
                                'biases': tf.Variable(tf.random_normal([1]), name='biases')}
                self.__variable_summaries(output_layer['weights'], "weights")
                self.__variable_summaries(output_layer['biases'], "biases")
                mul = tf.matmul(x, output_layer['weights'])
            logits= tf.add(mul,output_layer['biases'],name="logits")
            self.__variable_summaries(mul, "Multiply")
            self.__variable_summaries(logits, "logits")
        return logits
    def pretrained_test(self,data):
        tf.reset_default_graph()
        x = tf.placeholder(dtype=self.__datatype, shape=[None, self.__sequence_length,self.__sequence_dimensions],name="input")
        y = tf.placeholder(dtype=self.__datatype, shape=[None],name="labels")
        logits = self.__fc(self.__lstm(x))

        with tf.name_scope('Total'):
            cross_entropy = tf.reduce_mean(tf.square(tf.subtract(logits, y)), name='Cross_entropy')
            #cross_entropy = tf.reduce_mean(tf.abs(tf.subtract(logits, y)), name='Cross_entropy')
            # cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y,name='Cross_entropy'))
            tf.summary.scalar("Cross_Entropy", cross_entropy)

        with tf.name_scope('Optimizer'):
            learningRate = tf.train.exponential_decay(learning_rate=self.__learning_rate,global_step=1,
                                                  decay_steps=self.__shape[0],decay_rate=0.97,staircase=True,name='Learning_Rate')
            #optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cross_entropy)
            optimizer = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)

        session = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(self.__working_dir + "/model/"))
        merged = tf.summary.merge_all()
        if data.has_key('test_x') and data['test_x'].shape[0] > 0:
            i = 0
            iteration = 0
            epoch_loss = 0
            test_prediction = []
            while i < len(data['test_x']):
                start = i
                end = i + self.__batch_size
                if (end > len(data['test_x'])): end = len(data['test_x'])
                batch_x = data['test_x'][start:end]
                if data.has_key('test_y') and data['test_y'].shape[0] > 0:
                    batch_y = data['test_y'][start:end]
                    pred, loss = session.run([logits, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                    epoch_loss += loss
                else:
                    pred = session.run([logits], feed_dict={x: batch_x})
                iteration += 1
                i += self.__batch_size
                test_prediction += list(pred)
            if data.has_key('test_y') and data['test_y'].shape[0] > 0:
                return np.reshape(np.array(test_prediction), [-1]), epoch_loss
            else:
                return np.reshape(np.array(test_prediction), [-1])
    def train(self,data):
        if self.__batch_size == None: self.__batch_size = data['train_x'].shape[0]
        tf.reset_default_graph()

        x = tf.placeholder(dtype=self.__datatype, shape=[None, self.__sequence_length,self.__sequence_dimensions],name="input")
        y = tf.placeholder(dtype=self.__datatype, shape=[None],name="labels")
        logits = self.__fc(self.__lstm(x))

        with tf.name_scope('Total'):
            cross_entropy = tf.reduce_mean(tf.square(tf.subtract(logits, y)), name='Cross_entropy')
            #cross_entropy = tf.reduce_mean(tf.abs(tf.subtract(logits, y)), name='Cross_entropy')
            # cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y,name='Cross_entropy'))
            tf.summary.scalar("Cross_Entropy", cross_entropy)

        with tf.name_scope('Optimizer'):
            learningRate = tf.train.exponential_decay(learning_rate=self.__learning_rate,global_step=1,
                                                  decay_steps=self.__shape[0],decay_rate=0.97,staircase=True,name='Learning_Rate')
            #optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cross_entropy)
            optimizer = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)

        init = tf.global_variables_initializer()
        session = tf.InteractiveSession()
        session.run(init)
        saver = tf.train.Saver()
        if self.__model_restore == True:
            name = self.__look_for_last_checkpoint(self.__working_dir + "/model/")
            if name is not None:
                saver.restore(session, self.__working_dir + "/model/" + name)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.__working_dir + '/train', session.graph)
        test_writer = tf.summary.FileWriter(self.__working_dir + '/test')
        test_result = []
        train_result = []
        for epoch in range(self.__epochs):
            epoch_loss = 0
            i = 0
            iteration = 0
            while i < len(data['train_x']):
                start = i
                end = i + self.__batch_size
                if (end > len(data['train_x'])): end = len(data['train_x'])
                batch_x = data['train_x'][start:end]
                batch_y = data['train_y'][start:end]
                summary, _, loss = session.run([merged, optimizer, cross_entropy],
                                                    feed_dict={x: batch_x, y: batch_y})
                epoch_loss += loss
                iteration += 1
                i += self.__batch_size
            train_writer.add_summary(summary, epoch)
            train_result.append([epoch, epoch_loss])
            if data.has_key('test_x') and data['test_x'].shape[0] > 0:
                epoch_loss = 0
                i = 0
                iteration = 0
                while i < len(data['test_x']):
                    start = i
                    end = i + self.__batch_size
                    if (end > len(data['test_x'])): end = len(data['test_x'])
                    batch_x = data['test_x'][start:end]
                    batch_y = data['test_y'][start:end]
                    summary, loss = session.run([merged, cross_entropy],
                                                     feed_dict={x: batch_x, y: batch_y})
                    epoch_loss += loss
                    iteration += 1
                    i += self.__batch_size
                test_writer.add_summary(summary, epoch)
                test_result.append([epoch, epoch_loss])
                print "Training:", train_result[len(train_result) - 1], "Test:", test_result[len(test_result) - 1]
            else:
                print "Training :", train_result[len(train_result) - 1]

            if self.__model_saving == True:
                save_path = saver.save(session, self.__working_dir + "/model/" + 'nnr_model%d.ckpt' % epoch)
                # print("Model saved in file: %s" % save_path)
        print 'Training Succesfully Complete'
        if data.has_key('test_x') and data['test_x'].shape[0] > 0:
            i = 0
            epoch_loss = 0
            iteration = 0
            test_prediction = []
            while i < len(data['test_x']):
                start = i
                end = i + self.__batch_size
                if (end > len(data['test_x'])): end = len(data['test_x'])
                batch_x = data['test_x'][start:end]
                batch_y = data['test_y'][start:end]
                pred, loss= session.run([logits, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += loss
                iteration += 1
                i += self.__batch_size
                test_prediction += list(pred)
            self.test_prediction = np.reshape(np.array(test_prediction), [-1])
            self.test_loss = epoch_loss
        self.training_loss_history = train_result
        self.test_loss_history =test_result