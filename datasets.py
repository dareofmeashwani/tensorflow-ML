import numpy as np
import utils
import sys,os,time
def read_mnist(path,one_hot=False):
    import gzip,os,sys,time
    def extract_data(filename, num_images):
        IMAGE_SIZE = 28
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            # data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
            data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
            return data

    def extract_labels(filename, num_images):
        """Extract the labels into a vector of int64 label IDs."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels.reshape(num_images,1)
    train_x=extract_data(path+'train-images-idx3-ubyte.gz',60000)
    train_y = extract_labels(path + 'train-labels-idx1-ubyte.gz', 60000)
    test_x=extract_data(path+'t10k-images-idx3-ubyte.gz',10000)
    test_y = extract_labels(path + 't10k-labels-idx1-ubyte.gz', 10000)
    if one_hot == True:
        train_y=utils.convert_to_onehot(train_y,10)
        test_y = utils.convert_to_onehot(test_y, 10)
    return {'train_x':train_x.reshape([-1,28*28]),'train_y':train_y,'test_x':test_x.reshape([-1,28*28]),'test_y':test_y}



def read_cifar10(path,one_hot=False):
    import os
    text_labels=utils.load_model(path+'batches.meta')['label_names']
    if os.name == 'nt':
        for i in range(5):
            data=utils.load_encoding_model(path + 'data_batch_'+str(i+1),encode='bytes')
            if i==0:
                train_x=data[b'data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3)
                train_y=np.array(data[b'labels']).reshape(10000,1)
                continue
            train_x= np.vstack((train_x,data[b'data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3)))
            train_y = np.vstack((train_y,np.array(data[b'labels']).reshape(10000,1)))
        data=utils.load_encoding_model(path +'test_batch',encode='bytes')
        test_x=data[b'data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3)
        test_y=np.array(data[b'labels']).reshape(10000,1)
        if one_hot == True:
            train_y=utils.convert_to_onehot(train_y,10)
            test_y = utils.convert_to_onehot(test_y, 10)
    else:
        for i in range(5):
            data=utils.load_model(path + 'data_batch_'+str(i+1))
            if i==0:
                train_x=data['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3)
                train_y=np.array(data['labels']).reshape(10000,1)
                continue
            train_x= np.vstack((train_x,data['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3)))
            train_y = np.vstack((train_y,np.array(data['labels']).reshape(10000,1)))
        data=utils.load_model(path +'test_batch')
        test_x=data['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3)
        test_y=np.array(data['labels']).reshape(10000,1)
        if one_hot == True:
            train_y=utils.convert_to_onehot(train_y,10)
            test_y = utils.convert_to_onehot(test_y, 10)
    return {'train_x':train_x,'train_y':train_y,'test_x':test_x,'test_y':test_y,'text_labels':text_labels}

def read_iris(classes=3,one_hot=False):
    from sklearn.datasets import load_iris
    iris = load_iris()
    x = iris['data']
    y = iris['target']
    train_x1 = x[0:40, :]
    train_x2 = x[50:90, :]
    train_x3 = x[100:140, :]
    train_y1 = y[0:40].reshape(40,1)
    train_y2 = y[50:90].reshape(40,1)
    train_y3 = y[100:140].reshape(40,1)
    test_x1 = x[40:50, :]
    test_x2 = x[90:100, :]
    test_x3 = x[140:150, :]
    test_y1 = y[40:50].reshape(10,1)
    test_y2 = y[90:100].reshape(10,1)
    test_y3 = y[140:150].reshape(10,1)
    train_x = np.vstack((train_x1, train_x2))
    train_y = np.vstack((train_y1, train_y2))
    test_x = np.vstack((test_x1, test_x2))
    test_y = np.vstack((test_y1, test_y2))
    if(classes==3):
        train_x = np.vstack((train_x, train_x3))
        train_y = np.vstack((train_y, train_y3))
        test_x = np.vstack((test_x, test_x3))
        test_y = np.vstack((test_y, test_y3))
    if one_hot == True:
        train_y=utils.convert_to_onehot(train_y,classes)
        test_y = utils.convert_to_onehot(test_y,classes)
    return {'train_x':train_x,'train_y':train_y,'test_x':test_x,'test_y':test_y}

def read_cifar100(path,one_hot=True):
    import os
    if os.name == 'nt':
        text_labels=utils.load_model(path+'meta')['fine_label_names']
        for i in range(5):
            data=utils.load_model(path + 'data_batch_'+str(i+1))
            if i==0:
                train_x=data['data']
                train_y=np.array(data['labels']).reshape(10000,1)
                continue
            train_x= np.vstack((train_x,data['data']))
            train_y = np.vstack((train_y,np.array(data['labels']).reshape(10000,1)))
        data=utils.load_model(path +'test_batch')
        test_x=data['data']
        test_y=np.array(data['labels']).reshape(10000,1)
        if one_hot == True:
            train_y=utils.convert_to_onehot(train_y,10)
            test_y = utils.convert_to_onehot(test_y, 10)
        return {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y, 'text_labels': text_labels}


