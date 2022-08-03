'''
Manuscript Associated: Neural operator learning of heterogeneous mechanobiological insults contributing to aortic aneurysms
Authors: Somdatta Goswami, Postdoctoral Researcher, Brown University
         David S Li, Postdoctoral Researcher, Yale University
Tensorflow Version Required: TF1.15     
This code is for executing the aneurysms with 5x5 sensors.   
'''
import tensorflow as tf
import numpy as np

class FNN:
    def __init__(self):
        pass
    
    def hyper_initial(self, layers):
        L = len(layers)
        W = []
        b = []
        for l in range(1, L):
            in_dim = layers[l-1]
            out_dim = layers[l]
            std = np.sqrt(2./(in_dim + out_dim))
            weight = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], stddev=std))
            bias = tf.Variable(tf.zeros(shape=[1, out_dim]))
            W.append(weight)
            b.append(bias)

        return W, b

    def fnn_T(self, W, b, X, Xmin, Xmax):
        A = 2.0*(X - Xmin)/(Xmax - Xmin) - 1.0
        L = len(W)
        for i in range(L-1):
            A = tf.nn.leaky_relu(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        
        return Y
    
    def fnn_B(self, W, b, X):
        A = X
        L = len(W)
        for i in range(L-1):
            A = tf.nn.leaky_relu(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        
        return Y
    
    def l2_regularizer(self, W):
        
        l2 = 0.0
        L = len(W)
        for i in range(L-1):
            l2 += tf.nn.l2_loss(W[i])
        
        return l2  