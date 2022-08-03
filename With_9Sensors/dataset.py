'''
Manuscript Associated: Neural operator learning of heterogeneous mechanobiological insults contributing to aortic aneurysms
Authors: Somdatta Goswami, Postdoctoral Researcher, Brown University
         David S Li, Postdoctoral Researcher, Yale University
Tensorflow Version Required: TF1.15     
This code is for executing the aneurysms with 9 sensors.   
'''
import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as io
import sys
np.random.seed(1234)

class DataSet:
    def __init__(self, num, bs):
        self.num = num
        self.bs = bs
        self.F1_train, self.F2_train, self.F3_train, self.F4_train, self.U_train, self.F1_test, self.F2_test, self.F3_test, self.F4_test, self.U_test, \
        self.X, self.U_minmax = self.load_data()

    def decoder(self, x):

        x = (x + 1)*(self.U_minmax[0,1] - self.U_minmax[0,0])/2 + self.U_minmax[0,0]
        
        return x

    def load_data(self):
        
        num_train = 550
        num_test = 45
        num_noisy = 50
        
        data_train = np.load('Load Training Data and Testing')# Training Data
        data_test = np.load('Load Out_of_Distribution Data') # Test Data
        
        f1_train = data_train['F1_train']
        f2_train = data_train['F1_loc_train']
        f3_train= data_train['F2_train']
        f4_train = data_train['F2_loc_train']
        u_train = data_train['U_train']
        
        f1_test = data_test['F1_test']
        f2_test = data_test['F1_loc_test']
        f3_test = data_test['F2_test']
        f4_test = data_test['F2_loc_test']
        
        u_test = data_test['U_test']
      
        X = data_train['Pts'][:,1:3]
        X_mod = np.zeros((X.shape[0],3))
        X_mod[:,0:1] = np.cos(X[:,0:1])
        X_mod[:,1:2] = np.sin(X[:,0:1])
        X_mod[:,2:3] = X[:,1:2]
        X = X_mod
        
        s = 9
        pt = 1
        res = 1640

        f1_train_mean = np.mean(np.reshape(f1_train, (-1, s)), 0)
        f1_train_std = np.std(np.reshape(f1_train, (-1, s)), 0)
        f1_train_mean = np.reshape(f1_train_mean, (-1, 1, s))
        F1_train = np.reshape(f1_train, (-1, 1, s))
        F1_train = (F1_train - f1_train_mean)/(f1_train_std + 1.0e-9) + 1.0
        
        f2_train_mean = np.mean(np.reshape(f2_train, (-1, pt)), 0)
        f2_train_std = np.std(np.reshape(f2_train, (-1, pt)), 0)
        f2_train_mean = np.reshape(f2_train_mean, (-1, 1, pt))
        F2_train = np.reshape(f2_train, (-1, 1, pt))
        F2_train = (F2_train - f2_train_mean)/(f2_train_std + 1.0e-9) 

        f3_train_mean = np.mean(np.reshape(f3_train, (-1, s)), 0)
        f3_train_std = np.std(np.reshape(f3_train, (-1, s)), 0)
        f3_train_mean = np.reshape(f3_train_mean, (-1, 1, s))
        F3_train = np.reshape(f3_train, (-1, 1, s))
        F3_train = (F3_train - f3_train_mean)/(f3_train_std + 1.0e-9) + 4.0
        
        f4_train_mean = np.mean(np.reshape(f4_train, (-1, pt)), 0)
        f4_train_std = np.std(np.reshape(f4_train, (-1, pt)), 0)
        f4_train_mean = np.reshape(f4_train_mean, (-1, 1, pt))
        F4_train = np.reshape(f4_train, (-1, 1, pt))
        F4_train = (F4_train - f4_train_mean)/(f4_train_std + 1.0e-9) 
        
        F1_test = np.reshape(f1_test, (-1, 1, s))
        F1_test = (F1_test - f1_train_mean)/(f1_train_std + 1.0e-9) + 1.0
        F2_test = np.reshape(f2_test, (-1, 1, pt))
        F2_test = (F2_test - f2_train_mean)/(f2_train_std + 1.0e-9)
        F3_test = np.reshape(f3_test, (-1, 1, s))
        F3_test = (F3_test - f3_train_mean)/(f3_train_std + 1.0e-9) + 4.0
        F4_test = np.reshape(f4_test, (-1, 1, pt))
        F4_test = (F4_test - f4_train_mean)/(f4_train_std + 1.0e-9)
        
        u_minmax = np.array([[np.min(u_train), np.max(u_train)]])

        u_train = np.reshape(u_train, (-1, res, 1))
        u_test = np.reshape(u_test, (-1, res, 1))
        
        u_train = 2.0*(u_train - u_minmax[0,0])/(u_minmax[0,1] - u_minmax[0,0]) - 1.0
        u_test = 2.0*(u_test - u_minmax[0,0])/(u_minmax[0,1] - u_minmax[0,0]) - 1.0
        
        U_train = u_train
        U_test = u_test   
            
        return F1_train, F2_train, F3_train, F4_train, U_train, F1_test, F2_test, F3_test, F4_test, U_test, X, u_minmax

        
    def minibatch(self):

        batch_id = np.random.choice(self.F1_train.shape[0], self.bs, replace=False)

        f1_train = [self.F1_train[i:i+1] for i in batch_id]
        f1_train = np.concatenate(f1_train, axis=0)
        f2_train = [self.F2_train[i:i+1] for i in batch_id]
        f2_train = np.concatenate(f2_train, axis=0)
        f3_train = [self.F3_train[i:i+1] for i in batch_id]
        f3_train = np.concatenate(f3_train, axis=0)
        f4_train = [self.F4_train[i:i+1] for i in batch_id]
        f4_train = np.concatenate(f4_train, axis=0)


        u_train = [self.U_train[i:i+1] for i in batch_id]
        u_train = np.concatenate(u_train, axis=0)

        x_train = self.X

        Xmin = np.array([-1., -1.,  0.]).reshape((-1, 3))
        Xmax = np.array([1., 1.,  15.]).reshape((-1, 3))

        return x_train, f1_train, f2_train, f3_train, f4_train, u_train, Xmin, Xmax
    
    def alldata(self):

        f1_train = self.F1_train
        f2_train = self.F2_train
        f3_train = self.F3_train
        f4_train = self.F4_train        
        u_train = self.U_train

        x_train = self.X

        Xmin = np.array([-1., -1.,  0.]).reshape((-1, 3))
        Xmax = np.array([1., 1.,  15.]).reshape((-1, 3))

        return x_train, f1_train, f2_train, f3_train, f4_train, u_train, Xmin, Xmax    

    def testbatch(self, num_test):

        batch_id = np.arange(num_test)
        f1_test = [self.F1_test[i:i+1] for i in batch_id]
        f1_test = np.concatenate(f1_test, axis=0)
        f2_test = [self.F2_test[i:i+1] for i in batch_id]
        f2_test = np.concatenate(f2_test, axis=0)
        f3_test = [self.F3_test[i:i+1] for i in batch_id]
        f3_test = np.concatenate(f3_test, axis=0)
        f4_test = [self.F4_test[i:i+1] for i in batch_id]
        f4_test = np.concatenate(f4_test, axis=0)
        u_test = [self.U_test[i:i+1] for i in batch_id]
        u_test = np.concatenate(u_test, axis=0)

        x_test = self.X

        return x_test, f1_test, f2_test, f3_test, f4_test, u_test

    def noisybatch(self, num_noisy):

        batch_id = np.arange(num_noisy)
        f1_noisy = [self.F1_noisy[i:i+1] for i in batch_id]
        f1_noisy = np.concatenate(f1_noisy, axis=0)
        f2_noisy = [self.F2_noisy[i:i+1] for i in batch_id]
        f2_noisy = np.concatenate(f2_noisy, axis=0)
        f3_noisy = [self.F3_noisy[i:i+1] for i in batch_id]
        f3_noisy = np.concatenate(f3_noisy, axis=0)
        f4_noisy = [self.F4_noisy[i:i+1] for i in batch_id]
        f4_noisy = np.concatenate(f4_noisy, axis=0)
        u_noisy = [self.U_noisy[i:i+1] for i in batch_id]
        u_noisy = np.concatenate(u_noisy, axis=0)

        x_noisy = self.X

        return x_noisy, f1_noisy, f2_noisy, f3_noisy, f4_noisy, u_noisy