'''
Manuscript Associated: Neural operator learning of heterogeneous mechanobiological insults contributing to aortic aneurysms
Authors: Somdatta Goswami, Postdoctoral Researcher, Brown University
         David S Li, Postdoctoral Researcher, Yale University
Tensorflow Version Required: TF1.15     
This code is for executing the aneurysms with full-field images.    
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
        self.F1_train, self.F2_train, self.U_train, self.F1_test, self.F2_test, self.U_test, \
        self.X, self.U_minmax, self.F1_noisy, self.F2_noisy, self.U_noisy = self.load_data()

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
        f2_train = data_train['F2_train'] 
        u_train = data_train['U_train']
        f1_test = data_test['F1_test']
        f2_test = data_test['F2_test'] 
        u_test = data_test['U_test']
        
        # Noisy data
        nx = f1_train.shape[1]
        ny = f1_train.shape[2]
        f1_noisy = np.zeros((num_noisy, nx, ny))
        f2_noisy = np.zeros((num_noisy, nx, ny))
        for k in range(num_noisy):
            for i in range(nx):
                for j in range(ny):
                    f1_noisy[k, i, j] = f1_train[k, i, j] + np.random.normal(0,0.05*f1_train[k, i, j])
                    f2_noisy[k, i, j] = f2_train[k, i, j] + np.random.normal(0,0.05*f2_train[k, i, j])
        
        u_noisy = u_train[:num_noisy, :]   
         
        r = 1
        f1_train = f1_train[:,::r,::r]
        f2_train = f2_train[:,::r,::r]
        f1_test = f1_test[:,::r,::r]
        f2_test = f2_test[:,::r,::r]
      
        X = data_train['Pts'][:,1:3]
        
        X_mod = np.zeros((X.shape[0],3))
        X_mod[:,0:1] = np.cos(X[:,0:1])
        X_mod[:,1:2] = np.sin(X[:,0:1])
        X_mod[:,2:3] = X[:,1:2]
        X = X_mod
        
        s = f1_train.shape[1]*f1_train.shape[2]  
        res = 1640

        f1_train_mean = np.mean(np.reshape(f1_train, (-1, s)), 0)
        f1_train_std = np.std(np.reshape(f1_train, (-1, s)), 0)
        f1_train_mean = np.reshape(f1_train_mean, (-1, 1, s))
        F1_train = np.reshape(f1_train, (-1, 1, s))
        F1_train = (F1_train - f1_train_mean)/(f1_train_std + 1.0e-9) + 3.0

        f2_train_mean = np.mean(np.reshape(f2_train, (-1, s)), 0)
        f2_train_std = np.std(np.reshape(f2_train, (-1, s)), 0)
        f2_train_mean = np.reshape(f2_train_mean, (-1, 1, s))
        F2_train = np.reshape(f2_train, (-1, 1, s))
        F2_train = (F2_train - f2_train_mean)/(f2_train_std + 1.0e-9) + 3.0
        
        F1_test = np.reshape(f1_test, (-1, 1, s))
        F1_test = (F1_test - f1_train_mean)/(f1_train_std + 1.0e-9) + 3.0
        F2_test = np.reshape(f2_test, (-1, 1, s))
        F2_test = (F2_test - f2_train_mean)/(f2_train_std + 1.0e-9) + 3.0

        F1_noisy = np.reshape(f1_noisy, (-1, 1, s))
        F1_noisy = (F1_noisy - f1_train_mean)/(f1_train_std + 1.0e-9) + 3.0
        F2_noisy = np.reshape(f2_noisy, (-1, 1, s))
        F2_noisy = (F2_noisy - f2_train_mean)/(f2_train_std + 1.0e-9) + 3.0
        
        u_minmax = np.array([[np.min(u_train), np.max(u_train)]])

        u_train = np.reshape(u_train, (-1, res, 1))
        u_test = np.reshape(u_test, (-1, res, 1))
        u_noisy = np.reshape(u_noisy, (-1, res, 1))
        
        u_train = 2.0*(u_train - u_minmax[0,0])/(u_minmax[0,1] - u_minmax[0,0]) - 1.0
        u_test = 2.0*(u_test - u_minmax[0,0])/(u_minmax[0,1] - u_minmax[0,0]) - 1.0
        u_noisy = 2.0*(u_noisy - u_minmax[0,0])/(u_minmax[0,1] - u_minmax[0,0]) - 1.0
        
        U_train = u_train
        U_test = u_test 
        U_noisy = u_noisy               
        '''
        U_ref = np.reshape(U_test, (U_test.shape[0], U_test.shape[1]))
        np.savetxt('./Output/u_ref', U_ref, fmt='%e')
        '''

        return F1_train, F2_train, U_train, F1_test, F2_test, U_test, X, u_minmax, F1_noisy, F2_noisy, U_noisy

        
    def minibatch(self):

        batch_id = np.random.choice(self.F1_train.shape[0], self.bs, replace=False)
        
        f1_train = [self.F1_train[i:i+1] for i in batch_id]
        f1_train = np.concatenate(f1_train, axis=0)
        f2_train = [self.F2_train[i:i+1] for i in batch_id]
        f2_train = np.concatenate(f2_train, axis=0)
        u_train = [self.U_train[i:i+1] for i in batch_id]
        u_train = np.concatenate(u_train, axis=0)

        '''
        x = np.linspace(0., 1, self.num)
        y = np.linspace(0., 1, self.num)
        xx, yy = np.meshgrid(x, y)
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        x_train = np.hstack((xx, yy))
        '''
        x_train = self.X

        Xmin = np.array([-1., -1.,  0.]).reshape((-1, 3))
        Xmax = np.array([1., 1.,  15.]).reshape((-1, 3))
        #x_train = np.linspace(-1, 1, self.N).reshape((-1, 1))

        return x_train, f1_train, f2_train, u_train, Xmin, Xmax
    
    def alldata(self):

        f1_train = self.F1_train
        f2_train = self.F2_train
        u_train = self.U_train

        '''
        x = np.linspace(0., 1, self.num)
        y = np.linspace(0., 1, self.num)
        xx, yy = np.meshgrid(x, y)
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        x_train = np.hstack((xx, yy))
        '''
        x_train = self.X

        Xmin = np.array([-1., -1.,  0.]).reshape((-1, 3))
        Xmax = np.array([1., 1.,  15.]).reshape((-1, 3))
        #x_train = np.linspace(-1, 1, self.N).reshape((-1, 1))

        return x_train, f1_train, f2_train, u_train, Xmin, Xmax    

    def testbatch(self, num_test):
#        batch_id = np.random.choice(self.F_test.shape[0], num_test, replace=False)
        batch_id = np.arange(num_test)
        f1_test = [self.F1_test[i:i+1] for i in batch_id]
        f1_test = np.concatenate(f1_test, axis=0)
        f2_test = [self.F2_test[i:i+1] for i in batch_id]
        f2_test = np.concatenate(f2_test, axis=0)
        u_test = [self.U_test[i:i+1] for i in batch_id]
        u_test = np.concatenate(u_test, axis=0)

        '''
        U_ref = np.reshape(u_test, (u_test.shape[0], u_test.shape[1]))
        np.savetxt('./Output/u_ref', U_ref, fmt='%e')
        '''

        '''
        x = np.linspace( 0., 1., self.num)
        y = np.linspace( 0., 1., self.num)
        xx, yy = np.meshgrid(x, y)
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        x_test = np.hstack((xx, yy))
        '''
        x_test = self.X
#
#        batch_id = np.reshape(batch_id, (-1, 1))

        return x_test, f1_test, f2_test, u_test
    
    def noisybatch(self, num_noisy):
#        batch_id = np.random.choice(self.F_test.shape[0], num_test, replace=False)
        batch_id = np.arange(num_noisy)
        f1_noisy = [self.F1_noisy[i:i+1] for i in batch_id]
        f1_noisy = np.concatenate(f1_noisy, axis=0)
        f2_noisy = [self.F2_noisy[i:i+1] for i in batch_id]
        f2_noisy = np.concatenate(f2_noisy, axis=0)
        u_noisy = [self.U_noisy[i:i+1] for i in batch_id]
        u_noisy = np.concatenate(u_noisy, axis=0)

        '''
        U_ref = np.reshape(u_test, (u_test.shape[0], u_test.shape[1]))
        np.savetxt('./Output/u_ref', U_ref, fmt='%e')
        '''

        '''
        x = np.linspace( 0., 1., self.num)
        y = np.linspace( 0., 1., self.num)
        xx, yy = np.meshgrid(x, y)
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        x_test = np.hstack((xx, yy))
        '''
        x_noisy = self.X
#
#        batch_id = np.reshape(batch_id, (-1, 1))

        return x_noisy, f1_noisy, f2_noisy, u_noisy
