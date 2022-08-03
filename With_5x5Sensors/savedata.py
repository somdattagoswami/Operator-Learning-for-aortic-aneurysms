'''
Manuscript Associated: Neural operator learning of heterogeneous mechanobiological insults contributing to aortic aneurysms
Authors: Somdatta Goswami, Postdoctoral Researcher, Brown University
         David S Li, Postdoctoral Researcher, Yale University
Tensorflow Version Required: TF1.15     
This code is for executing the aneurysms with 5x5 sensors. 
'''
import tensorflow.compat.v1 as tf
import numpy as np
import sys
from fnn import FNN
import os
import matplotlib.pyplot as plt
import scipy
    
class SaveData:
    def __init__(self):
        pass

    def save(self, sess, fnn_model, W_T, b_T, W_B1, b_B1, W_B2, b_B2, W_B3, b_B3, W_B4, \
             b_B4, Xmin, Xmax, f1_ph, f2_ph, f3_ph, f4_ph, u_ph, x_ph, data, num_test, num_noisy, save_results_to):
        
        x_test, f1_test, f2_test, f3_test, f4_test, u_test = data.testbatch(num_test)
        test_dict = {f1_ph: f1_test, f2_ph: f2_test, f3_ph: f3_test, f4_ph: f4_test, u_ph: u_test[:,:,0:1], x_ph: x_test}
        
        # Branch net
        u_B1 = fnn_model.fnn_B(W_B1, b_B1, f1_ph)
        u_B2 = fnn_model.fnn_B(W_B2, b_B2, f2_ph)
        u_B3 = fnn_model.fnn_B(W_B3, b_B3, f3_ph)
        u_B4 = fnn_model.fnn_B(W_B4, b_B4, f4_ph)
        
        u_B_1 = tf.einsum('ijk,ijk->ijk',u_B1,u_B2)
        u_B_2 = tf.einsum('ijk,ijk->ijk',u_B3,u_B4)
        u_B = tf.einsum('ijk,ijk->ijk',u_B_1,u_B_2)
        
        #Trunk net
        u_T = fnn_model.fnn_T(W_T, b_T, x_ph, Xmin, Xmax)
        
        u_nn = tf.einsum('ijk,mk->imk',u_B,u_T)        
        u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True) 

        u_pred_ = sess.run(u_pred, feed_dict=test_dict)  
        u_test = data.decoder(u_test)
        u_pred_ = data.decoder(u_pred_)           
        f1_test = np.reshape(f1_test, (f1_test.shape[0], -1))
        f2_test = np.reshape(f2_test, (f2_test.shape[0], -1))
        f3_test = np.reshape(f1_test, (f3_test.shape[0], -1))
        f4_test = np.reshape(f2_test, (f4_test.shape[0], -1))        
        u_pred_ = np.reshape(u_pred_[:,:,0:1], (u_test[:,:,0:1].shape[0], u_test[:,:,0:1].shape[1])) 
       
        u_ref = np.reshape(u_test[:,:,0:1], (u_test[:,:,0:1].shape[0], u_test[:,:,0:1].shape[1]))

        
        err_u = np.mean(np.linalg.norm(u_pred_ - u_ref, 2, axis=1)/np.linalg.norm(u_ref, 2, axis=1))

            
        print('Relative L2 Error_Delta: %.3f'%(err_u))
       
        err_u = np.reshape(err_u, (-1, 1))

        
        np.savetxt(save_results_to+'/err_u.txt', err_u, fmt='%e')
        
        scipy.io.savemat(save_results_to+'R4_5x5_pred_DeepONet.mat', 
                     mdict={'x1_test': f1_test,
                            'x2_test': f2_test,
                            'x3_test': f3_test,
                            'x4_test': f4_test,
                            'y_test': u_ref, 
                            'u_pred': u_pred_})
