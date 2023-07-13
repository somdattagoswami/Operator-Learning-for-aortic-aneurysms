import tensorflow.compat.v1 as tf
import numpy as np
import sys
from fnn import FNN
import os
import matplotlib.pyplot as plt
import scipy
plot_folder = './Plot/'    
class SaveData:
    def __init__(self):
        pass

    def save(self, sess, fnn_model, W_T, b_T, Xmin, Xmax, u_B1, u_B2, f1_ph, f2_ph, u_ph, x_ph, data, num_test, save_results_to):
             
        x_test, f1_test, f2_test, u_test = data.testbatch(num_test)
        test_dict = {f1_ph: f1_test, f2_ph: f2_test, u_ph: u_test[:,:,0:1], x_ph: x_test}
        
        u_B = u_B1*u_B2
        u_T = fnn_model.fnn_T(W_T, b_T, x_ph, Xmin, Xmax)
        u_nn = tf.einsum('ik,mk->imk',u_B,u_T)
        u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True) 
    
        u_pred_ = sess.run(u_pred, feed_dict=test_dict)  
        #u_test = data.decoder(u_test)
        #u_pred_ = data.decoder(u_pred_)           
        f1_test = np.reshape(f1_test, (f1_test.shape[0], -1))
        f2_test = np.reshape(f2_test, (f2_test.shape[0], -1))
        u_pred_ = np.reshape(u_pred_[:,:,0:1], (u_test[:,:,0:1].shape[0], u_test[:,:,0:1].shape[1])) 
       
        u_ref = np.reshape(u_test[:,:,0:1], (u_test[:,:,0:1].shape[0], u_test[:,:,0:1].shape[1]))

        
        err_u = np.mean(np.linalg.norm(u_pred_ - u_ref, 2, axis=1)/np.linalg.norm(u_ref, 2, axis=1))

            
        print('Relative L2 Error_Delta: %.3f'%(err_u))
       
        err_u = np.reshape(err_u, (-1, 1))

        
        np.savetxt(save_results_to+'/err_u.txt', err_u, fmt='%e')
        
        scipy.io.savemat(save_results_to+'aortic_pred_DeepONet.mat', 
                     mdict={'x1_test': f1_test,
                            'x2_test': f2_test,
                            'y_test': u_ref, 
                            'u_pred': u_pred_})
        