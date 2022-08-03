'''
Manuscript Associated: Neural operator learning of heterogeneous mechanobiological insults contributing to aortic aneurysms
Authors: Somdatta Goswami, Postdoctoral Researcher, Brown University
         David S Li, Postdoctoral Researcher, Yale University
Tensorflow Version Required: TF1.15     
This code is for executing the aneurysms with full-field images. 
Before running the code: Provide the path for the training and testing dataset in dataset.py Line: 34 and 35 respectively.   
'''
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from dataset import DataSet
from fnn import FNN
from savedata import SaveData
import sys

np.random.seed(1234)

#output dimension of Branch/Trunk
p = 100
num = 1640
#branch net
layer_B1 = [num, 64, 64, 64, 64, p]
layer_B2 = [num, 64, 64, 64, 64, p]
#trunk net
layer_T = [3, 100, 100, 100, p]

#resolution
h = num
#batch_size
bs = 125
bs_test = 45
#size of input for Trunk net
nx = 1640
x_num = nx
epochs = 2000001
lbfgs_iter = 5000
num_train = 550
num_test = 45
num_noisy = 50
beta = 0.009
def main(save_index):
    data = DataSet(nx, bs)
    x_train, f1_train, f2_train, u_train, Xmin, Xmax = data.minibatch()

    f1_ph = tf.placeholder(shape=[None, 1, num], dtype=tf.float32)
    f2_ph = tf.placeholder(shape=[None, 1, num], dtype=tf.float32)
    u_ph = tf.placeholder(shape=[None, x_num, 1], dtype=tf.float32)
    x_ph = tf.placeholder(shape=[x_num, 3], dtype=tf.float32)
    
    learning_rate = tf.placeholder(tf.float32, shape=[])
    fnn_model = FNN()
    
    # Branch net
    W_B1, b_B1 = fnn_model.hyper_initial(layer_B1)
    u_B1 = fnn_model.fnn_B(W_B1, b_B1, f1_ph)  

    W_B2, b_B2 = fnn_model.hyper_initial(layer_B2)
    u_B2 = fnn_model.fnn_B(W_B2, b_B2, f2_ph)

    u_B = tf.einsum('ijk,ijk->ijk',u_B1,u_B2)

    #Trunk net
    W_T, b_T = fnn_model.hyper_initial(layer_T)
    u_T = fnn_model.fnn_T(W_T, b_T, x_ph, Xmin, Xmax)

    #inner product    
    u_nn = tf.einsum('ijk,mk->imk',u_B,u_T)
    u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True) 

    regularizers = fnn_model.l2_regularizer(W_B1) + fnn_model.l2_regularizer(W_B2) 
    loss = tf.reduce_sum(tf.norm(u_pred - u_ph, 2, axis=1)/tf.norm(u_ph, 2, axis=1)) + beta*regularizers
    
    lbfgs_buffer = []
    train_adam = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.99).minimize(loss)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))  
    sess.run(tf.global_variables_initializer())
    
    def callback(lbfgs_buffer,loss):
        lbfgs_buffer = np.append(lbfgs_buffer, loss)
        print(loss)
        
    n = 0
    nt = 0
    start_time = time.perf_counter()
    time_step_0 = time.perf_counter()
    
    train_loss = np.zeros((epochs+1, 1))
    test_loss = np.zeros((int(epochs/1000)+1, 1))    
    while n <= epochs:
        
        if (epochs <= 300000): 
            lr = 0.001
        else:
            lr = 0.0005            
        x_train, f1_train, f2_train, u_train, _, _ = data.minibatch() 
        train_dict={f1_ph: f1_train, f2_ph: f2_train, u_ph: u_train[:,:,0:1], x_ph: x_train, learning_rate: lr}
        loss_,_ = sess.run([loss, train_adam], feed_dict=train_dict)
        loss_=sess.run(loss, feed_dict=train_dict)    
        train_loss[n,0] = loss_
        
        if n%1000 == 0:
            x_test, f1_test, f2_test, u_test = data.testbatch(bs_test)
            u_test_ = sess.run(u_pred, feed_dict={f1_ph: f1_test, f2_ph: f2_test, x_ph: x_test})
            u_test = data.decoder(u_test)
            u_test_ = data.decoder(u_test_)  
            err_u = np.mean(np.linalg.norm(u_test_ - u_test, 2, axis=1)/np.linalg.norm(u_test[:,:,0:1], 2, axis=1))
            
            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            print('Step: %d, Loss: %.3e, Test L2 error: %.3f, Time (secs): %.3f'%(n, loss_, err_u, T))
            time_step_0 = time.perf_counter()
            test_loss[nt,0] = err_u
            nt += 1

        n += 1
        
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                      method='L-BFGS-B',
                                                      options={'maxiter': lbfgs_iter,
                                                                'maxfun': lbfgs_iter,
                                                                'maxcor': 100,
                                                                  'maxls': 50,
                                                                  'ftol': 1.0 * np.finfo(float).eps})
    x_train, f1_train, f2_train, u_train, _, _ = data.alldata()
    train_dict={f1_ph: f1_train, f2_ph: f2_train, u_ph: u_train[:,:,0:1], x_ph: x_train, learning_rate: lr}
    optimizer.minimize(sess, feed_dict=train_dict, fetches=[lbfgs_buffer, loss],loss_callback= callback)     
    
    current_directory = os.getcwd() 
    case = "Case_"
    folder_index = str(save_index)
    
    results_dir = "/" + case + folder_index +"/"
    save_results_to = current_directory + results_dir
    if not os.path.exists(save_results_to):
        os.makedirs(save_results_to)
    
    save_models_to = save_results_to +"model/"
    if not os.path.exists(save_models_to):
        os.makedirs(save_models_to)      
    
    saver.save(sess, save_models_to+'Model')

    stop_time = time.perf_counter()
    print('Elapsed time (secs): %.3f'%(stop_time - start_time))
        
    np.savetxt(save_results_to+'/train_loss.txt', train_loss)
    np.savetxt(save_results_to+'/test_loss.txt', test_loss)

    data_save = SaveData()
    data_save.save(sess, fnn_model, W_T, b_T, W_B1, b_B1, W_B2, b_B2, Xmin, Xmax, \
    f1_ph, f2_ph, u_ph, x_ph, data, num_test, num_noisy, save_results_to)
    
    ## Plotting the loss history
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, train_loss[:,0], color='blue', label='Training Loss')
    # ax.plot(x, test_loss[:,0], color='red', label='Testing Loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig(save_results_to+'loss_his.png')

    ## Plotting the test error
    num_epoch = test_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, test_loss[:,0], color='blue', label='Testing Loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig(save_results_to+'loss_test.png')    

if __name__ == "__main__":
    save_index = 1
    main(save_index)
