import os
#os.environ['CUDA_VISIBLE_DEVICES']='1'
#import tensorflow.compat.v1 as tf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from conv import CNN
from dataset import DataSet
from fnn import FNN
from savedata import SaveData
import sys

np.random.seed(1234)
#tf.set_random_seed(1234)

#output dimension of Branch/Trunk
p = 100
num = 1640
#branch net
layer_B1 = [256, 256, p]
layer_B2 = [256, 256, p]
#trunk net
layer_T = [3, 100, 100, 100, p]



#parameters in CNN
n_channels = 1
#n_out_channels = 16
filter_size_1 = 3
filter_size_2 = 3
filter_size_3 = 3
filter_size_4 = 3
stride = 2

#filter size for each convolutional layer
num_filters_1 = 6
num_filters_2 = 6
num_filters_3 = 6
num_filters_4 = 6

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

h = 21
w = 20

def main(save_index):
    data = DataSet(nx, bs)
    x_train, f1_train, f2_train, u_train, Xmin, Xmax = data.minibatch()

    f1_ph = tf.placeholder(shape=[None, h, w, n_channels], dtype=tf.float32) #[bs, f_dim]
    f2_ph = tf.placeholder(shape=[None, h, w, n_channels], dtype=tf.float32)
    u_ph = tf.placeholder(shape=[None, x_num, 1], dtype=tf.float32) #[bs, x_num, 1]
    x_ph = tf.placeholder(shape=[x_num, 3], dtype=tf.float32)
    
    learning_rate = tf.placeholder(tf.float32, shape=[])
    fnn_model = FNN()
    
    # Branch net
    conv_model = CNN()

    #conv_linear = conv_model.linear_layer(f_ph, n_out_channels)
    conv_11, W11, b11 = conv_model.conv_layer(f1_ph, filter_size_1, num_filters_1, stride, actn=tf.nn.relu)  
    pool_11 = conv_model.max_pool(conv_11, ksize=2, stride=2)  
    conv_21, W21, b21 = conv_model.conv_layer(pool_11, filter_size_2, num_filters_2, stride, actn=tf.nn.relu)
    pool_21 = conv_model.max_pool(conv_21, ksize=2, stride=2) 
    conv_31, W3, b31 = conv_model.conv_layer(pool_21, filter_size_3, num_filters_3, stride, actn=tf.nn.relu)
    pool_31 = conv_model.max_pool(conv_31, ksize=2, stride=2)
    conv_41, W41, b41 = conv_model.conv_layer(pool_31, filter_size_4, num_filters_4, stride, actn=tf.nn.relu)
    pool_41 = conv_model.max_pool(conv_41, ksize=2, stride=2) 
    layer_flat1 = conv_model.flatten_layer(pool_41)

    fnn_layer_11, Wf11, bf11 = conv_model.fnn_layer(layer_flat1, layer_B1[0], actn=tf.nn.tanh, use_actn=True)
    fnn_layer_21, Wf21, bf21 = conv_model.fnn_layer(fnn_layer_11, layer_B1[1], actn=tf.nn.tanh, use_actn=True)
    u_B1, Wf31, bf31 = conv_model.fnn_layer(fnn_layer_21, layer_B1[-1], actn=tf.tanh, use_actn=False) #[bs, p]

    #conv_linear = conv_model.linear_layer(f_ph, n_out_channels)
    conv_12, W12, b12 = conv_model.conv_layer(f2_ph, filter_size_1, num_filters_1, stride, actn=tf.nn.relu)  
    pool_12 = conv_model.max_pool(conv_12, ksize=2, stride=2)  
    conv_22, W22, b22 = conv_model.conv_layer(pool_12, filter_size_2, num_filters_2, stride, actn=tf.nn.relu)
    pool_22 = conv_model.max_pool(conv_22, ksize=2, stride=2) 
    conv_32, W32, b32 = conv_model.conv_layer(pool_22, filter_size_3, num_filters_3, stride, actn=tf.nn.relu)
    pool_32 = conv_model.max_pool(conv_32, ksize=2, stride=2)
    conv_42, W42, b42 = conv_model.conv_layer(pool_32, filter_size_4, num_filters_4, stride, actn=tf.nn.relu)
    pool_42 = conv_model.max_pool(conv_42, ksize=2, stride=2) 
    layer_flat2 = conv_model.flatten_layer(pool_42)

    fnn_layer_12, Wf12, bf12 = conv_model.fnn_layer(layer_flat2, layer_B2[0], actn=tf.nn.tanh, use_actn=True)
    fnn_layer_22, Wf22, bf22 = conv_model.fnn_layer(fnn_layer_12, layer_B2[1], actn=tf.nn.tanh, use_actn=True)
    u_B2, Wf32, bf32 = conv_model.fnn_layer(fnn_layer_22, layer_B2[-1], actn=tf.tanh, use_actn=False) #[bs, p]
    
    u_B = u_B1*u_B2
    #Trunk net
    W_T, b_T = fnn_model.hyper_initial(layer_T)
    u_T = fnn_model.fnn_T(W_T, b_T, x_ph, Xmin, Xmax)
    #inner product    
    u_nn = tf.einsum('ik,mk->imk',u_B,u_T)
    u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True) 

    # loss = tf.reduce_mean(tf.square(u_ph - u_pred))
    loss = tf.reduce_sum(tf.norm(u_pred - u_ph, 2, axis=1)/tf.norm(u_ph, 2, axis=1)) #+ beta*regularizers
    
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
    data_save.save(sess, fnn_model, W_T, b_T, Xmin, Xmax, u_B1, u_B2, f1_ph, f2_ph, u_ph, x_ph, data, num_test, save_results_to)
    
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
    save_index = 5
    main(save_index)
