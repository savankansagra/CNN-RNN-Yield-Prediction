##############################
# Implementation of 10.3389/fpls.2021.709008 paper
# Many things to improve in the implementation
# Ref - https://github.com/saeedkhaki92/CNN-RNN-Yield-Prediction
###################################

import numpy as np
import time
import tensorflow as tf
import pandas as pd

# importing for trace stack
import traceback

# include fot logging
import pandas as pd
log_df = pd.DataFrame(columns={'Iteration', 'training_RMSE', 'Cor_train','test_RMSE', 'Cor_is'})


def conv_res_part_E(E_t, f, is_training, var_name):
    epsilon=0.0001
    f0=5
    s0 = 1
    # First Conv Layer
    print("-------Conv Layer 1 ------- Start E_t shape :", E_t.shape) #[None, 52, 1]
    X = tf.layers.conv1d(E_t, filters=4, kernel_size=7, strides=1, padding='valid',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                          name='Conv01' + var_name, data_format="channels_last", reuse=tf.AUTO_REUSE)
    X = tf.nn.relu(X)
    print("Conv01 out E", X.shape) # [None, 52-7+1=46, 4] => [None, 46, 4]
    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool')
    print("conv01 out E, with average pooling", X.shape) # [None, 23, 4]

    # Second Conv Layer
    print("-------Conv Layer 2 --------- Start E_t shape:", X.shape) # [None, 23, 4]
    X = tf.layers.conv1d(X, filters=4, kernel_size=4, strides=1, padding='valid', 
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name="Conv02"+var_name, data_format="channels_last", reuse=tf.AUTO_REUSE)
    X = tf.nn.relu(X)
    print("Conv02 out E,", X.shape) # [None, 23-4+1=20, 4] => [None, 20, 4]
    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format="channels_last", name="average_pool")
    print("Conv02 out E, with average pooling", X.shape) # [none, 10, 4]

    # Third Conv Layer
    # TODO: The kernel size taken by us to match with paper
    print("-------Conv Layer 3 --------- Start E_t shape:", X.shape) # [none, 10, 4]
    X = tf.layers.conv1d(X, filters=4, kernel_size=9, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name="Conv03"+var_name, data_format="channels_last", reuse=tf.AUTO_REUSE)
    print("Conv03 out E,", X.shape) # [None, 2, 4]
    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format="channels_last", name="average_pool")
    print("Conv03 out E, with average pooling", X.shape) # [none, 1, 4]

    return X
        

def conv_res_part_S(S_t, f, is_training, var_name):
    # First Conv Layer
    print("---------- Conv Layer 1 soil -------- start S_t.shpae", S_t.shape) # [None, 6, 1]
    X = tf.layers.conv1d(S_t, filters=4, kernel_size=3, strides=1, padding='valid', 
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name="Conv01"+var_name, data_format='channels_last', reuse=tf.AUTO_REUSE)
    X = tf.nn.relu(X)
    print("Conv01 out S shape",X.shape) # [None, 4, 4]
    # X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format="channels_last", name="average_pool")
    # print("Conv01 out S, with average pooling, X shape :", X.shape) # [None, 2, 4]

    # Second Conv Layer
    print("--------- Conv Layer 2 soil --------- start S_t.shape", S_t.shape) # [None, 4, 4]
    X = tf.layers.conv1d(X, filters=4, kernel_size=3, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name="Conv02"+var_name, data_format="channels_last", reuse=tf.AUTO_REUSE)
    X = tf.nn.relu(X)
    print("Conv02 out S shape", X.shape) # [None, 2, 4]
    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format="channels_last", name="average_pool")
    print("Conv02 out S, with average pooling, X shape :", X.shape) # [None, 1, 4]


    # Third Conv Layer
    # print("--------- Conv Layer 3 soil --------- start S_t.shape", S_t.shape) # [None, 2, 4]
    # X = tf.layers.conv1d(X, filters=4, kernel_size=2, strides=1, padding="valid",
    #                      kernel_constraint=tf.contrib.layers.xavier_initializer(), activation=None,
    #                      name="Conv03"+var_name, data_format="channels_last", reuse=tf.AUTO_REUSE)
    # X = tf.nn.relu(X)
    # print("Conv03 out S shape", X.shape) # [None, 1, 4]
    
    return X



def Cost_function(Y, Yhat):

    E = Y - Yhat
    E2 = tf.pow(E, 2)

    MSE = tf.squeeze(tf.reduce_mean(E2))
    RMSE = tf.pow(MSE, 0.5)
    Loss = tf.losses.huber_loss(Y, Yhat, weights=1.0, delta=5.0)

    return RMSE, MSE, E, Loss





def get_sample(dic,L,avg,batch_size,time_steps,num_features):


    L_tr=L[:-1,:]



    out=np.zeros(shape=[batch_size,time_steps,num_features])

    for i in range(batch_size):

        r1 = np.squeeze(np.random.randint(L_tr.shape[0], size=1))

        years = L_tr[r1, :]

        for j, y in enumerate(years):
            X = dic[str(y)]
            ym=avg[str(y)]
            r2 = np.random.randint(X.shape[0], size=1)
            #n=X[r2, :]
            out[i, j, :] = np.concatenate((X[r2, :],np.array([[ym]])),axis=1)


    return out



def get_sample_te(dic,mean_last,avg,batch_size_te,time_steps,num_features):

    out = np.zeros(shape=[batch_size_te, time_steps, num_features]) # num_features should be 398

    X = dic[str(2018)]

  #  r1 = np.random.randint(X.shape[0], size=batch_size_te)
    # out[:, 0:4, :] += mean_last.reshape(1,4,3+6*52+1+100+16+4)
    out[:, 0:4, :] += mean_last.reshape(1,4,6*52+66+14+4)  # 398
    #n=X[r1, :]
    #print(n.shape)
    ym=np.zeros(shape=[batch_size_te,1])+avg['2018']

    out[:,4,:]=np.concatenate((X,ym),axis=1)

    return out





def main_process(E_t1,E_t2,E_t3,E_t4,E_t5,E_t6,S_t1,S_t2,S_t3,S_t4,S_t5,S_t6,S_t7,S_t8,S_t9,S_t10,P_t,Ybar,S_t_extra,f,is_training,num_units,num_layers,dropout):
    # W-CNN
    e_out1 = conv_res_part_E(E_t1, f, is_training=is_training, var_name='v1w1') # [None, 4, 1]
    e_out1 = tf.contrib.layers.flatten(e_out1)  #[None, 4]
    e_out2 = conv_res_part_E(E_t2, f, is_training=is_training, var_name='v1w2')
    e_out2 = tf.contrib.layers.flatten(e_out2)
    e_out3 = conv_res_part_E(E_t3, f, is_training=is_training, var_name='v1w3')
    e_out3 = tf.contrib.layers.flatten(e_out3)
    e_out4 = conv_res_part_E(E_t4, f, is_training=is_training, var_name='v1w4')
    e_out4 = tf.contrib.layers.flatten(e_out4)
    e_out5 = conv_res_part_E(E_t5, f, is_training=is_training, var_name='v1w5')
    e_out5 = tf.contrib.layers.flatten(e_out5)
    e_out6 = conv_res_part_E(E_t6, f, is_training=is_training, var_name='v1w6')
    e_out6 = tf.contrib.layers.flatten(e_out6)
    e_out = tf.concat([e_out1, e_out2, e_out3, e_out4, e_out5, e_out6], axis=1) # [None, 24]
    print('weather after concatation', e_out)
    
    # Fully connected layer on weather CNN
    e_out = tf.contrib.layers.fully_connected(inputs=e_out, num_outputs=60, activation_fn=tf.nn.relu,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.zeros_initializer())

    #S_CNN
    s_out1 = conv_res_part_S(S_t1, f, is_training=is_training, var_name='v1s1')
    s_out1 = tf.contrib.layers.flatten(s_out1)
    s_out2 = conv_res_part_S(S_t2, f, is_training=is_training, var_name='v1s2')
    s_out2 = tf.contrib.layers.flatten(s_out2)
    s_out3 = conv_res_part_S(S_t3, f, is_training=is_training, var_name='v1s3')
    s_out3 = tf.contrib.layers.flatten(s_out3)
    s_out4 = conv_res_part_S(S_t4, f, is_training=is_training, var_name='v1s4')
    s_out4 = tf.contrib.layers.flatten(s_out4)
    s_out5 = conv_res_part_S(S_t5, f, is_training=is_training, var_name='v1s5')
    s_out5 = tf.contrib.layers.flatten(s_out5)
    s_out6 = conv_res_part_S(S_t6, f, is_training=is_training, var_name='v1s6')
    s_out6 = tf.contrib.layers.flatten(s_out6)
    s_out7 = conv_res_part_S(S_t7, f, is_training=is_training, var_name='v1s7')
    s_out7 = tf.contrib.layers.flatten(s_out7)
    s_out8 = conv_res_part_S(S_t8, f, is_training=is_training, var_name='v1s8')
    s_out8 = tf.contrib.layers.flatten(s_out8)
    s_out9 = conv_res_part_S(S_t9, f, is_training=is_training, var_name='v1s9')
    s_out9 = tf.contrib.layers.flatten(s_out9)
    s_out10 = conv_res_part_S(S_t10, f, is_training=is_training, var_name='v1s10')
    s_out10 = tf.contrib.layers.flatten(s_out10)
    
    s_out = tf.concat([s_out1, s_out2, s_out3, s_out4, s_out5, s_out6, s_out7, s_out8, s_out9, s_out10], axis=1)
    print("soil after concatenate ", s_out)

    s_out = tf.contrib.layers.fully_connected(inputs=s_out, num_outputs=40, activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.zeros_initializer())
    s_out = tf.nn.relu(s_out)

    ##FC1 - fully Connected layer 1 with 64, 32 and 16 neurons
    P_t_flat=tf.contrib.layers.flatten(P_t)
    FC_1_1 = tf.contrib.layers.fully_connected(inputs=P_t_flat, num_outputs=64, activation_fn=None,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())
    FC_1_2 = tf.contrib.layers.fully_connected(inputs=FC_1_1, num_outputs=32, activation_fn=None,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())
    FC_1_3 = tf.contrib.layers.fully_connected(inputs=FC_1_2, num_outputs=16, activation_fn=None,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())
    print("The shape of FC1", FC_1_3.shape)
    
    ## Combine all the data and reshape
    combined_data_layer_1 = tf.concat([e_out, s_out, FC_1_3], axis=1)
    print("combined data layer 1 shape: ", combined_data_layer_1.shape)
    time_step=5
    combined_data_layer_1_reshaped = tf.reshape(combined_data_layer_1, shape=[-1, time_step, combined_data_layer_1.get_shape().as_list()[-1]])
    print("combined data layer 1 after reshape: ", combined_data_layer_1_reshaped.shape)
    # Combine result with Ybar
    combined_data_layer_1_with_Ybar = tf.concat([combined_data_layer_1_reshaped, Ybar], axis=-1)


    ## FC2 - fully connected layer 2 with 128 and 64 nuerons.    
    FC_2_1 = tf.contrib.layers.fully_connected(inputs=combined_data_layer_1_with_Ybar, num_outputs=128, activation_fn=None,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())
    FC_2_2 = tf.contrib.layers.fully_connected(inputs=FC_2_1, num_outputs=64, activation_fn=None,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())

    print("The shape of FC2", FC_2_2.shape)

    # Reshape the output back
    output = tf.reshape(FC_2_2, shape=[-1, FC_2_2.get_shape().as_list()[-1]])

    # Map the output to 1 unit
    output = tf.contrib.layers.fully_connected(inputs=output, num_outputs=1, activation_fn=None,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())
    print(output)
    print("output shape after fully connected to 1 unit:", output.shape)

    # output for all time steps
    output = tf.reshape(output, shape=[-1, 5])
    print("output of all time steps", output)
    
    Yhat1 = tf.gather(output, indices=[4], axis=1)
    print("Yhat1 :", Yhat1)

    Yhat2 = tf.gather(output, indices=[0, 1, 2, 3], axis=1)
    print("Yhat2 :", Yhat2)

    return Yhat1, Yhat2
    


def main_program(X, Index, num_units, num_layers, Max_it, learning_rate, batch_size_tr, le, l):
    global log_df
    with tf.device('/cpu:0'):
        # Weather placeholders
        E_t1 = tf.placeholder(shape=[None, 52, 1], dtype=tf.float32, name='E_t1')
        E_t2 = tf.placeholder(shape=[None, 52, 1], dtype=tf.float32, name='E_t2')
        E_t3 = tf.placeholder(shape=[None, 52, 1], dtype=tf.float32, name='E_t3')
        E_t4 = tf.placeholder(shape=[None, 52, 1], dtype=tf.float32, name='E_t4')
        E_t5 = tf.placeholder(shape=[None, 52, 1], dtype=tf.float32, name='E_t5')
        E_t6 = tf.placeholder(shape=[None, 52, 1], dtype=tf.float32, name='E_t6')

        # Soil Placeholders
        S_t1 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t1')
        S_t2 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t2')
        S_t3 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t3')
        S_t4 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t4')
        S_t5 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t5')
        S_t6 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t6')
        S_t7 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t7')
        S_t8 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t8')
        S_t9 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t9')
        S_t10 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t10')
        S_t_extra = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t_extra')

        # Place Placeholders
        P_t = tf.placeholder(shape=[None, 14, 1], dtype=tf.float32, name='P_t') # Plant Data

        Ybar = tf.placeholder(shape=[None, 5, 1], dtype=tf.float32, name='Ybar')

        Y_t = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Y_t")

        Y_t_2 = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='Y_t_2')

        is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        lr = tf.placeholder(shape=[], dtype=tf.float32, name="learning_rate")
        dropout = tf.placeholder(tf.float32, name='dropout')
        
        f=3
        Yhat1, Yhat2 = main_process(E_t1, E_t2, E_t3, E_t4, E_t5, E_t6, 
                                    S_t1, S_t2, S_t3, S_t4, S_t5, S_t6, S_t7, S_t8, S_t9, S_t10,
                                    P_t, Ybar, S_t_extra, f, is_training, num_units, num_layers, dropout)
        
        Yhat1 = tf.identity(Yhat1, name='Yhat1')
        # Yhat2 is the prediction we got before the final time step (year t)
        print("Yhatttttttttttt", Yhat1)

        # Total parameters
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Domension
            print(variable)
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('total_parameters :', total_parameters)

        with tf.name_scope("loss_function"):
            RMSE,_,_,Loss1=Cost_function(Y_t, Yhat1)
            _, _, _, Loss2 = Cost_function(Y_t_2, Yhat2)
            Tloss=tf.constant(l,dtype=tf.float32)*Loss1+tf.constant(le,dtype=tf.float32)*Loss2        

        RMSE = tf.identity(RMSE, name='RMSE')

        with tf.name_scope('train'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(Tloss)
    
        
        init = tf.global_variables_initializer()

        sess = tf.Session()

        sess.run(init)
        writer = tf.summary.FileWriter("./tensorboard")
        writer.add_graph(sess.graph)

        t1=time.time()

        A = []

        for i in range(4, 39):
            A.append([i-4, i-3, i-3, i-1, i])
        
        A = np.vstack(A)
        A += 1980
        print(A.shape)

        dic = {}
        for i in range(39):
            dic[str(i+1980)] = X[X[:, 1] == i+1980]

        avg = {}
        avg2 = []
        for i in range(39):
            avg[str(i + 1980)] = np.mean(X[X[:, 1] == i + 1980][:, 2])
            avg2.append(np.mean(X[X[:, 1] == i + 1980][:, 2]))

        print('avgggggg', avg)

        mm = np.mean(avg2)
        ss = np.std(avg2)
        

        avg = {}

        for i in range(39):
            avg[str(i + 1980)] = (np.mean(X[X[:, 1] == i + 1980][:, 2]) - mm) / ss

        avg['2018'] = avg['2017']
        #avg['2017']=avg['2016']

        a8 = np.concatenate((np.mean(dic['2014'], axis=0), [avg['2014']]))

        a9 = np.concatenate((np.mean(dic['2015'], axis=0), [avg['2015']]))
        a10 = np.concatenate((np.mean(dic['2016'], axis=0), [avg['2016']]))

        a11 = np.concatenate((np.mean(dic['2017'], axis=0), [avg['2017']]))

        mean_last = np.concatenate((a8, a9, a10,a11))

        loss_validation=[]

        loss_train=[]

        for i in range(Max_it):

            # out_tr = get_sample(dic, A, avg,batch_size_tr, time_steps=5, num_features=316+100+16+4)
            out_tr = get_sample(dic, A, avg,batch_size_tr, time_steps=5, num_features=312+66+14+3+1)



            Ybar_tr=out_tr[:, :, -1].reshape(-1,5,1)

            # Batch_X_e = out_tr[:, :, 3:-1].reshape(-1,6*52+100+16+4)
            Batch_X_e = out_tr[:, :, 3:-1].reshape(-1,6*52+66+14) # 392


            Batch_X_e=np.expand_dims(Batch_X_e,axis=-1)
            Batch_Y = out_tr[:, -1, 2]
            Batch_Y = Batch_Y.reshape(len(Batch_Y), 1)

            Batch_Y_2 = out_tr[:, np.arange(0,4), 2]


            if i==60000:
                learning_rate=learning_rate/2
                print('learningrate1',learning_rate)
            elif i==120000:
                learning_rate = learning_rate/2
                print('learningrate2', learning_rate)
            elif i==180000:
                learning_rate = learning_rate/2
                print('learningrate3', learning_rate)



            # sess.run(train_op, feed_dict={ E_t1: Batch_X_e[:,0:52,:],E_t2: Batch_X_e[:,52*1:2*52,:],E_t3: Batch_X_e[:,52*2:3*52,:],
            #                                E_t4: Batch_X_e[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e[:,52*4:5*52,:],E_t6: Batch_X_e[:,52*5:52*6,:],
            #                                S_t1:Batch_X_e[:,312:322,:],S_t2:Batch_X_e[:,322:332,:],S_t3:Batch_X_e[:,332:342,:],
            #                                S_t4:Batch_X_e[:,342:352,:],S_t5:Batch_X_e[:,352:362,:],S_t6:Batch_X_e[:,362:372,:],
            #                                S_t7: Batch_X_e[:, 372:382, :], S_t8: Batch_X_e[:, 382:392, :],
            #                                S_t9: Batch_X_e[:, 392:402, :], S_t10: Batch_X_e[:, 402:412, :],P_t: Batch_X_e[:, 412:428, :],S_t_extra:Batch_X_e[:, 428:, :],
            #                                Ybar:Ybar_tr,Y_t: Batch_Y,Y_t_2: Batch_Y_2,is_training:True,lr:learning_rate,dropout:0.0})

            sess.run(train_op, feed_dict={ E_t1: Batch_X_e[:,0:52,:],E_t2: Batch_X_e[:,52*1:2*52,:],E_t3: Batch_X_e[:,52*2:3*52,:],
                                           E_t4: Batch_X_e[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e[:,52*4:5*52,:],E_t6: Batch_X_e[:,52*5:52*6,:],
                                           S_t1:Batch_X_e[:, 312:312 + 6*1, :],S_t2:Batch_X_e[:, 312 + 6*1:312 + 6*2 ,:],S_t3:Batch_X_e[:,312 + 6*2:312 + 6*3,:],
                                           S_t4:Batch_X_e[:,312 + 6*3:312 + 6*4,:],S_t5:Batch_X_e[:,312 + 6*4:312 + 6*5,:],S_t6:Batch_X_e[:,312 + 6*5:312 + 6*6,:],
                                           S_t7: Batch_X_e[:, 312 + 6*6:312 + 6*7, :], S_t8: Batch_X_e[:, 312 + 6*7:312 + 6*8, :],
                                           S_t9: Batch_X_e[:, 312 + 6*8:312 + 6*9, :], S_t10: Batch_X_e[:, 312 + 6*9:312 + 6*10, :],P_t: Batch_X_e[:, 378:378 + 14, :],
                                           S_t_extra:Batch_X_e[:, 312 + 6*10:312 + 6*11, :],
                                           Ybar:Ybar_tr, 
                                           Y_t: Batch_Y,
                                           Y_t_2: Batch_Y_2,
                                           is_training:True, 
                                           lr:learning_rate,dropout:0.6})


            if i%1000==0:

                # out_tr = get_sample(dic, A, avg, batch_size=1000, time_steps=5, num_features=316 + 100 + 16 + 4)
                out_tr = get_sample(dic, A, avg, batch_size=1000, time_steps=5, num_features=312+66+14+3+1)

                Ybar_tr = out_tr[:, :, -1].reshape(-1, 5, 1)

                # Batch_X_e = out_tr[:, :, 3:-1].reshape(-1, 6 * 52 + 100 + 16 + 4)
                Batch_X_e = out_tr[:, :, 3:-1].reshape(-1, 6 * 52 + 66 + 14)  # 394


                Batch_X_e = np.expand_dims(Batch_X_e, axis=-1)
                Batch_Y = out_tr[:, -1, 2]
                Batch_Y = Batch_Y.reshape(len(Batch_Y), 1)

                Batch_Y_2 = out_tr[:, np.arange(0, 4), 2]

                # out_te = get_sample_te(dic, mean_last, avg,np.sum(Index), time_steps=5, num_features=316+100+16+4)
                out_te = get_sample_te(dic, mean_last, avg,np.sum(Index), time_steps=5, num_features=312+66+14+4) # num_features should be 398
                print(out_te.shape)
                Ybar_te = out_te[:, :, -1].reshape(-1, 5, 1)
                # Batch_X_e_te = out_te[:, :, 3:-1].reshape(-1,6*52+100+16+4)
                Batch_X_e_te = out_te[:, :, 3:-1].reshape(-1,6*52+66+14)
                Batch_X_e_te = np.expand_dims(Batch_X_e_te, axis=-1)
                Batch_Y_te = out_te[:, -1, 2]
                Batch_Y_te = Batch_Y_te.reshape(len(Batch_Y_te), 1)
                Batch_Y_te2 = out_te[:, np.arange(0,4), 2]


                # rmse_tr,yhat1_tr,loss_tr = sess.run([RMSE,Yhat1,Tloss], feed_dict={ E_t1: Batch_X_e[:,0:52,:],E_t2: Batch_X_e[:,52*1:2*52,:],E_t3: Batch_X_e[:,52*2:3*52,:],
                #                            E_t4: Batch_X_e[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e[:,52*4:5*52,:],E_t6: Batch_X_e[:,52*5:52*6,:],
                #                            S_t1:Batch_X_e[:,312:322,:],S_t2:Batch_X_e[:,322:332,:],S_t3:Batch_X_e[:,332:342,:],
                #                            S_t4:Batch_X_e[:,342:352,:],S_t5:Batch_X_e[:,352:362,:],S_t6:Batch_X_e[:,362:372,:],
                #                            S_t7: Batch_X_e[:, 372:382, :], S_t8: Batch_X_e[:, 382:392, :],
                #                            S_t9: Batch_X_e[:, 392:402, :], S_t10: Batch_X_e[:, 402:412, :],P_t: Batch_X_e[:, 412:428, :],S_t_extra:Batch_X_e[:, 428:, :],
                #                            Ybar:Ybar_tr,Y_t: Batch_Y,Y_t_2: Batch_Y_2,is_training:True,lr:learning_rate,dropout:0.0})
                
                rmse_tr,yhat1_tr,loss_tr = sess.run([RMSE,Yhat1,Tloss], feed_dict={ E_t1: Batch_X_e[:,0:52,:],E_t2: Batch_X_e[:,52*1:2*52,:],E_t3: Batch_X_e[:,52*2:3*52,:],
                                           E_t4: Batch_X_e[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e[:,52*4:5*52,:],E_t6: Batch_X_e[:,52*5:52*6,:],
                                           S_t1:Batch_X_e[:, 312:312+6*1, :],S_t2:Batch_X_e[:, 312+6*1:312+6*2, :],
                                           S_t3:Batch_X_e[:, 312+6*2:312+6*3, :],
                                           S_t4:Batch_X_e[:, 312+6*3:312+6*4,:],
                                           S_t5:Batch_X_e[:, 312+6*4:312+6*5, :],
                                           S_t6:Batch_X_e[:, 312+6*5:312+6*6,:],
                                           S_t7: Batch_X_e[:, 312+6*6:312+6*7, :],
                                           S_t8: Batch_X_e[:, 312+6*7:312+6*8, :],
                                           S_t9: Batch_X_e[:, 312+6*8:312+6*9, :], 
                                           S_t10: Batch_X_e[:, 312+6*9:312+6*10, :],
                                           P_t: Batch_X_e[:, 378:378+14, :],
                                           S_t_extra:Batch_X_e[:, 312+6*10:312+6*11, :],
                                           Ybar:Ybar_tr,Y_t: Batch_Y,Y_t_2: Batch_Y_2,is_training:True,lr:learning_rate,dropout:0.6})


                rc_tr = np.corrcoef(np.squeeze(Batch_Y), np.squeeze(yhat1_tr))[0, 1]


                # rmse_te,yhat1_te,loss_val = sess.run([RMSE,Yhat1,Tloss], feed_dict={ E_t1: Batch_X_e_te[:,0:52,:],E_t2: Batch_X_e_te[:,52*1:2*52,:],E_t3: Batch_X_e_te[:,52*2:3*52,:],
                #                            E_t4: Batch_X_e_te[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e_te[:,52*4:5*52,:],E_t6: Batch_X_e_te[:,52*5:52*6,:],
                #                            S_t1:Batch_X_e_te[:,312:322,:],S_t2:Batch_X_e_te[:,322:332,:],S_t3:Batch_X_e_te[:,332:342,:],
                #                            S_t4:Batch_X_e_te[:,342:352,:],S_t5:Batch_X_e_te[:,352:362,:],S_t6:Batch_X_e_te[:,362:372,:],
                #                            S_t7: Batch_X_e_te[:, 372:382, :], S_t8: Batch_X_e_te[:, 382:392, :],
                #                            S_t9: Batch_X_e_te[:, 392:402, :], S_t10: Batch_X_e_te[:, 402:412, :],P_t: Batch_X_e_te[:, 412:428, :],S_t_extra:Batch_X_e_te[:, 428:, :],
                #                            Ybar:Ybar_te,Y_t: Batch_Y_te,Y_t_2: Batch_Y_te2,is_training:True,lr:learning_rate,dropout:0.0})
                rmse_te,yhat1_te,loss_val = sess.run([RMSE,Yhat1,Tloss], feed_dict={ E_t1: Batch_X_e_te[:,0:52,:],E_t2: Batch_X_e_te[:,52*1:2*52,:],E_t3: Batch_X_e_te[:,52*2:3*52,:],
                                           E_t4: Batch_X_e_te[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e_te[:,52*4:5*52,:],E_t6: Batch_X_e_te[:,52*5:52*6,:],
                                           S_t1:Batch_X_e_te[:, 312:312+6*1, :],
                                           S_t2:Batch_X_e_te[:, 312+6*1:312+6*2, :],
                                           S_t3:Batch_X_e_te[:, 312+6*2:312+6*3, :],
                                           S_t4:Batch_X_e_te[:, 312+6*3:312+6*4, :],
                                           S_t5:Batch_X_e_te[:, 312+6*4:312+6*5, :],
                                           S_t6:Batch_X_e_te[:, 312+6*5:312+6*6, :],
                                           S_t7: Batch_X_e_te[:, 312+6*6:312+6*7, :],
                                           S_t8: Batch_X_e_te[:, 312+6*7:312+6*8, :],
                                           S_t9: Batch_X_e_te[:, 312+6*8:312+6*9, :], 
                                           S_t10: Batch_X_e_te[:, 312+6*9:312+6*10, :],
                                           P_t: Batch_X_e_te[:, 378:378+16, :],
                                           S_t_extra:Batch_X_e_te[:, 312+6*10:312+6*11, :],
                                           Ybar:Ybar_te,Y_t: Batch_Y_te,Y_t_2: Batch_Y_te2,is_training:True,lr:learning_rate,dropout:0.6})


                rc=np.corrcoef(np.squeeze(Batch_Y_te),np.squeeze(yhat1_te))[0,1]
                loss_validation.append(loss_val)

                loss_train.append(loss_tr)

                print(loss_tr,loss_val)
                print("Iteration %d , The training RMSE is %f and Cor train is %f  and test RMSE is %f and Cor is %f " % (i, rmse_tr,rc_tr, rmse_te,rc))
                # Append the data to logs
                log_df = log_df.append({"Iteration":i, "training_RMSE":rmse_tr, "Cor_train": rc_tr, "test_RMSE":rmse_te, "Cor_is":rc }, ignore_index=True)




    out_te = get_sample_te(dic, mean_last, avg,np.sum(Index), time_steps=5, num_features=316+100+16+4)

    Batch_X_e_te = out_te[:, :, 3:-1].reshape(-1,6*52+100+16+4)
    Ybar_te = out_te[:, :, -1].reshape(-1, 5, 1)
    Batch_X_e_te = np.expand_dims(Batch_X_e_te, axis=-1)
    Batch_Y_te = out_te[:, -1, 2]
    Batch_Y_te = Batch_Y_te.reshape(len(Batch_Y_te), 1)
    Batch_Y_te2 = out_te[:, np.arange(0, 4), 2]

    rmse_te,yhat1 = sess.run([RMSE,Yhat1], feed_dict={ E_t1: Batch_X_e_te[:,0:52,:],E_t2: Batch_X_e_te[:,52*1:2*52,:],E_t3: Batch_X_e_te[:,52*2:3*52,:],
                                           E_t4: Batch_X_e_te[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e_te[:,52*4:5*52,:],E_t6: Batch_X_e_te[:,52*5:52*6,:],
                                           S_t1:Batch_X_e_te[:,312:322,:],S_t2:Batch_X_e_te[:,322:332,:],S_t3:Batch_X_e_te[:,332:342,:],
                                           S_t4:Batch_X_e_te[:,342:352,:],S_t5:Batch_X_e_te[:,352:362,:],S_t6:Batch_X_e_te[:,362:372,:],
                                           S_t7: Batch_X_e_te[:, 372:382, :], S_t8: Batch_X_e_te[:, 382:392, :],
                                           S_t9: Batch_X_e_te[:, 392:402, :], S_t10: Batch_X_e_te[:, 402:412, :],P_t: Batch_X_e_te[:, 412:428, :],S_t_extra:Batch_X_e_te[:, 428:, :],
                                           Ybar:Ybar_te,Y_t: Batch_Y_te,Y_t_2: Batch_Y_te2,is_training:True,lr:learning_rate,dropout:0.6})


    print("The training RMSE is %f  and test RMSE is %f " % (rmse_tr, rmse_te))
    t2=time.time()

    print('the training time was %f' %(round(t2-t1,2)))
    saver = tf.train.Saver()
    saver.save(sess, './model_corn', global_step=i)  # Saving the model

    return  rmse_tr,rmse_te,loss_train,loss_validation
        



# Initialise the logging
log_df = pd.DataFrame(columns={'Iteration', 'training_RMSE', 'Cor_train','test_RMSE', 'Cor_is'})


#### Read the dataset from the file
BigX = np.load('./soyabean_samples_npz.npz')
# Data Format order W(52*6) S(6*10) P(16)
X = BigX['data']

# Traning data consider as till 2017, testing would be 2018
X_tr = X[X[:, 1]<=2017]

# Remove the first three columns, loc_ID, year, yield
X_tr = X_tr[:, 3:]

#### Data Preprocessing
# subtract mean and devide by standerd deviation
M = np.mean(X_tr, axis=0, keepdims=True)
S = np.std(X_tr, axis=0, keepdims=True)
X[:, 3:] = (X[:, 3:]-M)/S

# Convert nan to num 0
X = np.nan_to_num(X)
# Remove the low yeild records
index_low_yield = X[:, 2]<5
print('low yield observations', np.sum(index_low_yield))
print(X[index_low_yield][:,1])
X = X[np.logical_not(index_low_yield)]
del BigX

# validation year
Index = X[:, 1]==2018

print('Yield Std %.2f and mean %.2f of test ' %(np.std(X[Index][:,2]), np.mean(X[Index][:,2])))

print("train data", np.sum(np.logical_not(Index)))
print("test data", np.sum(Index))

# Model Training Parameters
Max_it=100000      #150000 could also be used with early stopping
learning_rate=0.0003   # Learning rate
batch_size_tr=25  # traning batch size
le=0.0  # Weight of loss for prediction using times before final time steps
l=1.0    # Weight of loss for prediction using final time step
num_units=64  # Number of hidden units for LSTM celss
num_layers=1  # Number of layers of LSTM cell

try:
    rmse_tr,rmse_te,train_loss,validation_loss=main_program(X, Index,num_units,num_layers,Max_it, learning_rate, batch_size_tr,le,l)
except Exception as e:
# store the csv file
    print(e)
    traceback.print_exc()
    log_df.to_csv('./logs/CNN_DNN_drop_out_0_point_6.csv')
