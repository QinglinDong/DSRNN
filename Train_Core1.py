###################################################################################################

# A Recurrent Neural Network implementation using TensorFlow library.
# This project aims to explore the property of cell types and numbers.
# The datasets applied in this project are the t-fMRI signals of HCP 900. 
# Two types of rnn cells can be selected in this project, LSTM & GRU.

# Author: Han Wang
# Date: 2017/11/16

###################################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import math
import random
import numpy as np
import scipy.io as sio
import tensorflow as tf

import read_data as rd


if len(sys.argv) != 4:
    print("Usage: python", sys.argv[0], "[ TASK ] [ TYPE ] [ MAGIC POSTFIX ]")
    exit()
else:
    TASK = sys.argv[1]
    Type_Cell = sys.argv[2]
    Num_Layer_RNN   =   2   #int(sys.argv[3])
    Num_Cell_RNN    =   128 #int(sys.argv[4])
    Num_Cell_FC     =   32  #int(sys.argv[5])
    L_R     =   0.01    #float(sys.argv[6])
    Num_Cycle   =   2000    #int(sys.argv[7])
    S_Beta      =   0.001 ##float(sys.argv[3]) #0.01
    W_Lambda    =   0.001 #float(sys.argv[4]) #0.0001
    KEEP_PROB   =   0.9 #float(sys.argv[3])
    Magic = sys.argv[3]
    print('Task:', TASK);
    print('Cell Type:', Type_Cell);
    print('RNN Layer Number:', Num_Layer_RNN);
    print('RNN Cell Number:', Num_Cell_RNN);
    print('FC Cell Number:', Num_Cell_FC);
    print('Learning Rate:', L_R);
    print('Cycle:', Num_Cycle);
    print('Keep:', KEEP_PROB);
    print('LR:', L_R);
    print('Beta:', S_Beta);
    print('Lambda:', W_Lambda);
    print('Magic:', Magic);

CP_Path = 'FORMAL/' + 'Checkpoint/' + TASK + '/'
RSLT_Path = 'FORMAL/' + 'Result/' + TASK + '/'
SMM_Path = 'FORMAL/' + 'Summary/' + TASK + '/' + Magic + '/'
if not os.path.exists(CP_Path):
	os.makedirs(CP_Path)
if not os.path.exists(RSLT_Path):
	os.makedirs(RSLT_Path)
if not os.path.exists(SMM_Path):
	os.makedirs(SMM_Path)

""" Project Configure """
PROCESS = 'FORMAL'
# Data Parameters
Size_Input = rd.SAMPLENUM              # Size of input data: the vector size of input data at a moment
Size_Layer1 = Num_Cell_FC
Size_Batch = rd.BATCHSIZE[TASK][PROCESS]             # Size of batch: the number of samples who are processed parallelly
Size_Minibatch = rd.TASKLEN[TASK]
Size_Output = rd.EVENTNUM[TASK]            # Size of output: 13 channel signals from 13 distinctive brain regions
Num_Cycle = Num_Cycle + 1           # The total cycle of data

# Network Hyper-Parameters
Decay_Threshold = 150        # The cycle threshold to decay the learning rate
Cycle_Display = 20         # Display one time every several steps

TRN_D, TRN_L, VLD_D, VLD_L = rd.read_data_file(TASK, 'H5', PROCESS, Size_Batch)
Len_Series = TRN_D.shape[1]
assert Size_Batch == TRN_D.shape[0] == TRN_L.shape[0], "Batch size mismatches."
assert Size_Input == TRN_D.shape[2], "Input size mismatches."
assert Size_Output == TRN_L.shape[2], "Output size mismatches."
Num_Minibatch = Len_Series//Size_Minibatch
print('Minibatch Number: ', Num_Minibatch)
print('Minibatch Size: ', Size_Minibatch)

with tf.device('/gpu:0'):
    # Build Model
    LR = tf.placeholder(tf.float32, name='LR')
    SBETA = tf.placeholder(tf.float32, name='Beta')
    WLAMB = tf.placeholder(tf.float32, name='Lambda')
    Xin = tf.placeholder(tf.float32, [Size_Batch, None, Size_Input], name='Xin')
    Yin = tf.placeholder(tf.int8, [Size_Batch, None, Size_Output], name='Yin')

    Wi = tf.get_variable("Wi", [Size_Input, Size_Layer1], dtype=tf.float32)
    bi = tf.get_variable("bi", [Size_Layer1], dtype=tf.float32)

    Xin_rshp = tf.reshape(Xin, [-1, Size_Input])
    Oi_rshp = tf.sigmoid(tf.nn.xw_plus_b(Xin_rshp, Wi, bi))            # [ Size_Batch x Size_Seqs, Size_Output ]
    #print('Oi_rshp shape: ', tf.shape(Oi_rshp), Oi_rshp.shape)
    Oi = tf.reshape(Oi_rshp, (Size_Batch, -1, Size_Layer1))
    Oi_C = tf.reduce_mean(Oi, 0)
    #print('Oi shape: ', tf.shape(Oi), Oi.shape)

    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(Num_Cell_RNN, forget_bias=1.0, state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse)
    def gru_cell():
        return tf.contrib.rnn.GRUCell(Num_Cell_RNN)
    def basic_cell():
        return tf.contrib.rnn.BasicRNNCell(Num_Cell_RNN)

    if Type_Cell == 'LSTM':
        cell_unit = lstm_cell
    elif Type_Cell == 'GRU':
        cell_unit = gru_cell
    elif Type_Cell == 'BASIC':
        cell_unit = basic_cell
    else:
        print('Invalid Cell Type: %s'%(Type_Cell))

    def rnn_cell():
        return tf.contrib.rnn.DropoutWrapper(cell_unit(), input_keep_prob=KEEP_PROB)

    multi_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(Num_Layer_RNN)], state_is_tuple=True)

    zerostate = multi_cell.zero_state(Size_Batch, dtype=tf.float32)
    #Or, Sr = tf.nn.dynamic_rnn(multi_cell, O2, dtype=tf.float32, initial_state=zerostate)
    Or, Sr = tf.nn.dynamic_rnn(multi_cell, Oi, dtype=tf.float32, initial_state=zerostate)
    Or_flat = tf.reshape(Or, [-1, Num_Cell_RNN])    # [ Size_Batch x Size_Seqs, Num_Cell_RNN]

    Wo = tf.get_variable("Wo", [Num_Cell_RNN, Size_Output], dtype=tf.float32)
    bo = tf.get_variable("bo", [Size_Output], dtype=tf.float32)
    #Wo = tf.Variable(tf.zeros([Num_Cell_RNN, Size_Output]))
    #bo = tf.Variable(tf.zeros([Size_Output]))
    Om_flat= tf.nn.xw_plus_b(Or_flat, Wo, bo)            # [ Size_Batch x Size_Seqs, Size_Output ]
    Os_flat = tf.nn.softmax(Om_flat, name='Os')        # [ Size_Batch x Size_Seqs, Size_Output ]
    Yout_flat = tf.argmax(Os_flat, 1)                          # [ BATCHSIZE x SEQLEN ]
    Yin_flat = tf.reshape(Yin, [-1, Size_Output])    # [ Size_Batch x Size_Seqs, Size_Output]

    #
    # Define loss and optimizer
    #
    # ERROR Cost
    ECost_flat = tf.nn.softmax_cross_entropy_with_logits(logits=Om_flat, labels=Yin_flat)  # [ BATCHSIZE x SEQLEN ]
    #ECost = tf.reshape(loss_flat, [Size_Batch, -1])      # [ BATCHSIZE, SEQLEN ]
    ECost = tf.reduce_mean(ECost_flat, name='ECost')      # [ BATCHSIZE, SEQLEN ]

    # SPARSE Cost
#    RHO = tf.reduce_mean(Oi_rshp, 0)
#    KL = SRHO * tf.log(SRHO / RHO) + (1 - SRHO) * tf.log((1 - SRHO) / (1 - RHO))
    SCost = SBETA * tf.reduce_mean(tf.abs(Oi_rshp))

    # WEIGHT Cost
    WCost = WLAMB * tf.reduce_sum(tf.abs(Wi))

    loss = ECost + SCost + WCost
    #loss = ECost + WCost
    Train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    #
    # stats for display
    #
    Yout_max = tf.reshape(Yout_flat, [Size_Batch, -1], name="Yout_max")  # [ BATCHSIZE, SEQLEN ]
    Yin_max = tf.reshape(tf.argmax(Yin, 2), [Size_Batch, -1], name="Yin_max")    # [ Size_Batch, Size_Seqs]

    ACC = tf.reduce_mean(tf.cast(tf.equal(tf.cast(Yin_max, tf.uint8), tf.cast(Yout_max, tf.uint8)), tf.float32))
    BL = tf.reduce_mean(loss)

    bl_summary = tf.summary.scalar("Batch_Loss", BL)
    acc_summary = tf.summary.scalar("Batch_Accuracy", ACC)
    lr_summary = tf.summary.scalar('Learning_Rate', LR)
    Wo_hist = tf.summary.histogram("Wo", Wo)
    bo_hist = tf.summary.histogram("bo", bo)
    summary = tf.summary.merge([bl_summary, acc_summary, lr_summary, Wo_hist, bo_hist])

    # Init Tensorboard stuff. This will save Tensorboard information into a different
    # folder at each run named 'log/<timestamp>/'.
    #    timestamp = str(math.trunc(time.time()))
    summary_writer = tf.summary.FileWriter(SMM_Path)
    saver = tf.train.Saver(max_to_keep=10)

    # init
    init = tf.global_variables_initializer()
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False) 
    config.gpu_options.allow_growth = True 
    #config.log_device_placement = True 
    sess= tf.Session(config=config)

    sess.run(init)
    acc_log = []
    acc_show = 0
    loss_show = 0
    Train_Acc_Old = 0
    Train_Loss_Old = 0
    cvg_cnt = 0
    lr_cnt = 0
    for idx_cycle in range(Num_Cycle):
        cycle_acc = 0
        cycle_bl = 0
        for idx_mini in range(Num_Minibatch):
            in_state = sess.run(zerostate)  # initial zero input state (a tuple)
            stt_base = idx_mini*Size_Minibatch
            end_base = (idx_mini+1)*Size_Minibatch
            series_len = int(np.random.uniform(0.3, 0.5)*Size_Minibatch)
            offset = np.random.randint(Size_Minibatch - series_len)
            sttoff = stt_base + offset
            endoff = stt_base + offset + series_len
            Feed_Dict = {Xin: TRN_D[:, sttoff:endoff, :], Yin: TRN_L[:, sttoff:endoff, :], LR: L_R, SBETA: S_Beta, WLAMB: W_Lambda}
            _, t_out_state, t_bl, t_l, t_acc, t_yi_max, t_yo_max, ecost, scost, wcost, smm= sess.run([Train_op, Sr, BL, loss,  ACC, Yin_max, Yout_max, ECost, SCost, WCost, summary], feed_dict=Feed_Dict)

            summary_writer.add_summary(smm, idx_cycle * Num_Minibatch + idx_mini + 1)
            cycle_acc += t_acc
            cycle_bl += t_bl

        cycle_acc /= Num_Minibatch
        cycle_bl /= Num_Minibatch
        acc_show += cycle_acc
        loss_show += cycle_bl

        if 0 == idx_cycle%Cycle_Display:
            acc_show /= Cycle_Display
            loss_show /= Cycle_Display
            print('[********* Train cycle-%d *********] '%idx_cycle)
            print('LR-%f   LOSS-%f   ACC-%f  Loss-%f    Ecost-%f    Scost-%f    Wcost-%f'%(L_R, loss_show, acc_show, t_l, ecost, scost, wcost))
    #		print('stt: %d, end: %d, series: %d, offset: %d'%(sttoff, endoff, series_len, offset))

            lr_cnt += 1
            if lr_cnt >= Decay_Threshold/Cycle_Display:
                L_R /= 4
                cvg_cnt = 0
                lr_cnt = 0
            elif abs(acc_show - Train_Acc_Old) < 0.0001:
                cvg_cnt += 1
                print('Convergence Count: %d'%cvg_cnt)
                if 3 == cvg_cnt:
                    L_R /= 4
                    cvg_cnt = 0
                    lr_cnt = 0
            else:
                cvg_cnt = 0


            Train_Acc_Old = acc_show
            Train_Loss_Old = loss_show
            acc_show = 0
            loss_show = 0

            ##################VALID##################
            Feed_Dict = {Xin: VLD_D, Yin: VLD_L, LR: L_R, SBETA: S_Beta, WLAMB: W_Lambda}
            out_state, v_bl, v_acc, v_yi_max, v_yo_max = sess.run([Sr, BL, ACC, Yin_max, Yout_max], feed_dict=Feed_Dict)
            print('[Valid cycle-%d] '%idx_cycle + '**Loss-%f**  '%v_bl + '**Acc-%f**  '%v_acc)

            acc_log.append([L_R, Train_Acc_Old, v_acc, Train_Loss_Old, v_bl, ecost, scost, wcost])

            if 5e-6 > L_R:
                print('Train Process Over...')
                break

# Save Model	
model_file = os.path.join(CP_Path, 'Final_%s.ckpt'%Magic)
save_path = saver.save(sess, model_file)  
print("Model saved in file: %s" % save_path)
# Save Result   
result_file = os.path.join(RSLT_Path, 'Result_%s.mat'%Magic)
Acc_Log = np.array(acc_log)
sio.savemat(result_file, {'Beta': S_Beta, 'Lambda': W_Lambda, 'Acclog': Acc_Log.astype(np.double),'V_Yout': v_yo_max.astype(np.int8), 'V_Yin': v_yi_max.astype(np.int8)})
print("Result saved in file: %s" % result_file)
