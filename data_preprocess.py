# -*- coding: utf-8 -*-

import os
import h5py  
import numpy as np 
import scipy.io as sio
import time

#taskset=["EMOTION",'GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']
taskset=['RELATIONAL','SOCIAL','WM']
for TASKName in taskset:
	print TASKName
	TASK = "tfMRI_"+TASKName+"_LR"
	signal_root_path = "/home/share/TmpData/Qinglin/HCP_4mm/"
	event_root_path = "/home/share/TmpData/Qinglin/HCP_Label/"
	h5_root_path = "/home/share/TmpData/Qinglin/HCP_4mm_RNN/HCP_4mm_h5"


	sub_dir_list = os.listdir("/home/share/TmpData/Qinglin/HCP_4mm/")
	sub_dir_list.sort()
	print("sub dir is: ", len(sub_dir_list))



	label_file = event_root_path  + TASKName+"_label.mat"
	#print("label file: "+label_file);
	label_data = sio.loadmat(label_file)
	#print("label loaded")



	time_start = time.time()
	for subject in sub_dir_list:

		if '150423' == subject or '547046' == subject:
			print('Subject %s: Relational data illegal, abandone...', subject)
			continue;

		signal_src_path = signal_root_path+ subject+"/MNINonLinear/Results/"+TASK+"/"+TASK+"_hp200_s4.feat/"


		if os.path.isdir(signal_src_path):
			signal_file = signal_src_path +"signal.mat"
			#print("signal file: "+signal_file);
			signal_data = sio.loadmat(signal_file)
		else:
			print("Subject-%s does not have signal data..."%(subject))


		h5_dst_path = os.path.join(h5_root_path, TASKName, subject)
		if not os.path.exists(h5_dst_path):
			os.makedirs(h5_dst_path)
		data_file = h5_dst_path + "/" + TASKName + "_data.h5"
		df = h5py.File(data_file,'w')   #创建一个h5文件，文件指针是f  
		df['data'] = signal_data['Signal'].astype(np.double)                 #将数据写入文件的主键data下面  
		df['label'] = label_data['Label'].astype(np.int8)           #将数据写入文件的主键labels下面  
		df.close()                           #关闭文件  


		#print("Subject--%s complete..."%(subject))

	time_end = time.time()
	#print("Time cost %d s..."%(time_end - time_start))
