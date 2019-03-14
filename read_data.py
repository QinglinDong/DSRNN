# -*- coding: utf-8 -*-
import os
import h5py
import math
import numpy as np  
import scipy.io as sio 
import time

h5_root_path = "/home/share/TmpData/Qinglin/HCP_4mm_RNN/HCP_4mm_h5/"
#mat_root_path = "/mnt/hbwc/Data/HCP-900/Mat_Data/"

TASKLEN = { \
	'WM'		:405, \
	'GAMBLING'	:253, \
	'MOTOR'		:284, \
	'LANGUAGE'	:316, \
	'SOCIAL'	:274, \
	'RELATIONAL'    :232, \
	'EMOTION'	:176 \
}

BATCHSIZE = { \
	'WM'		:{'EXPLORE':20, 'FORMAL':60, 'TEST':30}, \
	'GAMBLING'	:{'EXPLORE':20, 'FORMAL':60, 'TEST':30}, \
	'MOTOR'		:{'EXPLORE':20, 'FORMAL':60, 'TEST':30}, \
	'LANGUAGE'	:{'EXPLORE':20, 'FORMAL':60, 'TEST':30}, \
	'SOCIAL'	:{'EXPLORE':20, 'FORMAL':60, 'TEST':30}, \
	'RELATIONAL'    :{'EXPLORE':20, 'FORMAL':60, 'TEST':30}, \
	'EMOTION'	:{'EXPLORE':20, 'FORMAL':60, 'TEST':30} \
}

EVENTNUM = { \
	'WM'		:3, \
	'GAMBLING'	:3, \
	'MOTOR'		:6, \
	'LANGUAGE'	:3, \
	'SOCIAL'	:3, \
	'RELATIONAL'    :3, \
	'EMOTION'	:3 \
}

SAMPLENUM = 28549

EXPLORE_SUBJECT_NUM = { \
	'WM'		:{'TRAIN':40, 	'VALID':20}, \
	'GAMBLING'	:{'TRAIN':40, 	'VALID':20}, \
	'MOTOR'		:{'TRAIN':40, 	'VALID':20}, \
	'LANGUAGE'	:{'TRAIN':40, 	'VALID':20}, \
	'SOCIAL'	:{'TRAIN':40, 	'VALID':20}, \
	'RELATIONAL'    :{'TRAIN':40, 	'VALID':20}, \
	'EMOTION'	:{'TRAIN':40, 	'VALID':20} \
}

FORMAL_SUBJECT_NUM = { \
	'WM'		:{'TRAIN':240, 	'VALID':30}, \
	'GAMBLING'	:{'TRAIN':240, 	'VALID':30}, \
	'MOTOR'		:{'TRAIN':240, 	'VALID':30}, \
	'LANGUAGE'	:{'TRAIN':240, 	'VALID':30}, \
	'SOCIAL'	:{'TRAIN':240, 	'VALID':30}, \
	'RELATIONAL'    :{'TRAIN':240, 	'VALID':30}, \
	'EMOTION'	:{'TRAIN':300, 	'VALID':30} \
}
TEST_SUBJECT_NUM = { \
	'WM'		:{'TRAIN':180, 	'TEST':180}, \
	'GAMBLING'	:{'TRAIN':240, 	'TEST':240}, \
	'MOTOR'		:{'TRAIN':240, 	'TEST':240}, \
	'LANGUAGE'	:{'TRAIN':240, 	'TEST':240}, \
	'SOCIAL'	:{'TRAIN':240, 	'TEST':240}, \
	'RELATIONAL'    :{'TRAIN':240, 	'TEST':240}, \
	'EMOTION'	:{'TRAIN':300, 	'TEST':300} \
}
'''
TEST_SUBJECT_NUM = { \
	'WM'		:{'TRAIN':0, 	'TEST':240}, \
	'GAMBLING'	:{'TRAIN':0, 	'TEST':320}, \
	'MOTOR'		:{'TRAIN':0, 	'TEST':320}, \
	'LANGUAGE'	:{'TRAIN':0, 	'TEST':300}, \
	'SOCIAL'	:{'TRAIN':0, 	'TEST':320}, \
	'RELATIONAL'    :{'TRAIN':0, 	'TEST':320}, \
	'EMOTION'	:{'TRAIN':0, 	'TEST':400} \
}
'''

def read_data_file(Task_Type, File_Type, Use_Type, Batch_Size):
	if "H5" == File_Type:
		data_root_path = h5_root_path
		file_postfix = '.h5'
#	elif "MAT" == File_Type:
#		data_root_path = mat_root_path
#		file_postfix = '.mat'
	else:
		print("Invalid file type: %s"%(File_Type))
		exit()

	if "EXPLORE" == Use_Type:
		TRAIN_NUM = EXPLORE_SUBJECT_NUM[Task_Type]['TRAIN']
		VALID_NUM = EXPLORE_SUBJECT_NUM[Task_Type]['VALID']
	elif "FORMAL" == Use_Type:
		TRAIN_NUM = FORMAL_SUBJECT_NUM[Task_Type]['TRAIN']
		VALID_NUM = FORMAL_SUBJECT_NUM[Task_Type]['VALID']
	else:
		print("Invalid usage type: %s"%(Use_Type))
		exit()
	
	task_root_path = data_root_path + Task_Type
	sub_dir_list = os.listdir(task_root_path)
	sub_dir_list.sort()
#	print("sub dir is: ", len(sub_dir_list))
	
	time_start = time.time()
	for dataset in ['TRAIN', 'VALID']:
		if 'TRAIN' == dataset:
			sttpos = 0
			endpos = TRAIN_NUM
		elif 'VALID' == dataset:
			sttpos = TRAIN_NUM
			endpos = TRAIN_NUM + VALID_NUM
		else:
			print('Invalid Task...\n')
			break;

		datapool = []
		labelpool = []
		cnt = 0
		for subject in sub_dir_list[sttpos:endpos]:
			train_data_path = os.path.join(task_root_path, subject)
			if os.path.isdir(train_data_path):
				data_file = train_data_path + "/" + Task_Type +"_data" + file_postfix
				if "H5" == File_Type:
					h5fd = h5py.File(data_file,'r')   #打开h5文件  
					h5fd.keys()                            #可以查看所有的主键  
					data = h5fd['data'][:].astype(np.double)                    #取出主键为data的所有的键值  
					label = h5fd['label'][:].astype(np.int8)                    #取出主键为data的所有的键值  
					h5fd.close()
#				elif "MAT" == File_Type:
#					mat_data = sio.loadmat(data_file)
#					data = mat_data['data'].astype(np.double)
#					label = mat_data['label'].astype(np.int8)                    #取出主键为data的所有的键值  
				
				if len(data) != TASKLEN[Task_Type] or len(data[0]) != SAMPLENUM:
					print(subject)
					exit()

				if len(label) != TASKLEN[Task_Type] or len(label[0]) != EVENTNUM[Task_Type]:
					print('(%d %d) (%d %d)'%(len(label), TASKLEN[Task_Type], len(label[0]), EVENTNUM[Task_Type] ))
					print(subject)
					exit()

				datapool.append(data) 
				labelpool.append(label)
				#print("Subject--%s loaded..."%(subject));
				cnt += 1
				if 0 == cnt % 40:
					print("No.%d Subject--%s loaded..."%(cnt, subject));
			else:
				print("Subject-%s does not have signal data..."%(subject))

		if 'TRAIN' == dataset:
#			print('datapool size: (%d, %d, %d)'%(len(datapool), len(datapool[0]), len(datapool[0][0])))
			Train_Data = np.array(datapool, dtype=np.double)
#			print('Train Data: ', Train_Data)
			Train_Label = np.array(labelpool, dtype=np.int8)
#			print('Train Data size: (%d, %d, %d)'%(len(Train_Data), len(Train_Data[0]), len(Train_Data[0][0])))
#			print('Train Data type: ', Train_Data.dtype, 'Shape: ', Train_Data.shape)
#			print('Train Label type: ', Train_Label.dtype, 'Shape: ', Train_Label.shape)
			Train_Data = Train_Data.reshape(Batch_Size, -1, SAMPLENUM)
			Train_Label = Train_Label.reshape(Batch_Size, -1, EVENTNUM[Task_Type])
			print('Train Data type: ', Train_Data.dtype, 'Shape: ', Train_Data.shape)
			print('Train Label type: ', Train_Label.dtype, 'Shape: ', Train_Label.shape)
			#print('Train Data Test:')
			#for i in range(Batch_Size):
			#	print(Train_Data[i, 0:5, 0:10])
		elif 'VALID' == dataset:
			Valid_Data = []
			Valid_Label = []
#			print('datapool size: (%d, %d, %d)'%(len(datapool), len(datapool[0]), len(datapool[0][0])))
			for idx_valid in range(math.ceil(VALID_NUM/Batch_Size)*Batch_Size):
				Valid_Data.append(datapool[idx_valid%VALID_NUM])
				Valid_Label.append(labelpool[idx_valid%VALID_NUM])
#			print('Valid Data size: (%d, %d, %d)'%(len(Valid_Data), len(Valid_Data[0]), len(Valid_Data[0][0])))
			Valid_Data = np.array(Valid_Data, dtype=np.double)
			Valid_Label = np.array(Valid_Label, dtype=np.int8)
#			print('Valid Data size: (%d, %d, %d)'%(len(Valid_Data), len(Valid_Data[0]), len(Valid_Data[0][0])))
#			print('Valid Data type: ', Valid_Data.dtype, 'Shape: ', Valid_Data.shape)
#			print('Valid Label type: ', Valid_Label.dtype, 'Shape: ', Valid_Label.shape)
			Valid_Data = Valid_Data.reshape(Batch_Size, -1, SAMPLENUM)
			Valid_Label = Valid_Label.reshape(Batch_Size, -1, EVENTNUM[Task_Type])
			print('Valid Data type: ', Valid_Data.dtype, 'Shape: ', Valid_Data.shape)
			print('Valid Label type: ', Valid_Label.dtype, 'Shape: ', Valid_Label.shape)
	time_end = time.time()
	print('Time Cost: %.06f s'%(time_end - time_start))

	return Train_Data, Train_Label, Valid_Data, Valid_Label

def read_test_file(Task_Type, File_Type, Batch_Size):
	if "H5" == File_Type:
		data_root_path = h5_root_path
		file_postfix = '.h5'
#	elif "MAT" == File_Type:
#		data_root_path = mat_root_path
#		file_postfix = '.mat'
	else:
		print("Invalid file type: %s"%(File_Type))
		exit()

	TRAIN_NUM = TEST_SUBJECT_NUM[Task_Type]['TRAIN']
	TEST_NUM = TEST_SUBJECT_NUM[Task_Type]['TEST']
	
#	task_root_path = os.path.join(data_root_path, Task_Type)
	task_root_path = data_root_path
	sub_dir_list = os.listdir(task_root_path)
	sub_dir_list.sort()
#	print("sub dir is: ", len(sub_dir_list))
	
	time_start = time.time()
	sttpos = TRAIN_NUM 
	endpos = TRAIN_NUM + TEST_NUM

	datapool = []
	labelpool = []
	cnt = 0
	for subject in sub_dir_list[sttpos:endpos]:
		train_data_path = os.path.join(task_root_path, subject)
		if os.path.isdir(train_data_path):
			data_file = train_data_path + "/" + Task_Type +"_data" + file_postfix
			if "H5" == File_Type:
				h5fd = h5py.File(data_file,'r')   #打开h5文件  
				h5fd.keys()                            #可以查看所有的主键  
				data = h5fd['data'][:].astype(np.double)                    #取出主键为data的所有的键值  
				label = h5fd['label'][:].astype(np.int8)                    #取出主键为data的所有的键值  
				h5fd.close()
#			elif "MAT" == File_Type:
#				mat_data = sio.loadmat(data_file)
#				data = mat_data['data'].astype(np.double)
#				label = mat_data['label'].astype(np.int8)                    #取出主键为data的所有的键值  
			if len(data) != TASKLEN[Task_Type] or len(data[0]) != SAMPLENUM:
				print(subject)
				exit()

			if len(label) != TASKLEN[Task_Type] or len(label[0]) != EVENTNUM[Task_Type]:
				print(subject)
				exit()

			datapool.append(data) 
			labelpool.append(label)
			#print("Subject--%s loaded..."%(subject));
			cnt += 1
			if 0 == cnt % 40:
				print("No.%d Subject--%s loaded..."%(cnt, subject));
		else:
			print("Subject-%s does not have signal data..."%(subject))
	Test_Data = np.array(datapool)
	Test_Label = np.array(labelpool)
	Test_Data = Test_Data.reshape(Batch_Size, -1, SAMPLENUM)
	Test_Label = Test_Label.reshape(Batch_Size, -1, EVENTNUM[Task_Type])
	print('Test Data Shape: ', Test_Data.shape)
	print('Test Label Shape: ', Test_Label.shape)
	time_end = time.time()
	print('Time Cost: %.06f s'%(time_end - time_start))

	return Test_Data, Test_Label

def read_fake_data(Task_Type, Batch_Size):
	time_start = time.time()
	Ones = np.ones([SERIESSIZE[Task_Type], SAMPLENUM])
	Zeros = np.zeros([SERIESSIZE[Task_Type], SAMPLENUM])
	datapool = []
	for idx in range(int(Batch_Size/2)):
			datapool.append(Ones) 
	for idx in range(int(Batch_Size/2), Batch_Size):
			datapool.append(Zeros) 
	Fake_Data = np.array(datapool)
	Fake_Data = Fake_Data.reshape(Batch_Size, -1, SAMPLENUM)
	print('Fake Data Shape: ', Fake_Data.shape)
	time_end = time.time()
	print('Time Cost: %.06f s'%(time_end - time_start))

	return Fake_Data

def read_softmax_file(Task_Type, File_Path, File_Magic):
	time_start = time.time()
	numDIM = 512
	file_name = File_Path + '/Org_Train_' + File_Magic + '.mat'
	mat_data = sio.loadmat(file_name)
	Trace = mat_data['T_Trace'].astype(np.double)
	numBATCH = Trace.shape[0]
	numSUB = Trace.shape[3]
	T_Yin = mat_data['T_Yin'].astype(np.int8)
	
	datapool = []
	labelpool = []
	for batch in range(numBATCH):
		for subject in range(numSUB):
			data = Trace[batch, :, :, subject]
			datapool.append(data.T) 
			onehot = np.eye(EVENTNUM[Task_Type])[T_Yin[subject, :]]
			labelpool.append(onehot)

	Train_Data = np.array(datapool)
	Train_Label = np.array(labelpool)
	Train_Data = Train_Data.reshape(-1, numDIM)
	Train_Label = Train_Label.reshape(-1, EVENTNUM[Task_Type])
	print('Train Data Shape: ', Train_Data.shape)
	print('Train Label Shape: ', Train_Label.shape)

	file_name = File_Path + '/Org_Test_' + File_Magic + '.mat'
	mat_data = sio.loadmat(file_name)
	Trace = mat_data['T_Trace'].astype(np.double)
	numBATCH = Trace.shape[0]
	numSUB = Trace.shape[3]
	T_Yin = mat_data['T_Yin'].astype(np.int8)

	datapool = []
	labelpool = []
	for batch in range(numBATCH):
		for subject in range(numSUB):
			data = Trace[batch, :, :, subject]
			datapool.append(data.T) 
			onehot = np.eye(EVENTNUM[Task_Type])[T_Yin[subject, :]]
			labelpool.append(onehot)

	Test_Data = np.array(datapool)
	Test_Label = np.array(labelpool)
	Test_Data = Test_Data.reshape(-1, numDIM)
	Test_Label = Test_Label.reshape(-1, EVENTNUM[Task_Type])
	print('Test Data Shape: ', Test_Data.shape)
	print('Test Label Shape: ', Test_Label.shape)
	time_end = time.time()
	print('Time Cost: %.06f s'%(time_end - time_start))

	return Train_Data, Train_Label, Test_Data, Test_Label

