import argparse
import os
import pyedflib
import numpy as np
import pandas as pd
import pickle
import time
from scipy.io import savemat, loadmat



np.random.seed(0)
def get_args():
	parser = argparse.ArgumentParser()
	
	hpstr = "set dataset directory"
	input_dir = "../raw_data/" # original repository: "/home/dadafly/datasets/EEG_motor_imagery/"

	parser.add_argument('-d', '--directory', default=input_dir, nargs='*', type=str, help=hpstr)


	hpstr = "set window size"
	parser.add_argument('-w', '--window', default=10, nargs='*', type=int, help=hpstr)

	hpstr = "set whether parallel"
	parser.add_argument('--parallel', default = False, action='store_true', help=hpstr) # add default

	hpstr = "set whether convert to 2D matrix"
	parser.add_argument('--convert', default = True, action='store_true', help=hpstr) # add default

	hpstr = "set whether segment data"
	parser.add_argument('--segment', default = True, action='store_true', help=hpstr) # add default

	hpstr = "set begin person"
	parser.add_argument('-b', '--begin', default=1, nargs='?', type=int, help=hpstr)

	hpstr = "set end person"
	parser.add_argument('-e', '--end', default=108, nargs='?', type=int, help=hpstr)

	hpstr = "set output directory"
	parser.add_argument('-o', '--output_dir', default=input_dir, nargs='*', help=hpstr)

	hpstr = "set whether store data"
	parser.add_argument('--set_store', default = True, action='store_true', help=hpstr) # add default

	args = parser.parse_args()
	return(args)
		   
def print_top(subject_id, window_size, parallel, convert, segment, output_dir, set_store):
	# 		   \n#### Author: Dalin Zhang	UNSW, Sydney	email: zhangdalin90@gmail.com #####	\
	print("######################## Six-class MI EEG data preprocess ########################	\
	   	   \n# subject ID:	%s \
		   \n# window size:		%d 	\
		   \n# parallel:	%s 	\
		   \n# convert:		%s 	\
		   \n# segment:		%s 	\
		   \n# output directory:	%s	\
		   \n# set store:		%s 	\
		   \n##############################################################################"% \
			(subject_id,	\
			window_size,	\
			parallel,       \
			convert,		\
			segment,		\
			output_dir,		\
			set_store))
	return None


def read_data(file_name):
	f = pyedflib.EdfReader(file_name)
	n = f.signals_in_file
	signal_labels = f.getSignalLabels()
	sigbufs = np.zeros((n, f.getNSamples()[0]))
	for i in np.arange(n):
	    sigbufs[i, :] = f.readSignal(i)
	sigbuf_transpose = np.transpose(sigbufs)
	signal = np.asarray(sigbuf_transpose)
	signal_labels = np.asarray(signal_labels)
	f._close()
	del f
	return signal, signal_labels

# def data_1Dto2D(data, Y=10, X=11): # comment out by TZ
# 	data_2D = np.zeros([Y, X])
# 	data_2D[0] = ( 	   	 0, 	   0,  	   	 0, 	   0, data[21], data[22], data[23], 	   0,  	     0, 	   0, 	 	 0) 
# 	data_2D[1] = (	  	 0, 	   0,  	   	 0, data[24], data[25], data[26], data[27], data[28], 	   	 0,   	   0, 	 	 0) 
# 	data_2D[2] = (	  	 0, data[29], data[30], data[31], data[32], data[33], data[34], data[35], data[36], data[37], 	 	 0) 
# 	data_2D[3] = (	  	 0, data[38],  data[0],  data[1],  data[2],  data[3],  data[4],  data[5],  data[6], data[39], 		 0) 
# 	data_2D[4] = (data[42], data[40],  data[7],  data[8],  data[9], data[10], data[11], data[12], data[13], data[41], data[43]) 
# 	data_2D[5] = (	  	 0, data[44], data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[45], 		 0) 
# 	data_2D[6] = (	  	 0, data[46], data[47], data[48], data[49], data[50], data[51], data[52], data[53], data[54], 		 0) 
# 	data_2D[7] = (	  	 0, 	   0, 	 	 0, data[55], data[56], data[57], data[58], data[59], 	   	 0, 	   0, 		 0) 
# 	data_2D[8] = (	  	 0, 	   0, 	 	 0, 	   0, data[60], data[61], data[62], 	   0, 	   	 0, 	   0, 		 0) 
# 	data_2D[9] = (	  	 0, 	   0, 	 	 0, 	   0, 	     0, data[63], 		 0, 	   0, 	   	 0, 	   0, 		 0) 
# 	return data_2D

def data_1Dto2D(data, Y=10, X=11): # add by TZ for huang's proj
	data_2D = np.zeros([Y, X])
	data_2D[0] = ( 	   	 0, 	   0,  	   	 0, 	   0, 	     0, 	   0, 	     0, 	   0,  	     0, 	   0, 	 	 0) 
	data_2D[1] = (	  	 0, 	   0,  	   	 0, 	   0, 	     0, 	   0, 	     0, 	   0, 	   	 0,   	   0, 	 	 0) 
	data_2D[2] = (	  	 0, 	   0,  data[0],  data[1],  data[2],  data[3],  data[4],  data[5],  data[6],   	   0, 	 	 0) 
	data_2D[3] = (	  	 0,   	   0,  data[7],  data[8], data[9], data[10], data[11], data[12], data[13],   	   0, 		 0) 
	data_2D[4] = (   	 0,   	   0, data[14], data[15], data[16], data[17], data[18], data[19], data[20],    	   0,   	 0) 
	data_2D[5] = (	  	 0,   	   0,        0, data[21], data[22], data[23], data[24], data[25],        0,   	   0, 		 0) 
	data_2D[6] = (	  	 0,   	   0,        0, data[26], data[27], data[28], data[29], data[30],    	 0,   	   0, 		 0) 
	data_2D[7] = (	  	 0, 	   0, 	 	 0,    	   0,   	 0,   	   0,   	 0,   	   0, 	   	 0, 	   0, 		 0) 
	data_2D[8] = (	  	 0, 	   0, 	 	 0, 	   0,        0,   	   0,   	 0, 	   0, 	   	 0, 	   0, 		 0) 
	data_2D[9] = (	  	 0, 	   0, 	 	 0, 	   0, 	     0,   	   0, 		 0, 	   0, 	   	 0, 	   0, 		 0) 
	return data_2D

def norm_dataset(dataset_1D):
	norm_dataset_1D = np.zeros([dataset_1D.shape[0], 31])
	for i in range(dataset_1D.shape[0]):
		norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
	return norm_dataset_1D

def feature_normalize(data):
	mean = data[data.nonzero()].mean()
	sigma = data[data.nonzero()].std()
	data_normalized = data
	data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
	return data_normalized

def dataset_1Dto2D(dataset_1D):
	dataset_2D = np.zeros([dataset_1D.shape[0], 10, 11])
	for i in range(dataset_1D.shape[0]):
		dataset_2D[i] = data_1Dto2D(dataset_1D[i])
	return dataset_2D

def norm_dataset_1Dto2D(dataset_1D):
	norm_dataset_2D = np.zeros([dataset_1D.shape[0], 10, 11])
	for i in range(dataset_1D.shape[0]):
		norm_dataset_2D[i] = feature_normalize(data_1Dto2D(dataset_1D[i]))
	return norm_dataset_2D

def windows(data, size):
	start = 0
	while ((start+size) < data.shape[0]):
		yield int(start), int(start + size)
		start += (size/2)


def segment_signal_without_transition(data, label, window_size):
    # start_time = time.time()  # Record the start time
    # total_steps = len(data) - window_size + 1  # Total number of possible windows
    # processed_steps = 0  # To track how many windows are processed

    for (start, end) in windows(data, window_size):

        # processed_steps += 1
        # # Progress and remaining time estimation
        # if processed_steps % 1000 == 0:  # Print progress every 1000 steps
        #     elapsed_time = time.time() - start_time
        #     progress = processed_steps / total_steps
        #     remaining_time = elapsed_time / progress - elapsed_time
        #     print(f"Progress: {progress:.2%}, "
        #           f"Elapsed Time: {elapsed_time:.2f}s, "
        #           f"Estimated Remaining Time: {remaining_time:.2f}s")

        # Original functionality
        if ((len(data[start:end]) == window_size) and (len(set(label[start:end])) == 1)):
            if start == 0:
                segments = data[start:end]
                labels = np.array(list(set(label[start:end])))
            else:
                segments = np.vstack([segments, data[start:end]])
                labels = np.append(labels, np.array(list(set(label[start:end]))))
    return segments, labels



def apply_mixup(all_data, all_label, parallel, convert, segment, window_size):

	# initial empty label arrays
	label_inter	= np.empty([0])
	# initial empty data arrays
	if (parallel == True):
		data_inter_cnn	= np.empty([0, window_size, 10, 11])
		data_inter_rnn	= np.empty([0, window_size, 64]) # 64-> 32??
	elif ((convert == False) and (segment == False)):
		data_inter	= np.empty([0, 64]) # 64-> 32??
	elif ((convert == False) and (segment == True)): 
		data_inter	= np.empty([0, window_size, 64]) # 64-> 32??
	elif ((convert == True) and (segment == False)): 
		data_inter	= np.empty([0, 10, 11])
	elif ((convert == True) and (segment == True)): # cascade
		data_inter	= np.empty([0, window_size, 10, 11])

	all_data = norm_dataset(all_data)

	step_size = 2401*10
	for start in range (0, all_data.shape[0], step_size):
		end = start + step_size
		data = all_data[start:end, :]
		label = all_label[start:end]
		print("Now processing until: "+str(end))
		if (parallel == True):
			# segment data
			data, label	= segment_signal_without_transition(data, label, window_size)
			# cnn data process
			data_cnn	= dataset_1Dto2D(data)
			data_cnn	= data_cnn.reshape(int(data_cnn.shape[0]/window_size), window_size, 10, 11)
			# rnn data process
			data_rnn	= data_cnn.reshape(int(data.shape[0]/window_size), window_size, 64)
		elif ((convert == False) and (segment == False)):
			pass
		elif ((convert == False) and (segment == True)):
			# segment data with sliding window 
			data, label	= segment_signal_without_transition(data, label, window_size)
			data		= data.reshape(int(data.shape[0]/window_size), window_size, 64)
		elif ((convert == True) and (segment == False)): 
			# convert 1D data to 2D
			data		= dataset_1Dto2D(data)
		elif ((convert == True) and (segment == True)): # cascade
			# convert 1D data to 2D
			data		= dataset_1Dto2D(data)
			# segment data with sliding window 
			data, label	= segment_signal_without_transition(data, label, window_size)
			data		= data.reshape(int(data.shape[0]/window_size), window_size, 10, 11)

		# append new data and label
		if (parallel == True):
			data_inter_cnn	= np.vstack([data_inter_cnn, data_cnn])
			data_inter_rnn	= np.vstack([data_inter_rnn, data_rnn])
			label_inter	= np.append(label_inter, label)
		else:
			data_inter	= np.vstack([data_inter, data])
			label_inter	= np.append(label_inter, label)


	# shuffle data
	index = np.array(range(0, len(label_inter)))
	np.random.shuffle(index)
	if (parallel==True):
		shuffled_data_cnn	= data_inter_cnn[index]
		shuffled_data_rnn	= data_inter_rnn[index]
		shuffled_label 	= label_inter[index]
	else:
		shuffled_data	= data_inter[index]
		shuffled_label 	= label_inter[index]
	return shuffled_data, shuffled_label


def get_raw_data(subject_id):
    data_dir = '../../training dataset/' # CHANGE THE PATH HERE!!!!
    # subject_id = 4
    raw_data_file = 'task_Sub' + str(subject_id)
    raw_data = loadmat(data_dir + raw_data_file)
    # file_name = 'sub6-2' # filename for saving model weight
    data = raw_data['data']
    # data = np.rollaxis(data, 2, 1) # use this line depending on the expected data shape
    label = raw_data['label'] - 2
    # label = np.rollaxis(label, 1, 0)
    # from sklearn.model_selection import train_test_split
    # data, x_test, label, y_test = train_test_split(x_subject, y_subject, test_size=0.33, random_state=42)
    data_reshaped = np.empty([data.shape[0]*data.shape[2], data.shape[1]])
    label_reshaped = np.empty([data.shape[0]*data.shape[2]])
    for trial in range (data.shape[0]):
        for data_point in range (data.shape[2]):
            reshaped_row = trial * data.shape[2] + data_point
            data_reshaped[reshaped_row, :] = data[trial, :, data_point]
            label_reshaped[reshaped_row] = label[0,trial]
	
    return data_reshaped, label_reshaped




# dataset_dir		=	"../../raw_data_CRNN/" # get_args().directory
window_size		=	10 # get_args().window
parallel		=	False # get_args().parallel
convert			=	True # get_args().convert
segment			=	True # get_args().segment
output_dir		=	"../../training dataset for CRNN/" # CHANGE THE PATH HERE!!!
set_store		=	True # get_args().set_store

for subject_id in range (1,8):
	data, label = get_raw_data(subject_id)
	print_top(subject_id, window_size, parallel, convert, segment, output_dir, set_store)
	shuffled_data, shuffled_label = apply_mixup(data, label, parallel, convert, segment, window_size)
	if (set_store == True):
		if (parallel == True):
			output_data_cnn = output_dir+"parallel_cnn_rnn/"+str(subject_id)+"_shuffle_cnn_dataset.pkl"
			output_data_rnn = output_dir+"parallel_cnn_rnn/"+str(subject_id)+"_shuffle_rnn_dataset.pkl"
			output_label= output_dir+"parallel_cnn_rnn/"+str(subject_id)+"_shuffle_labels.pkl"
		elif ((convert == False) and (segment == False)): # default to here
			output_data = output_dir+"1D_CNN/"+str(subject_id)+"_shuffle_dataset_1D.pkl"
			output_label= output_dir+"1D_CNN/"+str(subject_id)+"_shuffle_labels_1D.pkl"
		elif ((convert == False) and (segment == True)): 
			output_data = output_dir+"1D_CNN/window_1D/"+str(subject_id)+"_shuffle_dataset_1D_win_"+str(window_size)+".pkl"
			output_label= output_dir+"1D_CNN/window_1D/"+str(subject_id)+"_shuffle_labels_1D_win_"+str(window_size)+".pkl"
		elif ((convert == True) and (segment == False)): 
			output_data = output_dir+"2D_CNN/"+str(subject_id)+"_shuffle_dataset_2D.pkl"
			output_label= output_dir+"2D_CNN/"+str(subject_id)+"_shuffle_labels_2D.pkl"
		elif ((convert == True) and (segment == True)): # cascade
			print("FOR CASCADE!")
			output_data = output_dir+"3D_CNN/"+str(subject_id)+"_shuffle_dataset_3D_win_"+str(window_size)+".pkl"
			output_label= output_dir+"3D_CNN/"+str(subject_id)+"_shuffle_labels_3D_win_"+str(window_size)+".pkl"
		os.makedirs(os.path.dirname(output_data), exist_ok=True) # add by TZ
		if (parallel ==True):
			with open(output_data_cnn, "wb") as fp:
				pickle.dump(shuffled_data_cnn, fp, protocol=4) 
			with open(output_data_rnn, "wb") as fp:\
		        pickle.dump(shuffled_data_rnn, fp, protocol=4) 
			with open(output_label, "wb") as fp:
				pickle.dump(shuffled_label, fp)
		else: # default to here
			with open(output_data, "wb") as fp:
				pickle.dump(shuffled_data, fp, protocol=4)
			with open(output_label, "wb") as fp:
				pickle.dump(shuffled_label, fp)