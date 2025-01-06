import numpy as np
import matplotlib.pyplot as plt

def data_1Dto2D(data, Y=10, X=11):
	data_2D = np.zeros([Y, X])
	data_2D[0] = ( 	   	 0, 	   0,  	   	 0, 	   0, 	     0, 	   0, 	     0, 	   0,  	     0, 	   0, 	 	 0) 
	data_2D[1] = (	  	 0, 	   0,  	   	 0, 	   0, 	     0, 	   0, 	     0, 	   0, 	   	 0,   	   0, 	 	 0) 
	data_2D[2] = (	  	 0, 	   0,  data[1],  data[2],  data[3],  data[4],  data[5],  data[6],  data[7],   	   0, 	 	 0) 
	data_2D[3] = (	  	 0,   	   0,  data[8],  data[9], data[10], data[11], data[12], data[13], data[14],   	   0, 		 0) 
	data_2D[4] = (   	 0,   	   0, data[15], data[16], data[17], data[18], data[19], data[20], data[21],    	   0,   	 0) 
	data_2D[5] = (	  	 0,   	   0,        0, data[22], data[23], data[24], data[25], data[26],        0,   	   0, 		 0) 
	data_2D[6] = (	  	 0,   	   0,        0, data[27], data[28], data[29], data[30], data[31],    	 0,   	   0, 		 0) 
	data_2D[7] = (	  	 0, 	   0, 	 	 0,    	   0,   	 0,   	   0,   	 0,   	   0, 	   	 0, 	   0, 		 0) 
	data_2D[8] = (	  	 0, 	   0, 	 	 0, 	   0,        0,   	   0,   	 0, 	   0, 	   	 0, 	   0, 		 0) 
	data_2D[9] = (	  	 0, 	   0, 	 	 0, 	   0, 	     0,   	   0, 		 0, 	   0, 	   	 0, 	   0, 		 0) 
	return data_2D

def data_1Dto2D_1(data, Y=5, X=7):
	data_2D = np.zeros([Y, X])
	data_2D[0] = (data[1],  data[2],  data[3],  data[4],  data[5],  data[6],  data[7]) 
	data_2D[1] = (data[8],  data[9], data[10], data[11], data[12], data[13], data[14]) 
	data_2D[2] = (data[15], data[16], data[17], data[18], data[19], data[20], data[21]) 
	data_2D[3] = (        0, data[22], data[23], data[24], data[25], data[26],        0) 
	data_2D[4] = (        0, data[27], data[28], data[29], data[30], data[31],    	 0) 
	return data_2D

data = np.arange(32)
data_2D = data_1Dto2D(data)
plt.matshow(data_2D)
plt.show()