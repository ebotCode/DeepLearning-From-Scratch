import numpy as np 
import os 

dataset_dir = ".//dataset"
def load_planar_dataset(filename):

	with open(os.path.join(dataset_dir, filename),'r') as f:
		shape = [int(i) for i in f.readline().strip().split(",")]
		X = np.zeros((shape[0],shape[1]))
		Y = np.zeros((shape[0],shape[2]))
		counter = 0 
		while 1: 
			if counter >= 400:
				break 
			line = f.readline().strip().split(",")
			line = [float(i) for i in line]		
			X[counter,:] = np.array(line[:X.shape[1]])
			Y[counter,:] = np.array(line[X.shape[1]:(Y.shape[1] + X.shape[1])])

			counter += 1

		return X.T,Y.T



if __name__ == '__main__':
	x,y = load_planar_dataset("planar_dataset.csv")

	print(x.shape)
	print(y.shape)