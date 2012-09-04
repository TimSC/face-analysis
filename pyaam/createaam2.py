
import pickle, shelve
import numpy as np
import sklearn.linear_model as linear_model
import sklearn.ensemble as ensemble
import matplotlib.pyplot as plt

if __name__ == "__main__":

	
	#M = np.array([[0.2,0.1,-0.5],[1.0,0.,0.2],[-0.3,0.3,-0.5]])
	#B = np.random.rand(3,100)
	#A = np.dot(M,B)
	#Bi = np.linalg.pinv(B)
	#print np.dot(A,Bi)

	#pixelSubset = pickle.load(open("pixelSubset.dat","rb"))

	finas = ["perterbs0.dat", "perterbs1.dat", "perterbs2.dat", "perterbs3.dat"]
	perturbIndex, perturbCollect, diffInts = [], [], []

	#Get perturb index
	for fina in finas:
		s = shelve.open(fina)
		perturbIndex.extend([val[0] for val in s.values()])

	perturbIndex = np.array(perturbIndex)
	print perturbIndex.shape

	#Get perturb size
	for fina in finas:
		s = shelve.open(fina)
		perturbCollect.extend([val[1] for val in s.values()])

	perturbCollect = np.array(perturbCollect)
	print perturbCollect.shape

	#Get image intensity
	for fina in finas:
		s = shelve.open(fina)
		diffInts.extend([val[2] for val in s.values()])

	diffInts = np.array(diffInts)
	print diffInts.shape
	
	comp = np.where(perturbIndex == 0)
	model = linear_model.LinearRegression()
	model.fit(diffInts[comp,:], perturbCollect[comp,:])
	
	pred = model.predict(diffInts[comp,:])
	
	plt.plot(diffInts[comp,:], pred, '.')
	plt.show()
	
