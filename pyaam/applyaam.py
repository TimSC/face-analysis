import pickle, pcacombined, picseqloader
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	perturboutFiNa = "perturbs.dat"
	diffoutFiNa = "diffVals.dat"

	#Load combined model, annotations and images
	combinedModel = pickle.load(open("combinedmodel.dat","rb"))
	(pics, posData) = picseqloader.LoadTimDatabase()

	#Load prediction model
	predictors = pickle.load(open("predictors.dat", "rb"))

	
